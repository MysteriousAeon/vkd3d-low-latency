#pragma once
#include "vulkan_queue.h"
#include "device.h"
#include "nvapi_pacing_adapter.h"
#include "simulation_ledger.h"
#include "util/util_time.h"
#include "telemetry.h"

namespace pacer {

    class SubmitIterator;

    class CommandQueue {
        using time_point = dxvk::high_resolution_clock::time_point;
        using high_resulution_clock = dxvk::high_resolution_clock;
        friend class SubmitIterator;
    public:

        struct Properties {
            uint16_t id;
            D3D12_COMMAND_LIST_TYPE type;
            void* vkd3d_command_queue;
            void* vkd3d_queue;
        };

        const Properties m_properties;

        CommandQueue( Device* device, const Properties& properties, VulkanQueue* vulkanQueue )
            : m_properties(properties), m_vulkanQueue(vulkanQueue), m_device(device) {}

        ~CommandQueue() {}

        // we need to support multithreaded access here
        uint64_t notifySubmit() {
            uint64_t accountingEpoch = m_device->m_pacer->getAccountingState();
            if (FramePacer::isReflexAccountingState(accountingEpoch)) {
                CaptureLeaseAcquisition acquisition =
                        m_device->m_pacer->m_simulationLedger.acquireCaptureLease(
                                accountingEpoch);
                return acquisition.result == CaptureAcquireResult::Acquired
                        ? commitCapturedSubmit(acquisition.token) : 0;
            }

            return notifyLegacySubmit();
        }

        uint64_t notifyLegacySubmit() {
            uint64_t id = m_submitCounter.fetch_add(1, std::memory_order_acq_rel);
            uint16_t index = id % NUM_SUBMITS;
            time_point now = high_resulution_clock::now();

            testHook(TestHookPoint::ProducerBeforeSlot);
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (m_submitGenerations[index].load(std::memory_order_relaxed) > id)
                    return 0;
                resetSlotLocked(index, id, now);
            }
            testHook(TestHookPoint::ProducerAfterSlot);
            return id;
        }

        CaptureLeaseAcquisition acquireCaptureLease() {
            uint64_t accountingEpoch = m_device->m_pacer->getAccountingState();
            if (!FramePacer::isReflexAccountingState(accountingEpoch))
                return {CaptureAcquireResult::NotReflex, {}};
            return m_device->m_pacer->m_simulationLedger.acquireCaptureLease(
                    accountingEpoch);
        }

        uint64_t commitCapturedSubmit(const CaptureToken& token) {
            uint64_t id = m_submitCounter.fetch_add(1, std::memory_order_acq_rel);
            uint16_t index = id % NUM_SUBMITS;
            time_point now = high_resulution_clock::now();

            if (!m_device->m_pacer->m_simulationLedger.commitCapturedSubmit(
                    token, m_ledgerSubmits[index], this, id))
                return 0;

            bool published = false;
            testHook(TestHookPoint::ProducerBeforeSlot);
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (m_submitGenerations[index].load(std::memory_order_relaxed) <= id) {
                    resetSlotLocked(index, id, now);
                    published = true;
                }
            }
            testHook(TestHookPoint::ProducerAfterSlot);
            if (!published) {
                m_device->m_pacer->abandonReflexSubmit(m_ledgerSubmits[index],
                        this, id, nullptr, 0);
                return 0;
            }
            return id;
        }

        void retireCaptureLease(const CaptureToken& token,
                CaptureRetireReason reason) {
            m_device->m_pacer->retireCaptureLease(token, reason);
        }

        // this method is accessed single threaded
        bool notifyVulkanSubmit( uint64_t commandId, uint64_t vulkanId ) {
            uint16_t index = commandId % NUM_SUBMITS;
            void* rollbackVulkanQueue = nullptr;
            uint64_t rollbackVulkanGeneration = 0;
            const telemetry::QueueRole submissionRole =
                    m_telemetryRole.load(std::memory_order_relaxed);
            SubmitTelemetryMetadata capturedMetadata;
            SlotSnapshot publicationSnapshot;
            const bool publicationAvailable = telemetry::isEnabled() &&
                    snapshotSlot(commandId, &publicationSnapshot) &&
                    publicationSnapshot.submit != time_point{};
            SubmitPublicationResult publication =
                    m_device->m_pacer->m_simulationLedger.publishVulkanSubmit(
                            m_ledgerSubmits[index], this, commandId,
                            m_vulkanQueue, vulkanId, static_cast<uint8_t>(submissionRole),
                            publicationSnapshot.submit.time_since_epoch().count(),
                            publicationAvailable,
                            &rollbackVulkanQueue,
                            &rollbackVulkanGeneration, &capturedMetadata);
            const bool captured = publication != SubmitPublicationResult::NotCaptured;

            if (publication == SubmitPublicationResult::Rejected) {
                m_device->m_pacer->abandonReflexSubmit(m_ledgerSubmits[index],
                        this, commandId, rollbackVulkanQueue,
                        rollbackVulkanGeneration);
                return false;
            }

            bool published = false;
            testHook(TestHookPoint::VulkanBeforeSlot);
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (commandId &&
                        m_submitGenerations[index].load(std::memory_order_relaxed) == commandId) {
                    m_vulkanQueueIds[index].store(vulkanId, std::memory_order_relaxed);
                    if (!publicationAvailable)
                        publicationSnapshot.submit = m_submits[index].load(std::memory_order_relaxed);
                    publicationSnapshot.vulkanId = vulkanId;
                    publicationSnapshot.queueRole = submissionRole;
                    m_submitTelemetryRoles[index] = submissionRole;
                    published = true;
                }
            }
            testHook(TestHookPoint::VulkanAfterSlot);

            if (!published && captured)
                m_device->m_pacer->abandonReflexSubmit(m_ledgerSubmits[index],
                        this, commandId, rollbackVulkanQueue,
                        rollbackVulkanGeneration);
            if (published && telemetry::isEnabled() &&
                    (!captured || capturedMetadata.publicationAvailable)) {
                telemetry::Event event;
                event.type = telemetry::Type::Submit;
                event.deviceId = m_device->m_telemetryId;
                event.epochId = captured ? capturedMetadata.accountingEpoch
                        : m_device->m_pacer->getAccountingState();
                event.simulationId = captured ? capturedMetadata.simulationId : 0;
                event.captureGeneration = captured
                        ? capturedMetadata.captureGeneration : 0;
                event.id0 = uint64_t(m_properties.id) + 1;
                event.id1 = commandId;
                event.id2 = uint64_t(m_vulkanQueue->m_properties.id) + 1;
                event.timestamp0 = vulkanId;
                event.timestamp1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        publicationSnapshot.submit.time_since_epoch()).count();
                event.queueRole = publicationSnapshot.queueRole;
                event.captureClass = captured ? telemetry::CaptureClass::RenderCaptured
                        : telemetry::CaptureClass::Uncaptured;
                event.flags = 1; /* publication */
                telemetry::emit(event);
            }
            return published;
        }

        void setTelemetryQueueRole(telemetry::QueueRole role) {
            m_telemetryRole.store(role, std::memory_order_relaxed);
        }

        void notifyLegacyPresent( void* vkd3d_swapchain,
                uint64_t accountingEpoch ) {
            if (FramePacer::isReflexAccountingState(accountingEpoch))
                return;
            uint64_t commandId = m_submitCounter.load(std::memory_order_acquire) - 1;
            uint16_t index = commandId % NUM_SUBMITS;
            if (!m_device->m_pacer->finishCpu(accountingEpoch)) {
                m_device->m_pacer->m_waitableDxgiSwapchain.releaseSemaphore(
                        vkd3d_swapchain, 1);
                return;
            }
            uint64_t vulkanId = INVALID_ID;
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (commandId &&
                        m_submitGenerations[index].load(std::memory_order_relaxed) == commandId) {
                    m_presentEpochs[index].store(accountingEpoch, std::memory_order_relaxed);
                    m_presentFlags[index].store(true, std::memory_order_relaxed);
                    vulkanId = m_vulkanQueueIds[index].load(std::memory_order_relaxed);
                }
            }
            testHook(TestHookPoint::PresentAfterSnapshot);
            if (vulkanId == INVALID_ID) {
                m_device->m_pacer->m_waitableDxgiSwapchain.sleep(
                        vkd3d_swapchain, accountingEpoch );
                return;
            }

            uint64_t t_gpu = m_vulkanQueue->getGpuExecutionEnd(vulkanId);
            bool claimed = false;
            if (t_gpu) {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (m_submitGenerations[index].load(std::memory_order_relaxed) == commandId &&
                        m_vulkanQueueIds[index].load(std::memory_order_relaxed) == vulkanId &&
                        m_presentEpochs[index].load(std::memory_order_relaxed) == accountingEpoch &&
                        m_presentFlags[index].load(std::memory_order_relaxed)) {
                    m_presentFlags[index].store(false, std::memory_order_relaxed);
                    claimed = true;
                }
            }
            testHook(TestHookPoint::PresentAfterClaim);
            if (claimed)
                m_device->m_pacer->finishRender(t_gpu, accountingEpoch);

            m_device->m_pacer->m_waitableDxgiSwapchain.sleep(
                    vkd3d_swapchain, accountingEpoch );
        }

        void notifyVulkanGpuExecutionEnd( uint64_t commandId, uint64_t vulkanId,
                uint64_t gpuTimestamp ) {
            uint16_t index = commandId % NUM_SUBMITS;
            const bool telemetryEnabled = telemetry::isEnabled();

            auto completeLegacy = [&] {
                uint64_t presentEpoch = 0;
                if (gpuTimestamp) {
                    std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                    if (m_submitGenerations[index].load(std::memory_order_relaxed) == commandId &&
                            m_vulkanQueueIds[index].load(std::memory_order_relaxed) == vulkanId &&
                            m_presentFlags[index].load(std::memory_order_relaxed)) {
                        presentEpoch = m_presentEpochs[index].load(std::memory_order_relaxed);
                        m_presentFlags[index].store(false, std::memory_order_relaxed);
                    }
                }
                testHook(TestHookPoint::CompletionAfterSlot);
                if (presentEpoch)
                    m_device->m_pacer->finishRender(gpuTimestamp, presentEpoch);
            };

            /* Preserve the Layer-1 disabled path: an early legacy completion
             * only checks ledger ownership and then uses its slot-local path. */
            if (!telemetryEnabled) {
                if (m_device->m_pacer->m_simulationLedger.ownsSubmit(
                        m_ledgerSubmits[index], this, commandId))
                    m_device->m_pacer->accountReflexCompletion(m_ledgerSubmits[index],
                            this, commandId, m_vulkanQueue, vulkanId, gpuTimestamp,
                            0, false, nullptr);
                else
                    completeLegacy();
                return;
            }

            SubmitTelemetryMetadata capturedMetadata;
            uint64_t gpuStart = 0;
            const bool gpuStartAvailable =
                    m_vulkanQueue->snapshotGpuExecutionStart(vulkanId, &gpuStart) && gpuStart;
            testHook(TestHookPoint::CompletionBeforeAccounting);
            SubmitCompletionResult completion =
                    m_device->m_pacer->accountReflexCompletion(m_ledgerSubmits[index],
                            this, commandId, m_vulkanQueue, vulkanId, gpuTimestamp,
                            gpuStart, gpuStartAvailable,
                            &capturedMetadata);
            const bool captured = completion != SubmitCompletionResult::NotCaptured;
            if (completion == SubmitCompletionResult::Duplicate)
                return;

            SlotSnapshot commandSnapshot;
            bool telemetryCompletionClaimed = false;
            if (!captured) {
                telemetryCompletionClaimed = claimTelemetryCompletionSlot(
                        commandId, vulkanId, &commandSnapshot);
                completeLegacy();
            }

            testHook(TestHookPoint::CompletionAfterAccounting);

            /* Captured telemetry is emitted only by the callback whose ledger
             * completion was accepted. Its identity and role come from that
             * immutable ledger generation; a physical slot may already be gone. */
            const bool capturedTelemetryAvailable = captured &&
                    capturedMetadata.publicationAvailable &&
                    capturedMetadata.gpuExecutionStartAvailable;
            if (telemetryEnabled && (capturedTelemetryAvailable ||
                    (telemetryCompletionClaimed && gpuStartAvailable))) {
                gpuStart = captured ? capturedMetadata.gpuExecutionStart : gpuStart;
                const auto calibration =
                        m_device->m_pacer->getTelemetryCalibrationSnapshot();
                auto topHost = m_device->m_calibratedDeviceTimestamps.getHostTimestamp(
                        gpuStart, calibration);
                auto bottomHost = m_device->m_calibratedDeviceTimestamps.getHostTimestamp(
                        gpuTimestamp, calibration);
                telemetry::Event event;
                event.type = telemetry::Type::Submit;
                event.deviceId = m_device->m_telemetryId;
                event.epochId = captured ? capturedMetadata.accountingEpoch
                        : m_device->m_pacer->getAccountingState();
                event.simulationId = captured ? capturedMetadata.simulationId : 0;
                event.captureGeneration = captured
                        ? capturedMetadata.captureGeneration : 0;
                event.id0 = uint64_t(m_properties.id) + 1;
                event.id1 = commandId;
                event.id2 = uint64_t(m_vulkanQueue->m_properties.id) + 1;
                event.timestamp0 = vulkanId;
                event.timestamp1 = captured ? capturedMetadata.publicationCpuTimestampNs
                        : std::chrono::duration_cast<std::chrono::nanoseconds>(
                                commandSnapshot.submit.time_since_epoch()).count();
                event.timestamp2 = gpuStart;
                event.timestamp3 = gpuTimestamp;
                event.value0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        topHost.time_since_epoch()).count();
                event.value1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        bottomHost.time_since_epoch()).count();
                event.value2 = calibration.deviceTimestamp;
                event.value3 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        calibration.hostTimestamp.time_since_epoch()).count();
                event.calibrationDeviationNs = calibration.maxDeviation;
                event.queueRole = captured
                        ? static_cast<telemetry::QueueRole>(capturedMetadata.queueRole)
                        : commandSnapshot.queueRole;
                event.captureClass = captured ? telemetry::CaptureClass::RenderCaptured
                        : telemetry::CaptureClass::Uncaptured;
                event.flags = 2; /* completion */
                telemetry::emit(event);
            }
        }

        void notifySubmitFailed(uint64_t commandId, uint64_t vulkanId) {
            uint16_t index = commandId % NUM_SUBMITS;

            if (!commandId)
                return;
            if (m_device->m_pacer->m_simulationLedger.ownsSubmit(
                    m_ledgerSubmits[index], this, commandId))
                m_device->m_pacer->abandonReflexSubmit(m_ledgerSubmits[index],
                        this, commandId, m_vulkanQueue, vulkanId);
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (m_submitGenerations[index].load(std::memory_order_relaxed) == commandId)
                    m_presentFlags[index].store(false, std::memory_order_relaxed);
            }
        }

        VulkanQueue* m_vulkanQueue;
        Device* m_device;

        enum class TestHookPoint : uint32_t {
            ProducerBeforeSlot,
            ProducerAfterSlot,
            VulkanBeforeSlot,
            VulkanAfterSlot,
            PresentAfterSnapshot,
            PresentAfterClaim,
            CompletionAfterSlot,
            CompletionBeforeAccounting,
            CompletionAfterAccounting,
            IteratorAfterSnapshot,
        };
#ifdef VKD3D_ENABLE_TEST_HOOKS
        using TestHook = void (*)(uint32_t, void*);

        void setTestHook(TestHook hook, void* userdata) {
            std::unique_lock<dxvk::mutex> lock(m_testHookMutex);
            m_testHook = nullptr;
            m_testHookUserdata = nullptr;
            m_testHookCond.notify_all();
            m_testHookDrainWaiting = m_testHookInFlight != 0;
            if (m_testHookDrainWaiting)
                m_testHookCond.notify_all();
            m_testHookCond.wait(lock, [this] { return m_testHookInFlight == 0; });
            m_testHookDrainWaiting = false;
            m_testHook = hook;
            m_testHookUserdata = userdata;
            m_testHookCond.notify_all();
        }

        void pauseTestHookDispatch() {
            std::lock_guard<dxvk::mutex> lock(m_testHookMutex);
            m_testHookDispatchPaused = true;
            m_testHookDispatchArrived = false;
        }

        void waitTestHookDispatchArrived() const {
            std::unique_lock<dxvk::mutex> lock(m_testHookMutex);
            m_testHookCond.wait(lock, [this] { return m_testHookDispatchArrived; });
        }

        void waitTestHookCleared() const {
            std::unique_lock<dxvk::mutex> lock(m_testHookMutex);
            m_testHookCond.wait(lock, [this] { return m_testHook == nullptr; });
        }

        void waitTestHookDrain() const {
            std::unique_lock<dxvk::mutex> lock(m_testHookMutex);
            m_testHookCond.wait(lock, [this] { return m_testHookDrainWaiting; });
        }

        void resumeTestHookDispatch() {
            std::lock_guard<dxvk::mutex> lock(m_testHookMutex);
            m_testHookDispatchPaused = false;
            m_testHookCond.notify_all();
        }

        uint64_t testSubmitCount() const {
            return m_submitCounter.load(std::memory_order_acquire) - 1;
        }

        uint64_t testVulkanSubmitId( uint64_t commandId ) const {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            return m_submitGenerations[index].load(std::memory_order_relaxed) == commandId
                    ? m_vulkanQueueIds[index].load(std::memory_order_relaxed) : 0;
        }

        bool testPresentPending( uint64_t commandId ) const {
            uint16_t index = commandId % NUM_SUBMITS;
            if (m_device->m_pacer->m_simulationLedger.ownsSubmit(
                    m_ledgerSubmits[index], const_cast<CommandQueue*>(this), commandId))
                return m_device->m_pacer->m_simulationLedger.testSubmitPending(
                        m_ledgerSubmits[index], const_cast<CommandQueue*>(this), commandId);
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            return m_submitGenerations[index].load(std::memory_order_relaxed) == commandId &&
                    m_presentFlags[index].load(std::memory_order_relaxed);
        }

        uint64_t testSubmitSimulation( uint64_t commandId ) const {
            return m_device->m_pacer->m_simulationLedger.testSubmitSimulation(
                    m_ledgerSubmits[commandId % NUM_SUBMITS],
                    const_cast<CommandQueue*>(this), commandId);
        }

        bool testSubmitCompletionAccounted(uint64_t commandId) const {
            return m_device->m_pacer->m_simulationLedger.testSubmitCompletionAccounted(
                    m_ledgerSubmits[commandId % NUM_SUBMITS],
                    const_cast<CommandQueue*>(this), commandId);
        }

        void testSlotState(uint64_t commandId, uint64_t* generation,
                uint64_t* vulkanId, bool* present, uint64_t* presentEpoch,
                uint64_t* submitTimestamp, bool* hasTimestamp) const {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            *generation = m_submitGenerations[index].load(std::memory_order_relaxed);
            *vulkanId = m_vulkanQueueIds[index].load(std::memory_order_relaxed);
            *present = m_presentFlags[index].load(std::memory_order_relaxed);
            *presentEpoch = m_presentEpochs[index].load(std::memory_order_relaxed);
            time_point submit = m_submits[index].load(std::memory_order_relaxed);
            *submitTimestamp = submit.time_since_epoch().count();
            *hasTimestamp = submit != time_point{};
        }

        bool testSetSubmitTimestamp(uint64_t commandId, uint64_t timestamp) {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) != commandId)
                return false;
            m_submits[index].store(time_point(time_point::duration(timestamp)),
                    std::memory_order_relaxed);
            return true;
        }

        bool testSetPresentState(uint64_t commandId, uint64_t presentEpoch) {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) != commandId)
                return false;
            m_presentEpochs[index].store(presentEpoch, std::memory_order_relaxed);
            m_presentFlags[index].store(true, std::memory_order_relaxed);
            return true;
        }
#endif

    private:

        static constexpr uint64_t INVALID_ID = 0;
        static constexpr uint16_t NUM_SUBMITS = 2048;

        struct SlotSnapshot {
            time_point submit = {};
            uint64_t vulkanId = INVALID_ID;
            telemetry::QueueRole queueRole = telemetry::QueueRole::Unknown;
        };

        bool claimTelemetryCompletionSlot(uint64_t commandId, uint64_t vulkanId,
                SlotSnapshot* snapshot) {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) != commandId ||
                    m_vulkanQueueIds[index].load(std::memory_order_relaxed) != vulkanId ||
                    m_telemetryCompletionEmitted[index])
                return false;
            snapshot->submit = m_submits[index].load(std::memory_order_relaxed);
            snapshot->vulkanId = vulkanId;
            snapshot->queueRole = m_submitTelemetryRoles[index];
            m_telemetryCompletionEmitted[index] = true;
            return true;
        }

        bool snapshotSlot(uint64_t commandId, SlotSnapshot* snapshot) const {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) != commandId)
                return false;
            snapshot->submit = m_submits[index].load(std::memory_order_relaxed);
            snapshot->vulkanId = m_vulkanQueueIds[index].load(std::memory_order_relaxed);
            snapshot->queueRole = m_submitTelemetryRoles[index];
            return true;
        }

        void resetSlotLocked(uint16_t index, uint64_t id, time_point now) {
            m_submits[index].store(now, std::memory_order_relaxed);
            m_vulkanQueueIds[index].store(INVALID_ID, std::memory_order_relaxed);
            m_presentFlags[index].store(false, std::memory_order_relaxed);
            m_presentEpochs[index].store(0, std::memory_order_relaxed);
            m_telemetryCompletionEmitted[index] = false;
            m_submitTelemetryRoles[index] = telemetry::QueueRole::Unknown;
            m_submitGenerations[index].store(id, std::memory_order_relaxed);
        }

        void testHook(
                TestHookPoint point
        ) const {
#ifdef VKD3D_ENABLE_TEST_HOOKS
            TestHook hook;
            void* userdata;
            {
                std::unique_lock<dxvk::mutex> lock(m_testHookMutex);
                hook = m_testHook;
                userdata = m_testHookUserdata;
                if (hook) {
                    ++m_testHookInFlight;
                    if (m_testHookDispatchPaused) {
                        m_testHookDispatchArrived = true;
                        m_testHookCond.notify_all();
                        m_testHookCond.wait(lock, [this] {
                            return !m_testHookDispatchPaused;
                        });
                    }
                }
            }
            if (!hook)
                return;
            hook(static_cast<uint32_t>(point), userdata);
            {
                std::lock_guard<dxvk::mutex> lock(m_testHookMutex);
                if (!--m_testHookInFlight)
                    m_testHookCond.notify_all();
            }
#endif
        }

        std::array<std::atomic<time_point>, NUM_SUBMITS> m_submits;
        std::array<std::atomic<uint64_t>, NUM_SUBMITS>   m_submitGenerations = { };
        std::array<std::atomic<uint64_t>, NUM_SUBMITS>   m_vulkanQueueIds = { };
        std::array<std::atomic<bool>, NUM_SUBMITS>       m_presentFlags = { };
        std::array<std::atomic<uint64_t>, NUM_SUBMITS>   m_presentEpochs = { };
        std::array<SubmitRecord, NUM_SUBMITS>            m_ledgerSubmits = { };
        std::array<bool, NUM_SUBMITS>                    m_telemetryCompletionEmitted = { };
        std::array<telemetry::QueueRole, NUM_SUBMITS>    m_submitTelemetryRoles = { };
        mutable std::array<dxvk::mutex, NUM_SUBMITS>     m_slotMutexes;
        std::atomic<uint64_t> m_submitCounter = { 1 };
        std::atomic<telemetry::QueueRole> m_telemetryRole = { telemetry::QueueRole::Normal };
#ifdef VKD3D_ENABLE_TEST_HOOKS
        mutable dxvk::mutex m_testHookMutex;
        mutable dxvk::condition_variable m_testHookCond;
        mutable TestHook m_testHook = nullptr;
        mutable void* m_testHookUserdata = nullptr;
        mutable uint32_t m_testHookInFlight = 0;
        mutable bool m_testHookDrainWaiting = false;
        mutable bool m_testHookDispatchPaused = false;
        mutable bool m_testHookDispatchArrived = false;
#endif

    };

}
