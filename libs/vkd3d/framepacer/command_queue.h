#pragma once
#include "vulkan_queue.h"
#include "device.h"
#include "nvapi_pacing_adapter.h"
#include "simulation_ledger.h"
#include "util/util_time.h"

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
            const bool captured = m_device->m_pacer->m_simulationLedger.ownsSubmit(
                    m_ledgerSubmits[index], this, commandId);
            void* rollbackVulkanQueue = nullptr;
            uint64_t rollbackVulkanGeneration = 0;

            if (captured && !m_device->m_pacer->m_simulationLedger.publishVulkanSubmit(
                    m_ledgerSubmits[index], this, commandId,
                    m_vulkanQueue, vulkanId, &rollbackVulkanQueue,
                    &rollbackVulkanGeneration)) {
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
                    published = true;
                }
            }
            testHook(TestHookPoint::VulkanAfterSlot);

            if (!published && captured)
                m_device->m_pacer->abandonReflexSubmit(m_ledgerSubmits[index],
                        this, commandId, rollbackVulkanQueue,
                        rollbackVulkanGeneration);
            return published;
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

            if (m_device->m_pacer->m_simulationLedger.ownsSubmit(
                    m_ledgerSubmits[index], this, commandId)) {
                m_device->m_pacer->accountReflexCompletion(m_ledgerSubmits[index],
                        this, commandId, m_vulkanQueue, vulkanId, gpuTimestamp);
                return;
            }

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
        };

        bool snapshotSlot(uint64_t commandId, SlotSnapshot* snapshot) const {
            uint16_t index = commandId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) != commandId)
                return false;
            snapshot->submit = m_submits[index].load(std::memory_order_relaxed);
            snapshot->vulkanId = m_vulkanQueueIds[index].load(std::memory_order_relaxed);
            return true;
        }

        void resetSlotLocked(uint16_t index, uint64_t id, time_point now) {
            m_submits[index].store(now, std::memory_order_relaxed);
            m_vulkanQueueIds[index].store(INVALID_ID, std::memory_order_relaxed);
            m_presentFlags[index].store(false, std::memory_order_relaxed);
            m_presentEpochs[index].store(0, std::memory_order_relaxed);
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
        mutable std::array<dxvk::mutex, NUM_SUBMITS>     m_slotMutexes;
        std::atomic<uint64_t> m_submitCounter = { 1 };
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
