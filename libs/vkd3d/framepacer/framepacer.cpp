#include "framepacer.h"
#include "framepacer_mode_low_latency.h"
#include "framepacer_mode_min_latency.h"
#include <algorithm>
#include <string>
#include <stdint.h>

namespace pacer {

#ifdef VKD3D_ENABLE_TEST_HOOKS
    std::atomic<bool> g_testLastTeardownWorkerStopped = { false };
#endif

    FramePacer::FramePacer( Device* device, uint64_t firstFrameId )
    : m_device(device), m_latencyMarkers(2), m_simulationLedger(firstFrameId),
      m_waitableDxgiSwapchain( device, m_latencyMarkers, m_frameSync,
        [this]() { return this->getAccountingState(); },
        [this](uint64_t frameId, time_point t, uint64_t accountingState) {
            return this->sleep(frameId, t, accountingState);
        } ),
      m_firstFrameId(firstFrameId) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
        g_testLastTeardownWorkerStopped.store(false, std::memory_order_release);
#endif
        // We'll default to LOW_LATENCY, which generally provides the best "input lag"
        // along with time consistency and often appears the smoothest too.
        // MAX_FRAME_LATENCY can have advantages in some games like God of War that provide inconsistent
        // cpu frametimes. Also, it's tuned for highest fps which can be relevant in benchmarks.
         FramePacerMode::Mode mode = FramePacerMode::LOW_LATENCY;
//        FramePacerMode::Mode mode = FramePacerMode::MIN_LATENCY;
        char env[8];

        // dxvk-low-latency is using default wait-latency 2 here, which pretty much can
        // max out fps in any game. For vkd3d however, we need to default it to 3
        // to prevent fps throttling in a lot of games. We want still give the user
        // the option to manually set it to 2 though.
        if (vkd3d_get_env_var("VKD3D_SWAPCHAIN_LATENCY_FRAMES", env, sizeof(env))) {
            unsigned long latency_override = strtoul(env, NULL, 0);
            if (latency_override >= 1 && latency_override <= 3)
                m_frameSync.m_waitLatency = latency_override;
        }

        telemetry::initialize(m_frameSync.m_waitLatency);
        m_device->m_telemetryId = telemetry::allocateDeviceId();

        // todo: add env var mode selection

        if (!m_device->m_calibratedDeviceTimestamps.canEnable() && mode != FramePacerMode::MIN_LATENCY) {
            WARN( "cannot enable low-latency frame pacing due to missing VK_KHR_calibrated_timestamps \n" );
            mode = FramePacerMode::MAX_FRAME_LATENCY;
        }

        switch (mode) {
        case FramePacerMode::MAX_FRAME_LATENCY:
            INFO( "Frame pace: max-frame-latency \n" );
            m_mode = std::make_unique<FramePacerMode>(FramePacerMode::MAX_FRAME_LATENCY, "max-frame-latency", &m_latencyMarkers, &m_frameSync, firstFrameId);
            break;

        case FramePacerMode::LOW_LATENCY:
            INFO( "Frame pace: low-latency \n" );
            INFO( "  m_frameSync.m_waitLatency = %i \n", m_frameSync.m_waitLatency.load() );
            m_device->m_calibratedDeviceTimestamps.enable();
            m_mode = std::make_unique<LowLatencyMode>(mode, &m_latencyMarkers, &m_frameSync, firstFrameId, m_device);
            break;

        case FramePacerMode::LOW_LATENCY_VRR:
        ////        Logger::info( "Frame pace: low-latency-vrr" );
        //        m_calibratedDeviceTimestamps.enable();
        //        m_mode = std::make_unique<LowLatencyMode>(mode, &m_latencyMarkers, &m_frameSync, options, firstFrameId, refreshRate);
        //        break;

        case FramePacerMode::MIN_LATENCY:
            INFO( "Frame pace: min-latency \n" );
            m_frameSync.m_waitLatency = 1;
            m_device->m_calibratedDeviceTimestamps.enable();
            m_mode = std::make_unique<MinLatencyMode>(mode, &m_latencyMarkers, &m_frameSync, firstFrameId);
            break;
        }

        m_frameSync.cpuFinished   = firstFrameId-1;
        m_frameSync.gpuFinished   = firstFrameId-1;
        m_frameSync.frameFinished = firstFrameId-1;

    }


    FramePacer::~FramePacer() {

        /* The worker can enter sleep(), which reads accounting state, the mode,
         * and marker storage. Stop it before destructor-body work or member
         * destruction can invalidate any of those objects. */
        m_waitableDxgiSwapchain.stop();
#ifdef VKD3D_ENABLE_TEST_HOOKS
        g_testLastTeardownWorkerStopped.store(
                m_waitableDxgiSwapchain.stopped(), std::memory_order_release);
#endif

        delete m_presentationStats.load();
        delete m_gpuBufferStats.load();

    }

    void FramePacer::setReflexMode(bool enable) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        uint64_t oldState = m_accountingState.load(std::memory_order_relaxed);

        if (isReflexAccountingState(oldState) == enable)
            return;

        uint64_t newState = (((oldState >> 1) + 1) << 1) | (enable ? 1 : 0);
        if (enable) {
            m_simulationLedger.activateEpoch(newState, m_firstFrameId);
        } else {
            m_simulationLedger.deactivateEpoch(oldState);
        }

        /* Timeline values are meaningful only together with accountingState.
         * Begin each legacy or Reflex generation from a neutral baseline so an
         * abandoned generation's numeric holes cannot be consumed by the next. */
        m_frameSync.cpuFinished.signal(m_firstFrameId - 1);
        m_frameSync.gpuFinished.signal(m_firstFrameId - 1);
        m_frameSync.frameFinished.signal(m_firstFrameId - 1);
        m_latencyMarkers.setGeneration(newState);
        m_frameMapping.setGeneration(newState);
        m_mode->resetAccountingGeneration(newState);
        m_accountingState.store(newState, std::memory_order_release);
        m_frameSync.gpuFinished.wake();
    }

    uint64_t FramePacer::beginReflexSimulation(uint64_t accountingEpoch,
            uint64_t externalReflexId, uint32_t threadId, time_point start) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        uint64_t simulationId = m_simulationLedger.beginSimulation(
                accountingEpoch, externalReflexId, threadId, start);
        if (!simulationId)
            return 0;
        m_latencyMarkers.updateMarkers(accountingEpoch, simulationId,
                [&](LatencyMarkers& markers) {
            if (markers.start == time_point{})
                markers.start = start;
        });
        return simulationId;
    }

    void FramePacer::beginReflexRenderSubmit(uint64_t accountingEpoch,
            uint64_t externalReflexId, uint32_t threadId, int32_t renderStart) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        m_simulationLedger.openRenderCapture(accountingEpoch,
                externalReflexId, threadId, renderStart);
    }

    void FramePacer::endReflexRenderSubmit(uint64_t accountingEpoch,
            uint64_t externalReflexId, uint32_t threadId, int32_t renderEnd) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        applySimulationProgress(m_simulationLedger.closeRenderCapture(
                accountingEpoch, externalReflexId, threadId, renderEnd));
    }

    PresentAttemptToken FramePacer::beginReflexPresent(uint64_t accountingEpoch,
            uint64_t simulationId, uint32_t threadId,
            const FrameInfo& frameInfo) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        PresentAttemptToken attemptToken;
        applySimulationProgress(m_simulationLedger.beginPresent(accountingEpoch,
                simulationId, threadId, frameInfo.renderStart,
                frameInfo.renderEnd, &attemptToken));
        if (attemptToken)
            m_uncapturedPresentAttempts[threadId] = attemptToken;
        return attemptToken;
    }

    void FramePacer::endReflexPresent(uint64_t accountingEpoch,
            uint64_t simulationId, uint32_t threadId) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        auto uncaptured = m_uncapturedPresentAttempts.find(threadId);
        if (uncaptured != m_uncapturedPresentAttempts.end() &&
                uncaptured->second.accountingEpoch == accountingEpoch &&
                uncaptured->second.simulationId == simulationId)
            m_uncapturedPresentAttempts.erase(uncaptured);
        applySimulationProgress(m_simulationLedger.endPresent(accountingEpoch,
                simulationId, threadId));
    }

    PresentAttemptToken FramePacer::capturePresentAttempt() {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        uint64_t accountingState = getAccountingState();
        uint32_t threadId = dxvk::this_thread::get_id();
        auto uncaptured = m_uncapturedPresentAttempts.find(threadId);

        if (uncaptured != m_uncapturedPresentAttempts.end()) {
            PresentAttemptToken attemptToken = uncaptured->second;
            m_uncapturedPresentAttempts.erase(uncaptured);
            return attemptToken;
        }

        if (isReflexAccountingState(accountingState))
            return {};

        return {accountingState, 0,
                m_nextLegacyPresentAttempt.fetch_add(1, std::memory_order_relaxed),
                threadId};
    }

    bool FramePacer::notifyReflexPresent(const PresentAttemptToken& attemptToken,
            void* swapchain, uint64_t sequence) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        if (getAccountingState() != attemptToken.accountingEpoch ||
                !isReflexAccountingState(attemptToken.accountingEpoch))
            return false;
        applySimulationProgress(m_simulationLedger.recordPresent(
                attemptToken, swapchain, sequence));
        return true;
    }

    void FramePacer::cancelReflexPresent(
            const PresentAttemptToken& attemptToken) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        applySimulationProgress(m_simulationLedger.cancelPresent(
                attemptToken));
    }

    void FramePacer::forceReflexPacingBypass() {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        if (isReflexAccountingState(getAccountingState()))
            m_simulationLedger.forcePacingBypass();
    }

    SubmitCompletionResult FramePacer::accountReflexCompletion(SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration, void* vulkanQueue,
            uint64_t vulkanGeneration, uint64_t gpuTimestamp,
            uint64_t gpuExecutionStart, bool gpuExecutionStartAvailable,
            SubmitTelemetryMetadata* telemetryMetadata) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        SubmitCompletionResult result;
        applySimulationProgress(m_simulationLedger.accountCompletion(submit,
                commandQueue, commandGeneration, vulkanQueue,
                vulkanGeneration, gpuTimestamp, gpuExecutionStart,
                gpuExecutionStartAvailable, &result, telemetryMetadata));
        return result;
    }

    void FramePacer::abandonReflexSubmit(SubmitRecord& submit,
            void* commandQueue, uint64_t commandGeneration, void* vulkanQueue,
            uint64_t vulkanGeneration) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        applySimulationProgress(m_simulationLedger.abandonSubmit(submit,
                commandQueue, commandGeneration, vulkanQueue,
                vulkanGeneration));
    }

    void FramePacer::retireCaptureLease(const CaptureToken& token,
            CaptureRetireReason reason) {
        std::lock_guard<dxvk::mutex> lock(m_progressMutex);
        applySimulationProgress(m_simulationLedger.retireCaptureLease(
                token, reason));
    }

    void FramePacer::applySimulationProgress(SimulationProgress&& progress) {
        if (!progress.accountingEpoch ||
                progress.accountingEpoch != m_accountingState.load(std::memory_order_acquire) ||
                !isReflexAccountingState(progress.accountingEpoch))
            return;
        for (const SimulationProgress::CpuCompletion& completion : progress.cpu) {
            m_frameMapping.registerMapping(progress.accountingEpoch,
                    completion.simulationId, completion.externalReflexId);
            m_latencyMarkers.updateMarkers(progress.accountingEpoch,
                    completion.simulationId, [&](LatencyMarkers& markers) {
                markers.start = completion.start;
                markers.renderStart = completion.renderStart;
                markers.renderEnd = completion.renderEnd;
                markers.cpuFinished = completion.cpuFinished;
            });
            m_frameSync.cpuFinished.signal(completion.simulationId);
        }

        if (!progress.gpu.empty())
            m_device->m_calibratedDeviceTimestamps.calibrate();

        for (const SimulationProgress::GpuCompletion& completion : progress.gpu) {
            time_point timestamp = m_device->m_calibratedDeviceTimestamps.getHostTimestamp(
                    completion.gpuTimestamp);

            m_latencyMarkers.updateMarkers(progress.accountingEpoch,
                    completion.simulationId, [&](LatencyMarkers& markers) {
                markers.gpuFinished = timestamp;
            });
            m_frameSync.gpuFinished.signal(completion.simulationId);
            if (telemetry::isEnabled()) {
            telemetry::Event event;
            event.type = telemetry::Type::GpuFrontier;
            event.deviceId = m_device->m_telemetryId;
            event.epochId = progress.accountingEpoch;
            event.simulationId = completion.simulationId;
            event.timestamp0 = completion.gpuTimestamp;
            event.timestamp1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    timestamp.time_since_epoch()).count();
            event.count0 = completion.publishedSubmits;
            event.count1 = completion.completedSubmits;
            telemetry::emit(event);
            }
            LatencyMarkers markers = m_latencyMarkers.getMarkers(
                    progress.accountingEpoch, completion.simulationId);
            if (markers.start != time_point{})
                m_mode->finishRender(progress.accountingEpoch,
                        completion.simulationId);
        }
    }

}
