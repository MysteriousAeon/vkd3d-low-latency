#pragma once

#include "framepacer_bridge.h"
#include "framepacer_mode.h"
#include "frame_sync.h"
#include "device.h"
#include "frame_mapping.h"
#include "waitable_dxgi_swapchain.h"
#include "latency_markers.h"
#include "simulation_ledger.h"
#include "latency_stats.h"
#include "jitter_stats.h"
#include "util/sync/sync_ringbuffer_allocator.h"
#include "util/util_log.h"
#include <unordered_map>
#ifdef VKD3D_ENABLE_TEST_HOOKS
#include "vkd3d_test_hooks.h"
#endif


/* \brief Frame pacer interface managing the CPU - GPU synchronization.
 *
 * GPUs render frames asynchronously to the game's and dxvk's CPU-side work
 * in order to improve fps-throughput. Aligning the cpu work to chosen time-
 * points allows to tune certain characteristics of the video presentation,
 * like smoothness and latency.
 */

namespace pacer {

    class CommandQueue;
    class VulkanQueue;


    class FramePacer {

        using microseconds = std::chrono::microseconds;
        using high_resolution_clock = dxvk::high_resolution_clock;
        using time_point   = dxvk::high_resolution_clock::time_point;

    public:

        struct FrameInfo {
            uint64_t externalId;
            uint64_t simulationId;
            uint64_t accountingEpoch;
            time_point start_t;
            int32_t renderStart;
            int32_t renderEnd;
        };

        FramePacer( Device* device, uint64_t firstFrameId );
        ~FramePacer();

        bool sleep( uint64_t frameId, time_point lastSimulationStart,
                uint64_t expectedAccountingState = 0 ) {
            uint64_t accountingState = expectedAccountingState
                    ? expectedAccountingState : getAccountingState();
            if (getAccountingState() != accountingState)
                return false;
#ifdef VKD3D_ENABLE_TEST_HOOKS
            m_testSleepDecision.call_count++;
            m_testSleepEntryCount.fetch_add(1, std::memory_order_release);
            m_testSleepDecision.entry_accounting_state = accountingState;
            m_testSleepDecision.exit_accounting_state = accountingState;
            m_testSleepDecision.frame_id = frameId;
            m_testSleepDecision.cpu_finished = m_frameSync.cpuFinished.load();
            m_testSleepDecision.gpu_finished = m_frameSync.gpuFinished.load();
            m_testSleepDecision.would_start_frame = false;
#endif
            _INFO( "sleep - frameId: %" PRIu64 ", m_frameSync.cpuFinished: %"
                PRIu64 " m_frameSync.gpuFinished: %" PRIu64 " \n",
                frameId, m_frameSync.cpuFinished.load(), m_frameSync.gpuFinished.load() );

            if (isReflexAccountingState(accountingState)
                    && m_simulationLedger.shouldBypassPacing()) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
                m_testSleepDecision.exit_accounting_state = getAccountingState();
#endif
                _WARN( "bypassing Reflex pacing after simulation tracking failure\n" );
                return true;
            }

            // wait for finished rendering of a previous frame, typically the one before last
            uint64_t waitId = frameId-m_frameSync.m_waitLatency;
#ifdef VKD3D_ENABLE_TEST_HOOKS
            m_testSleepDecision.wait_id = waitId;
            m_testSleepDecision.wait_satisfied =
                    m_testSleepDecision.gpu_finished >= waitId;
            if (m_testSleepBypass) {
                std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
                if (getAccountingState() != accountingState)
                    return false;
                m_testSleepDecision.exit_accounting_state = getAccountingState();
                m_testSleepDecision.would_start_frame = m_testSleepDecision.wait_satisfied;
                return true;
            }
#endif
            if (!m_frameSync.gpuFinished.wait(waitId, 200,
                    m_accountingState, accountingState)) {
                if (getAccountingState() != accountingState) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
                    m_testSleepDecision.exit_accounting_state = getAccountingState();
#endif
                    return false;
                }
                WARN( "timeout on waiting for gpu finish id %" PRIu64 " reached, resulting in stutter \n", waitId );
                return true;
            }
            // potentially wait some more if the cpu gets too much ahead
            std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
            if (getAccountingState() != accountingState)
                return false;
#ifdef VKD3D_ENABLE_TEST_HOOKS
            m_testSleepDecision.exit_accounting_state = getAccountingState();
            m_testSleepDecision.wait_satisfied = true;
#endif
            m_mode->startFrame(accountingState, frameId, lastSimulationStart);
#ifdef VKD3D_ENABLE_TEST_HOOKS
            m_testSleepDecision.would_start_frame = true;
#endif
            return true;
        }

        template<typename Commit>
        bool withValidAccountingState(uint64_t accountingEpoch, Commit&& commit) {
            std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
            if (m_accountingState.load(std::memory_order_acquire) != accountingEpoch ||
                    !isReflexAccountingState(accountingEpoch))
                return false;
            commit();
            return true;
        }

        bool finishCpu(uint64_t accountingEpoch) {
            std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
            if (m_accountingState.load(std::memory_order_acquire) != accountingEpoch ||
                    isReflexAccountingState(accountingEpoch))
                return false;
            uint64_t cpuId = 1 + m_frameSync.cpuFinished++;

            uint64_t newId = cpuId + 1;
            m_frameMapping.registerMapping(accountingEpoch, newId, 0);
            m_latencyMarkers.updateMarkers(accountingEpoch, newId + 1,
                    [](LatencyMarkers&) {});

            _INFO( "reset markers for frame %" PRIu64 " \n", newId+1 );

            m_latencyMarkers.updateMarkers(accountingEpoch, cpuId,
                    [&](LatencyMarkers& markers) {
                markers.cpuFinished = high_resolution_clock::now();
            });
            return true;
        }

        void finishRender(uint64_t gpuDeviceTimestamp, uint64_t accountingEpoch) {
            std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
            if (m_accountingState.load(std::memory_order_acquire) != accountingEpoch ||
                    isReflexAccountingState(accountingEpoch))
                return;
            // this is incredibly rare that finish render is triggered
            // for two consecutive frames from two threads simultaneously
            // but we've had that happen when logging was enabled
            std::lock_guard<dxvk::mutex> lock(m_finishMutex);
            m_device->m_calibratedDeviceTimestamps.calibrate();

            uint64_t gpuId = 1 + m_frameSync.gpuFinished;
            time_point t = m_device->m_calibratedDeviceTimestamps.getHostTimestamp(gpuDeviceTimestamp);
            m_latencyMarkers.updateMarkers(accountingEpoch, gpuId,
                    [&](LatencyMarkers& markers) { markers.gpuFinished = t; });

            m_frameSync.gpuFinished++;

            _INFO( "set gpuFinished to %" PRIu64 " \n", m_frameSync.gpuFinished.load() );

            LatencyMarkers markers = m_latencyMarkers.getMarkers(accountingEpoch, gpuId);
            if (markers.start == time_point{}) {
                _INFO( "m->start not set, skipping frame analysis \n" );
                return;
            }

            int32_t latency = std::chrono::duration_cast<microseconds> ( t - markers.start ).count();
            _INFO( "latency = %" PRIi32 "\n", latency );
//            auto now = high_resolution_clock::now();
//            int32_t t_vs_now = std::chrono::duration_cast<microseconds> ( t - now ).count();
//            INFO( "t_vs_now = %" PRIi32 "\n", t_vs_now );
//            INFO( "t = %" PRIu64 " \n", gpuDeviceTimestamp );
//            m_latencyAverage.push( m->gpuFinished );
            m_mode->finishRender(accountingEpoch, gpuId);
        }

        void setReflexMode(bool enable);
        uint64_t getAccountingState() const {
            return m_accountingState.load(std::memory_order_acquire);
        }
        uint64_t getReflexEpoch() const {
            uint64_t state = getAccountingState();
            return isReflexAccountingState(state) ? state : 0;
        }
        static bool isReflexAccountingState(uint64_t state) {
            return state & 1;
        }

        uint64_t beginReflexSimulation(uint64_t accountingEpoch,
                uint64_t externalReflexId, uint32_t threadId, time_point start);
        void beginReflexRenderSubmit(uint64_t accountingEpoch,
                uint64_t externalReflexId, uint32_t threadId,
                int32_t renderStart);
        void endReflexRenderSubmit(uint64_t accountingEpoch,
                uint64_t externalReflexId, uint32_t threadId,
                int32_t renderEnd);
        PresentAttemptToken beginReflexPresent(uint64_t accountingEpoch,
                uint64_t simulationId, uint32_t threadId,
                const FrameInfo& frameInfo);
        void endReflexPresent(uint64_t accountingEpoch,
                uint64_t simulationId, uint32_t threadId);
        PresentAttemptToken capturePresentAttempt();
        bool notifyReflexPresent(const PresentAttemptToken& attemptToken,
                void* swapchain, uint64_t sequence);
        void cancelReflexPresent(const PresentAttemptToken& attemptToken);
        void forceReflexPacingBypass();
        void accountReflexCompletion(SubmitRecord& submit, void* commandQueue,
                uint64_t commandGeneration, void* vulkanQueue,
                uint64_t vulkanGeneration, uint64_t gpuTimestamp);
        void abandonReflexSubmit(SubmitRecord& submit, void* commandQueue,
                uint64_t commandGeneration, void* vulkanQueue,
                uint64_t vulkanGeneration);
        void retireCaptureLease(const CaptureToken& token,
                CaptureRetireReason reason);
        void applySimulationProgress(SimulationProgress&& progress);

        // todo: implement
        void notifyGpuPresentEnd( uint64_t frameId ) {
            // // the frame has been displayed to the screen
            // m_latencyMarkers.registerFrameEnd(frameId);
            // m_mode->endFrame(frameId);
            // m_frameSync.frameFinished.signal(frameId);
            //
            // trackStats(frameId);
        }

        FramePacerMode::Mode getMode() const {
            return m_mode->m_mode;
        }

        FramePacerMode* getFramePacerMode() {
            return m_mode.get();
        }

#ifdef VKD3D_ENABLE_TEST_HOOKS
        void testSetPrediction(uint64_t frameId, int32_t optimizedGpuTime) {
            std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
            m_mode->testSetPrediction(frameId, optimizedGpuTime);
        }

        void testGetPredictionState(vkd3d_test_prediction_state *state) {
            std::lock_guard<dxvk::mutex> progressLock(m_progressMutex);
            state->accounting_state = getAccountingState();
            state->finished_frame_id = m_mode->testGetPredictionFrame();
            state->predicted_gpu_time = m_mode->testGetPredictionGpuTime();
        }
#endif

        void setFpsLimit( uint32_t minInterval ) {
            m_mode->setFpsLimit(minInterval);
        }

    //

        int32_t getLatencyAverage() const
          { return m_latencyAverage.getAverage(); }

        JitterTotal getJitterStats() const
          { return m_jitterStats.getJitterTotal(); }

        const LatencyStats* getGpuBufferStats() const
          { return m_gpuBufferStats.load(); }

        const LatencyStats* getPresentStats() const
          { return m_presentationStats.load(); }

        std::atomic< bool > m_enabledGpuBufferTracking = { false };
        std::atomic< bool > m_enabledVSyncBufferTracking = { false };
        std::atomic< bool > m_enabledJitterTracking = { false };


    private:


        void trackStats( uint64_t frameId ) {
            using std::chrono::duration_cast;
            uint64_t generation = getAccountingState();
            LatencyMarkers markersPrev2 = m_latencyMarkers.getMarkers(generation, frameId-2);
            LatencyMarkers markersPrev = m_latencyMarkers.getMarkers(generation, frameId-1);
            LatencyMarkers markers = m_latencyMarkers.getMarkers(generation, frameId);
            const LatencyMarkers* m_prev2 = &markersPrev2;
            const LatencyMarkers* m_prev = &markersPrev;
            const LatencyMarkers* m = &markers;

            if (m_enabledJitterTracking && frameId > m_mode->getFirstFrameId()+2) {
                JitterEntry e;
                e.t = m->start;
                e.frametime  = std::abs( duration_cast<microseconds>( m->start - m_prev->start ).count()
                    - duration_cast<microseconds>( m_prev->start - m_prev2->start ).count() );
                e.frametime += std::abs( duration_cast<microseconds>( m->end - m_prev->end ).count()
                    - duration_cast<microseconds>( m_prev->end - m_prev2->end ).count() );
                e.frametime >>= 1;

//                e.latency = std::abs( m->gpuFinished - m_prev->gpuFinished );

                m_jitterStats.push( std::move(e) );
            }

            // will be re-enabled for VK_EXT_present_timing, but for now this isn't accurate enough
//            if (false && m_enabledVSyncBufferTracking) {
//                if (!m_presentationStats)
//                    m_presentationStats.store( new LatencyStats(3000) );
//                m_presentationStats.load()->push( m->end, m->presentFinished - m->gpuFinished );
//            }

            //    todo: adapt for vkd3d pacing code base
            //
            //      if (m_enabledGpuBufferTracking) {
            //        if (!m_gpuBufferStats)
            //          m_gpuBufferStats.store( new LatencyStats(3000) );
            //
            //        int64_t minDiff = std::numeric_limits<int64_t>::max();
            //        size_t i = 0;
            //        while (m->numAppSubmits > i && m->numGpuReadySubmits > i) {
            //          int64_t diff = std::chrono::duration_cast<microseconds>(
            //            m->gpuReady[i] - m->gpuAppSubmit[i]).count();
            //          diff = std::max( (int64_t) 0, diff );
            //          minDiff = std::min( minDiff, diff );
            //          ++i;
            //        }
            //
            //        if (minDiff != std::numeric_limits<int64_t>::max())
            //          m_gpuBufferStats.load()->push( m->end, minDiff );
            //      }
        }

        Device* m_device;
        std::unique_ptr<FramePacerMode> m_mode;

    public:

        LatencyMarkersStorage m_latencyMarkers;
        FrameMapping m_frameMapping { 2 };
        FrameSync m_frameSync;
        SimulationLedger m_simulationLedger;
        WaitableDXGISwapchain m_waitableDxgiSwapchain;

#ifdef VKD3D_ENABLE_TEST_HOOKS
        bool m_testSleepBypass = false;
        vkd3d_test_latency_sleep_decision m_testSleepDecision = {};
        std::atomic<uint64_t> m_testSleepEntryCount = { 0 };
#endif

    private:

        dxvk::mutex m_finishMutex;
        dxvk::mutex m_progressMutex;
        std::atomic<uint64_t> m_accountingState = { 2 };
        std::atomic<uint64_t> m_nextLegacyPresentAttempt = { 1 };
        std::unordered_map<uint32_t, PresentAttemptToken> m_uncapturedPresentAttempts;
        const uint64_t m_firstFrameId;

        std::atomic<LatencyStats*> m_gpuBufferStats = { nullptr };
        std::atomic<LatencyStats*> m_presentationStats = { nullptr };
        LatencyAverage m_latencyAverage;
        JitterStats m_jitterStats;

    };

}
