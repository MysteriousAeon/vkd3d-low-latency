#pragma once

#include "util/sync/sync_atomic_signal.h"
#include <stdint.h>

namespace pacer {

    class WaitableDXGISwapchain {
    public:

        WaitableDXGISwapchain( Device* device, LatencyMarkersStorage& latencyMarkers, FrameSync& frameSync,
            std::function<uint64_t()> getAccountingState,
            std::function<bool(uint64_t, dxvk::high_resolution_clock::time_point, uint64_t)> sleep )
        : m_device(device), m_latencyMarkers(latencyMarkers), m_frameSync(frameSync),
          m_getAccountingState(std::move(getAccountingState)), m_sleep(std::move(sleep)),
          m_thread([this] { threadFunc(); }) { }

        ~WaitableDXGISwapchain() {

            stop();

        }

        void stop() {

            if (m_stopped.exchange(true, std::memory_order_acq_rel))
                return;
            m_signal.signal_one();
#ifdef VKD3D_ENABLE_TEST_HOOKS
            m_testResume.signal_one();
#endif
            if (m_thread.joinable())
                m_thread.join();

        }

        bool stopped() const {
            return m_stopped.load(std::memory_order_acquire);
        }

        void releaseSemaphore( void* vkd3d_swapchain, int count ) {

            vkd3d_native_sync_handle* latencyEvent = m_device->getLatencyEvent(vkd3d_swapchain);
            if (latencyEvent && vkd3d_native_sync_handle_is_valid(*latencyEvent))
                vkd3d_native_sync_handle_release(*latencyEvent, count);

        }

        void sleep( void* vkd3d_swapchain, uint64_t expectedAccountingState = 0 ) {

            uint64_t accountingState = m_getAccountingState();
            if (expectedAccountingState && accountingState != expectedAccountingState) {
                releaseSemaphore( vkd3d_swapchain, 1 );
                return;
            }
            if (accountingState & 1) {
                releaseSemaphore( vkd3d_swapchain, 3 );
                return;
            }

            ++m_presentCounter;

            vkd3d_native_sync_handle* latencyEvent = m_device->getLatencyEvent(vkd3d_swapchain);
            if (latencyEvent && vkd3d_native_sync_handle_is_valid(*latencyEvent)) {

                // we don't block here - in dxgi.present() - other than for some reason
                // the game isn't using its waitable swapchain handle such that it calls
                // two or more dxgi.present() in a row without attempting to decrement
                // the semaphore
                m_ready.wait();

                m_frameInfo.cpuId = m_frameSync.cpuFinished;
                m_frameInfo.latencyEvent = latencyEvent;
                m_frameInfo.accountingState = accountingState;
#ifdef VKD3D_ENABLE_TEST_HOOKS
                m_frameInfo.testOnly = false;
#endif

                m_signal.signal_one();

            }

            if ((m_presentCounter & (m_presentCounterPrint-1)) == 0) {

                m_presentCounterPrint <<= 1;
                uint64_t percentage = (m_presentCounterWaitableObject*100)/m_presentCounter;
                uint64_t percentage_decimal = ((m_presentCounterWaitableObject*1000)/m_presentCounter)%10;
                INFO( "waitable dxgi swapchain present percentage: %" PRIu64 ".%" PRIu64 "%%, total frames %" PRIu64 " \n",
                    percentage, percentage_decimal, m_presentCounter.load(std::memory_order_acquire) );

            }

        }

        bool isActive() {

            if (m_presentCounter < 32)
                return false;

            uint64_t percentage = (m_presentCounterWaitableObject*100)/m_presentCounter;
            if (percentage > 90)
                return true;

            return false;

        }

#ifdef VKD3D_ENABLE_TEST_HOOKS
        void testQueueTask(uint64_t accountingState, uint64_t cpuId) {
            m_ready.wait();
            m_testMarkerReads.store(0, std::memory_order_release);
            m_testMarkerWrites.store(0, std::memory_order_release);
            m_testRejectedTasks.store(0, std::memory_order_release);
            m_frameInfo.cpuId = cpuId;
            m_frameInfo.latencyEvent = nullptr;
            m_frameInfo.accountingState = accountingState;
            m_frameInfo.testOnly = true;
            m_signal.signal_one();
        }

        void testWaitDequeued() { m_testDequeued.wait(); }
        void testResumeTask() { m_testResume.signal_one(); }
        void testWaitIdle() {
            m_ready.wait();
            m_ready.signal_one();
        }
        uint64_t testMarkerReads() const { return m_testMarkerReads.load(std::memory_order_acquire); }
        uint64_t testMarkerWrites() const { return m_testMarkerWrites.load(std::memory_order_acquire); }
        uint64_t testRejectedTasks() const { return m_testRejectedTasks.load(std::memory_order_acquire); }
#endif


    private:

        void threadFunc() {

            while (!m_stopped.load(std::memory_order_acquire)) {

                m_signal.wait();
                if (m_stopped.load(std::memory_order_acquire))
                    return;

                FrameInfo frameInfo = m_frameInfo;
                vkd3d_native_sync_handle* latencyEvent = frameInfo.latencyEvent;
#ifdef VKD3D_ENABLE_TEST_HOOKS
                if (frameInfo.testOnly) {
                    m_testDequeued.signal_one();
                    m_testResume.wait();
                }
#endif

                /* Reject a dequeued task before touching generation-sensitive
                 * marker or prediction state. The semaphore handoff below is
                 * generation-neutral and keeps the single-task protocol live. */
                if (m_getAccountingState() != frameInfo.accountingState) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
                    m_testRejectedTasks.fetch_add(1, std::memory_order_release);
#endif
                    if (latencyEvent)
                        vkd3d_native_sync_handle_release(*latencyEvent, 1);
                    m_ready.signal_one();
                    continue;
                }

                uint64_t cpuId = frameInfo.cpuId;
                uint64_t newId = cpuId + 1;

#ifdef VKD3D_ENABLE_TEST_HOOKS
                m_testMarkerReads.fetch_add(1, std::memory_order_release);
#endif
                LatencyMarkers previous = m_latencyMarkers.getMarkers(
                        frameInfo.accountingState, cpuId - 1);
                bool generationValid = m_sleep(newId, previous.cpuFinished,
                        frameInfo.accountingState);

                bool usingWaitableSwapchain = true;
                ++m_presentCounterWaitableObject;
#ifdef VKD3D_ENABLE_TEST_HOOKS
                if (!frameInfo.testOnly) {
#endif
                if (WaitForSingleObject((*latencyEvent).handle, 0) == WAIT_OBJECT_0) {
                    usingWaitableSwapchain = false;
                    --m_presentCounterWaitableObject;
                }
                while (WaitForSingleObject((*latencyEvent).handle, 0) == WAIT_OBJECT_0)
                    { }
                vkd3d_native_sync_handle_release(*latencyEvent, 1);
#ifdef VKD3D_ENABLE_TEST_HOOKS
                }
#endif
                if (usingWaitableSwapchain && generationValid) {
                    _INFO( "setting m_new->start \n" );
                    bool updated = m_latencyMarkers.updateMarkers(frameInfo.accountingState,
                            newId, [&](LatencyMarkers& markers) {
                        markers.start = dxvk::high_resolution_clock::now();
                    });
                    (void)updated;
#ifdef VKD3D_ENABLE_TEST_HOOKS
                    if (updated)
                        m_testMarkerWrites.fetch_add(1, std::memory_order_release);
#endif
                }

                m_ready.signal_one();

            }

        }

        Device* m_device;
        LatencyMarkersStorage& m_latencyMarkers;
        FrameSync& m_frameSync;

        std::function<uint64_t()> m_getAccountingState;
        std::function<bool(uint64_t, dxvk::high_resolution_clock::time_point, uint64_t)> m_sleep;

        struct FrameInfo {
            uint64_t cpuId;
            vkd3d_native_sync_handle* latencyEvent;
            uint64_t accountingState;
#ifdef VKD3D_ENABLE_TEST_HOOKS
            bool testOnly = false;
#endif
        };

        FrameInfo m_frameInfo;
        std::atomic<bool> m_stopped = { false };
        sync::AtomicSignal m_signal = { "dxgi::m_signal", false };
        sync::AtomicSignal m_ready  = { "dxgi::m_ready", true };

        std::atomic<uint64_t> m_presentCounter = { 0 };
        std::atomic<uint64_t> m_presentCounterWaitableObject = { 0 };
        uint64_t m_presentCounterPrint = { 32 };

        dxvk::thread m_thread;

#ifdef VKD3D_ENABLE_TEST_HOOKS
        sync::AtomicSignal m_testDequeued = { "dxgi::m_testDequeued", false };
        sync::AtomicSignal m_testResume = { "dxgi::m_testResume", false };
        std::atomic<uint64_t> m_testMarkerReads = { 0 };
        std::atomic<uint64_t> m_testMarkerWrites = { 0 };
        std::atomic<uint64_t> m_testRejectedTasks = { 0 };
#endif

    };


}
