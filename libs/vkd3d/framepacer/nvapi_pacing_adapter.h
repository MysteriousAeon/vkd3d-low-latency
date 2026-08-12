#pragma once

#include "framepacer/framepacer.h"
#include "framepacer/device.h"
#include "util/util_log.h"

#include <algorithm>
#ifdef VKD3D_ENABLE_TEST_HOOKS
#include <stdexcept>
#endif


namespace pacer {

#ifdef VKD3D_ENABLE_TEST_HOOKS
    inline std::atomic<bool> g_testFailNvApiPacingAdapterConstruction = { false };
#endif

    class NvApi_FrameId {
        using time_point = dxvk::high_resolution_clock::time_point;
        using microseconds = std::chrono::microseconds;
    public:

        FramePacer::FrameInfo getFrameInfo(uint64_t nvId,
                uint64_t accountingEpoch) {
            uint64_t seq1, seq2;
            FramePacer::FrameInfo res;

            do {
                res = FramePacer::FrameInfo {};
                seq1 = m_seq.load(std::memory_order_acquire);
                for (Mapping& mapping : m_mapping) {
                    if (mapping.nvId.load(std::memory_order_relaxed) == nvId &&
                            mapping.accountingEpoch.load(std::memory_order_relaxed) == accountingEpoch) {
                        res.externalId = nvId;
                        res.simulationId = mapping.pacerId.load(std::memory_order_relaxed);
                        res.accountingEpoch = accountingEpoch;
                        res.start_t = mapping.start_t.load(std::memory_order_relaxed);
                        res.renderStart = mapping.renderStart.load(std::memory_order_acquire);
                        res.renderEnd =  mapping.renderEnd.load(std::memory_order_acquire);
                        break;
                    }
                }
                seq2 = m_seq.load(std::memory_order_acquire);
            } while ((seq1 & 1) == 1 || seq1 != seq2);

            return res;
        }

        void updateRenderStart(uint64_t nvId, uint64_t accountingEpoch,
                time_point t) {
            updateTimestamp<&Mapping::renderStart>(nvId, accountingEpoch, t);
        }

        void updateRenderEnd(uint64_t nvId, uint64_t accountingEpoch,
                time_point t) {
            updateTimestamp<&Mapping::renderEnd>(nvId, accountingEpoch, t);
        }

        uint64_t pushMapping(uint64_t nvId, uint64_t pacerId,
                uint64_t accountingEpoch, time_point t) {
            // making seq odd: preventing reads, assumes one simultaneous writer
            m_seq.fetch_add(1, std::memory_order_release);

            for (Mapping& mapping : m_mapping) {
                if (mapping.nvId.load(std::memory_order_relaxed) == nvId) {
                    if (mapping.pacerId.load(std::memory_order_relaxed) == pacerId &&
                            mapping.accountingEpoch.load(std::memory_order_relaxed) == accountingEpoch) {
                        m_seq.fetch_add(1, std::memory_order_release);
                        return pacerId;
                    }
                    mapping.pacerId.store( pacerId, std::memory_order_relaxed );
                    mapping.accountingEpoch.store(accountingEpoch, std::memory_order_relaxed);
                    mapping.start_t.store( t, std::memory_order_relaxed );
                    mapping.renderStart.store( 0, std::memory_order_relaxed );
                    mapping.renderEnd.store( 0, std::memory_order_relaxed );
                    m_seq.fetch_add(1, std::memory_order_release);
                    return pacerId;
                }
            }
            uint64_t index = m_curIndex % m_mapping.size();
            ++m_curIndex;
            m_mapping[index].pacerId.store( pacerId, std::memory_order_relaxed );
            m_mapping[index].accountingEpoch.store(accountingEpoch, std::memory_order_relaxed);
            m_mapping[index].start_t.store( t, std::memory_order_relaxed );
            m_mapping[index].renderStart.store( 0, std::memory_order_relaxed );
            m_mapping[index].renderEnd.store( 0, std::memory_order_relaxed );
            // release
            m_mapping[index].nvId.store( nvId, std::memory_order_release );
            m_seq.fetch_add(1, std::memory_order_release); // make seq even again

            _INFO( "registered nv-id %" PRIu64
                  " to pacer-frame-id %" PRIu64 " in nvApi mapping \n", nvId, pacerId );

            return pacerId;
        }

    private:

        template <auto Timestamp>
        void updateTimestamp(uint64_t nvId, uint64_t accountingEpoch,
                time_point t) {
            // don't check the seq lock here, trust the ringbuffer

            for (Mapping& mapping : m_mapping) {
                if (mapping.nvId.load(std::memory_order_acquire) == nvId &&
                        mapping.accountingEpoch.load(std::memory_order_relaxed) == accountingEpoch) {
                    int32_t value = std::chrono::duration_cast<microseconds>(
                        t - mapping.start_t.load(std::memory_order_relaxed)).count();
                    int32_t expected = 0;
                    (mapping.*Timestamp).compare_exchange_strong( expected, value,
                        std::memory_order_release, std::memory_order_relaxed );
                    return;
                }
            }
        }

        struct alignas(64) Mapping {
            std::atomic<uint64_t> pacerId;
            std::atomic<uint64_t> nvId;
            std::atomic<uint64_t> accountingEpoch;
            std::atomic<time_point> start_t;
            std::atomic<int32_t> renderStart;
            std::atomic<int32_t> renderEnd;
        };

        std::array< Mapping, 16 > m_mapping = { };
        int64_t m_curIndex = { 0 };
        alignas(64) std::atomic< uint64_t > m_seq = { 0 };
    };


    class NvApi_PacingAdapter {
        using time_point = dxvk::high_resolution_clock::time_point;

    public:
        NvApi_PacingAdapter( Device* device )
        : m_device(device) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
            if (g_testFailNvApiPacingAdapterConstruction.exchange(false,
                    std::memory_order_acq_rel))
                throw std::runtime_error("injected NvApi_PacingAdapter construction failure");
#endif
        }

        ~NvApi_PacingAdapter() {}

        void sleepAndBeginFrame() {
            uint64_t accountingEpoch = m_device->m_pacer->getReflexEpoch();
            if (!accountingEpoch)
                return;

            uint64_t pacerId = 0;
            int64_t drift = 0;
            time_point lastEndSleep = {};
            bool shouldSleep = false;
            if (!m_device->m_pacer->withValidAccountingState(
                    accountingEpoch, [&]() {
                uint64_t priorEpoch = m_pacerState.accountingEpoch.exchange(
                        accountingEpoch, std::memory_order_acq_rel);
                if (priorEpoch != accountingEpoch) {
                    m_pacerState.simulation.store(std::max(
                            m_device->m_pacer->m_frameSync.cpuFinished.load(),
                            m_device->m_pacer->m_frameSync.gpuFinished.load()),
                            std::memory_order_release);
                    m_pacerState.drift.store(0, std::memory_order_release);
                    m_lastEndSleep.store({}, std::memory_order_release);
                    m_pendingSleep.store(false, std::memory_order_release);
                }
                // only sleep once before seeing a simulation marker
                pacerId = m_pacerState.simulation.load(std::memory_order_acquire) + 1;
                drift = m_pacerState.drift.load(std::memory_order_acquire);
                lastEndSleep = m_lastEndSleep.load(std::memory_order_acquire);
                shouldSleep = !m_pendingSleep.load(std::memory_order_acquire);
            }))
                return;

            if (shouldSleep) {
                _INFO( "sleeping for pacerId %" PRIu64 " \n", pacerId );
                // todo: we changed this timestamp from being taken at simulation start to here
                //       to make the limiter acting correctly - need to check what this changes for other cases
                bool sleepValid = m_device->m_pacer->sleep(
                        pacerId + drift, lastEndSleep, accountingEpoch);
#ifdef VKD3D_ENABLE_TEST_HOOKS
                if (m_testPauseAfterSleep.load(std::memory_order_acquire)) {
                    m_testSleepReturned.store(true, std::memory_order_release);
                    while (m_testPauseAfterSleep.load(std::memory_order_acquire))
                        std::this_thread::yield();
                }
#endif
                if (!sleepValid)
                    return;
            }

            auto t = dxvk::high_resolution_clock::now();
            m_device->m_pacer->withValidAccountingState(accountingEpoch, [&]() {
                m_lastEndSleep.store(t, std::memory_order_release);
                /* Publish the timestamp before the release which makes the
                 * completed sleep visible to simulation-marker consumers. */
                m_pendingSleep.store(true, std::memory_order_release);
            });
        }

#ifdef VKD3D_ENABLE_TEST_HOOKS
        void testPauseAfterSleep(bool pause) {
            m_testPauseAfterSleep.store(pause, std::memory_order_release);
            if (pause)
                m_testSleepReturned.store(false, std::memory_order_release);
        }

        bool testSleepReturned() const {
            return m_testSleepReturned.load(std::memory_order_acquire);
        }

        void testGetState(vkd3d_test_nvapi_adapter_state *state) const {
            state->accounting_epoch = m_pacerState.accountingEpoch.load(
                    std::memory_order_acquire);
            state->simulation_id = m_pacerState.simulation.load(
                    std::memory_order_acquire);
            state->drift = m_pacerState.drift.load(std::memory_order_acquire);
            state->pending_sleep = m_pendingSleep.load(std::memory_order_acquire);
            state->last_end_sleep = m_lastEndSleep.load(std::memory_order_acquire)
                    .time_since_epoch().count();
        }
#endif

        void setLatencyMarker( uint64_t nvId, VkLatencyMarkerNV marker ) {
            uint64_t accountingEpoch = m_device->m_pacer->getReflexEpoch();
            if (!accountingEpoch)
                return;

            using namespace std::chrono;
            switch (marker) {

                case VK_LATENCY_MARKER_SIMULATION_START_NV: {
                    // commiting a frame start to the pacer here,
                    // the pacer might have different internal ids which we need to know
                    // when we perform the sleep, which is accounted for with the "drift" variable
                    _INFO( "VK_LATENCY_MARKER_SIMULATION_START_NV %" PRIu64 "\n", nvId );

                    if (!m_pendingSleep)
                        WARN( "Simulation marker without prior sleep. Game doesn't want to get paced? \n");

                    // we have filtered out most of concurrent access possibilities here already
                    // but there is still a tiny chance for that which we cannot let happen
                    std::lock_guard<dxvk::mutex> lockGuard(m_mappingGuard);

                    time_point t = {};
                    if (!m_device->m_pacer->withValidAccountingState(
                            accountingEpoch, [&]() {
                        uint64_t priorEpoch = m_pacerState.accountingEpoch.exchange(
                                accountingEpoch, std::memory_order_acq_rel);
                        if (priorEpoch != accountingEpoch) {
                            m_pacerState.drift.store(0, std::memory_order_release);
                            m_lastEndSleep.store({}, std::memory_order_release);
                            m_pendingSleep.store(false, std::memory_order_release);
                        }
                        t = m_lastEndSleep.load(std::memory_order_acquire);
                    }))
                        break;

                    uint32_t threadId = dxvk::this_thread::get_id();
                    uint64_t pacerId = m_device->m_pacer->beginReflexSimulation(
                            accountingEpoch, nvId, threadId, t);
                    if (pacerId)
                        pacerId = m_mapping.pushMapping(nvId, pacerId,
                                accountingEpoch, t);
                    _INFO( "timestamp stored for pacerId %" PRIu64 " \n", pacerId );
                    if (pacerId)
                        m_device->m_pacer->withValidAccountingState(
                                accountingEpoch, [&]() {
                            m_pacerState.simulation.store(pacerId,
                                    std::memory_order_release);
                            m_pendingSleep.store(false,
                                    std::memory_order_release);
                        });
                    break;
                }

                case VK_LATENCY_MARKER_SIMULATION_END_NV:
                    _INFO( "VK_LATENCY_MARKER_SIMULATION_END_NV %" PRIu64 "\n", nvId );
                    break;

                case VK_LATENCY_MARKER_RENDERSUBMIT_START_NV: {
                    auto now = dxvk::high_resolution_clock::now();
                    _INFO( "VK_LATENCY_MARKER_RENDERSUBMIT_START_NV %" PRIu64 "\n", nvId );
                    m_mapping.updateRenderStart(nvId, accountingEpoch, now);
                    FramePacer::FrameInfo frameInfo = m_mapping.getFrameInfo(
                            nvId, accountingEpoch);
                    m_device->m_pacer->beginReflexRenderSubmit(accountingEpoch,
                            nvId, dxvk::this_thread::get_id(),
                            frameInfo.renderStart);
                    break;
                }

                case VK_LATENCY_MARKER_RENDERSUBMIT_END_NV: {
                    auto now = dxvk::high_resolution_clock::now();
                    _INFO( "VK_LATENCY_MARKER_RENDERSUBMIT_END_NV %" PRIu64 "\n", nvId );
                    m_mapping.updateRenderEnd(nvId, accountingEpoch, now);
                    FramePacer::FrameInfo frameInfo = m_mapping.getFrameInfo(
                            nvId, accountingEpoch);
                    m_device->m_pacer->endReflexRenderSubmit(accountingEpoch,
                            nvId, dxvk::this_thread::get_id(),
                            frameInfo.renderEnd);
                    break;
                }

                case VK_LATENCY_MARKER_PRESENT_START_NV: {
                    _INFO( "VK_LATENCY_MARKER_PRESENT_START_NV %" PRIu64 "\n", nvId );
                    FramePacer::FrameInfo frameInfo = m_mapping.getFrameInfo(
                            nvId, accountingEpoch);
                    if (frameInfo.simulationId != INVALID_ID)
                        m_device->m_pacer->beginReflexPresent(accountingEpoch,
                                frameInfo.simulationId, dxvk::this_thread::get_id(),
                                frameInfo);
                    break;
                }

                case VK_LATENCY_MARKER_PRESENT_END_NV: {
                    _INFO( "VK_LATENCY_MARKER_PRESENT_END_NV %" PRIu64 "\n", nvId );
                    FramePacer::FrameInfo frameInfo = m_mapping.getFrameInfo(
                            nvId, accountingEpoch);
                    if (frameInfo.simulationId != INVALID_ID)
                        m_device->m_pacer->endReflexPresent(accountingEpoch,
                                frameInfo.simulationId, dxvk::this_thread::get_id());
                    break;
                }

                default:
                    _INFO( "unhandled VK_LATENCY_MARKER %" PRIu32 " for nv-id %" PRIu64 "\n", marker, nvId );
                    break;
            }
        }

    private:

        Device* m_device;

        static constexpr uint64_t INVALID_ID = { 0 };

        std::atomic<bool> m_pendingSleep = { false };
        std::atomic<time_point> m_lastEndSleep = { };

        struct PacerState {
            std::atomic<uint64_t> simulation  = { 1 };
            std::atomic<int64_t> drift        = { 0 };
            std::atomic<uint64_t> accountingEpoch = { 0 };
        };

        PacerState m_pacerState;
        NvApi_FrameId m_mapping;
        dxvk::mutex m_mappingGuard;
#ifdef VKD3D_ENABLE_TEST_HOOKS
        std::atomic<bool> m_testPauseAfterSleep = { false };
        std::atomic<bool> m_testSleepReturned = { false };
#endif

    };

}
