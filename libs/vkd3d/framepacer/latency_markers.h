#pragma once

#include <atomic>
#include <vector>
#include <array>
#include <functional>

#include "util/util_time.h"
#include "util/util_likely.h"
#include "util/thread.h"


namespace pacer {

    class FramePacer;
    class NvApi_PacingAdapter;


    struct LatencyMarkers {

        using time_point = dxvk::high_resolution_clock::time_point;

        uint64_t simulationId       = { 0 };
        time_point start             = { };
        time_point end               = { };

        time_point cpuFinished       = { };
        time_point gpuFinished       = { };
        int32_t presentFinished      = { 0 };

        int32_t renderStart          = { 0 };
        int32_t renderEnd            = { 0 };

    };


    class LatencyMarkersStorage {

        friend class FramePacer;
        friend class WaitableDXGISwapchain;

    public:

        explicit LatencyMarkersStorage(uint64_t generation = 0)
        : m_generation(generation) { }
        ~LatencyMarkersStorage() { }

        void setGeneration(uint64_t generation) {
            std::lock_guard<dxvk::mutex> lock(m_mutex);
            m_generation.store(generation, std::memory_order_release);
        }

        LatencyMarkers getMarkers(uint64_t generation, uint64_t frameId) const {
            std::lock_guard<dxvk::mutex> lock(m_mutex);
            const Slot& slot = m_markers[frameId % m_numMarkers];
            if (m_generation.load(std::memory_order_acquire) != generation ||
                    slot.generation != generation ||
                    slot.markers.simulationId != frameId)
                return {};
            return slot.markers;
        }

        bool updateMarkers(uint64_t generation, uint64_t frameId,
                const std::function<void(LatencyMarkers&)>& update) {
            std::lock_guard<dxvk::mutex> lock(m_mutex);
            if (m_generation.load(std::memory_order_acquire) != generation)
                return false;
            Slot& slot = m_markers[frameId % m_numMarkers];
            if (slot.generation != generation ||
                    slot.markers.simulationId != frameId) {
                slot.generation = generation;
                slot.markers = {};
                slot.markers.simulationId = frameId;
            }
            update(slot.markers);
            return true;
        }

    private:

        struct Slot {
            uint64_t generation = 0;
            LatencyMarkers markers = {};
        };

        // simple modulo hash mapping is used for frameIds. They are expected to monotonically increase by one.
        // only store a small number of past frames to keep the memory footprint low.
        static constexpr uint16_t m_numMarkers = 8;
        std::array<Slot, m_numMarkers> m_markers = { };
        std::atomic<uint64_t> m_generation = { 0 };
        mutable dxvk::mutex m_mutex;

    };

}
