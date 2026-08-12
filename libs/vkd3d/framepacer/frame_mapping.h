#pragma once

#include "util/thread.h"

namespace pacer {

    /*
     * internal-id (cpuId, gpuId) to frame-id
     */
    class FrameMapping {
    public:
        explicit FrameMapping(uint64_t generation = 0)
        : m_generation(generation) { }

        struct Mapping {
            std::atomic<uint64_t> sequence   = { 0 };
            std::atomic<uint64_t> generation = { 0 };
            std::atomic<uint64_t> internalId = { 0 };
            std::atomic<uint64_t> frameId    = { 0 };
        };

        void setGeneration(uint64_t generation) {
            std::lock_guard<dxvk::mutex> lock(m_mutex);
            m_generation.store(generation, std::memory_order_release);
        }

        uint64_t getFrameId(uint64_t generation, uint64_t internalId) const {
            if (m_generation.load(std::memory_order_acquire) != generation)
                return 0;
            uint16_t index = internalId % NUM_MAPPINGS;
            const Mapping& mapping = m_frameIds[index];
            uint64_t begin, end, result;
            do {
                begin = mapping.sequence.load(std::memory_order_acquire);
                result = mapping.generation.load(std::memory_order_relaxed) == generation &&
                        mapping.internalId.load(std::memory_order_relaxed) == internalId
                        ? mapping.frameId.load(std::memory_order_relaxed) : 0;
                end = mapping.sequence.load(std::memory_order_acquire);
            } while ((begin & 1) || begin != end);
            return m_generation.load(std::memory_order_acquire) == generation
                    ? result : 0;
        }

        void registerMapping(uint64_t generation, uint64_t internalId,
                uint64_t frameId) {
            std::lock_guard<dxvk::mutex> lock(m_mutex);
            if (m_generation.load(std::memory_order_acquire) != generation)
                return;
            _INFO( "register internal-id %" PRIu64 " to external-id %" PRIu64 " \n",
                internalId, frameId );
            uint16_t index = internalId % NUM_MAPPINGS;
            Mapping& mapping = m_frameIds[index];
            mapping.sequence.fetch_add(1, std::memory_order_acq_rel);
            mapping.generation.store(generation, std::memory_order_relaxed);
            mapping.internalId.store(internalId, std::memory_order_relaxed);
            mapping.frameId.store(frameId, std::memory_order_relaxed);
            mapping.sequence.fetch_add(1, std::memory_order_release);
        }

    private:
        static constexpr uint16_t NUM_MAPPINGS = 8;
        std::array< Mapping, NUM_MAPPINGS> m_frameIds = { };
        std::atomic<uint64_t> m_generation = { 0 };
        mutable dxvk::mutex m_mutex;
    };


}
