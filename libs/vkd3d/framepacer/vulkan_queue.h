#pragma once

#include "device.h"
#include "framepacer.h"
#include "nvapi_pacing_adapter.h"
#include "util/sync/sync_ringbuffer_allocator.h"
#include "util/util_time.h"
#include "util/util_likely.h"
#include <queue>

namespace pacer {

    class SubmitIterator;
    class CommandQueue;

    class VulkanQueue {
        using time_point = dxvk::high_resolution_clock::time_point;
        using high_resolution_clock = dxvk::high_resolution_clock;
        friend class SubmitIterator;
        friend class CommandQueue;
    public:

        struct Properties {
            uint16_t id;
            void* vkd3d_queue;
            pacer_vulkan_queue_info queueInfo;
        };

        const Properties m_properties;
        static constexpr uint16_t NUM_SUBMITS = 2048;

        VulkanQueue( Device* device, const Properties& properties )
            : m_properties( properties ), m_device( device ),
              m_thread([this] { threadFunc(); }) {
            initVulkanObjects();
        }

        ~VulkanQueue() {
            {   std::lock_guard<dxvk::mutex> lock(m_mutex);
                m_stopped.store( true );
                m_cond.notify_one();
            }
            m_thread.join();
            destroyVulkanObjects();
        }

        // we expect to handle one thread only here
        uint64_t notifySubmit( ) {
            uint64_t id = m_submitCounter.fetch_add(1, std::memory_order_acq_rel);
            uint16_t index = id % NUM_SUBMITS;
            time_point now = high_resolution_clock::now();

            testHook(TestHookPoint::ProducerBeforeSlot);
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                uint64_t previousGeneration = m_submitGenerations[index].load(
                        std::memory_order_relaxed);
                if (previousGeneration &&
                        !m_submitAccounted[index].load(std::memory_order_relaxed))
                    return 0;
                resetSlotLocked(index, id, now);
            }
            testHook(TestHookPoint::ProducerAfterSlot);
            return id;
        }

        uint64_t notifyGpuExecutionEnd( uint64_t vulkanId, pacer_query_pool* queryPool ) {

            if (queryPool == nullptr)
                return 0;

            uint16_t index = vulkanId % NUM_SUBMITS;
            uint64_t timestamp = 0;
            if (getQueryPoolResult(queryPool, &timestamp) != VK_SUCCESS)
                timestamp = 0;
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (m_submitGenerations[index].load(std::memory_order_relaxed) != vulkanId ||
                        m_submitAccounted[index].load(std::memory_order_relaxed))
                    timestamp = 0;
                else
                    m_gpuExecutionEnd[index].store(timestamp, std::memory_order_relaxed);
            }
            freeQueryPool(queryPool);
            return timestamp;

        }

        void finishSubmit(uint64_t vulkanId) {
            uint16_t index = vulkanId % NUM_SUBMITS;
            testHook(TestHookPoint::FinishBeforeSlot);
            {
                std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
                if (m_submitGenerations[index].load(std::memory_order_relaxed) == vulkanId)
                    m_submitAccounted[index].store(true, std::memory_order_relaxed);
            }
            testHook(TestHookPoint::FinishAfterSlot);
        }

        void abandonSubmit(uint64_t vulkanId) {
            finishSubmit(vulkanId);
        }

        uint64_t getGpuExecutionEnd(uint64_t vulkanId) const {
            SlotSnapshot snapshot;
            return snapshotSlot(vulkanId, &snapshot) ? snapshot.gpuExecutionEnd : 0;
        }

        time_point getSubmitTimestamp(uint64_t vulkanId) const {
            SlotSnapshot snapshot;
            return snapshotSlot(vulkanId, &snapshot) ? snapshot.submit : time_point{};
        }

        uint64_t getGpuExecutionStart(uint64_t vulkanId) const {
            SlotSnapshot snapshot;
            return snapshotSlot(vulkanId, &snapshot) ? snapshot.gpuExecutionStart : 0;
        }

        pacer_query_pool* allocQueryPool() {

            return m_queryPoolsBottomOfPipe.alloc();

        }

        pacer_query_pool* allocQueryPoolTopOfPipe() {

            pacer_query_pool* res = m_queryPoolsTopOfPipe.alloc();
            m_device->m_vkProcs.vkResetQueryPool( m_device->m_properties.vk_device,
                res->pool, 0, 1);
            return res;

        }

        void pushQueryPoolTopOfPipe( pacer_query_pool* queryPool, uint64_t submitId, bool pushIntoQueue ) {

            if (unlikely(!pushIntoQueue)) {
                // might trigger an out-of-order return log error, should be harmless / todo
                freeQueryPoolTopOfPipe( queryPool );
                return;
            }
            std::lock_guard<dxvk::mutex> lock(m_mutex);
            m_queryQueue.push( {queryPool, submitId} );
            m_cond.notify_one();

        }

        void freeQueryPool(pacer_query_pool* queryPool) {

            assert( queryPool );
            m_queryPoolsBottomOfPipe.free( queryPool );

        }

        enum class TestHookPoint : uint32_t {
            ProducerBeforeSlot,
            ProducerAfterSlot,
            FinishBeforeSlot,
            FinishAfterSlot,
            GetterBeforeSlot,
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

        void waitTestHookDrain() const {
            std::unique_lock<dxvk::mutex> lock(m_testHookMutex);
            m_testHookCond.wait(lock, [this] { return m_testHookDrainWaiting; });
        }

        void testSetSubmitTimestamp(uint64_t vulkanId, uint64_t timestamp) {
            uint16_t index = vulkanId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) == vulkanId)
                m_submits[index].store(time_point(time_point::duration(timestamp)),
                        std::memory_order_relaxed);
        }

        uint64_t testGpuExecutionEnd( uint64_t vulkanId ) const {
            return getGpuExecutionEnd(vulkanId);
        }

        void testSetGpuExecutionStart(uint64_t vulkanId, uint64_t timestamp) {
            publishGpuExecutionStart(vulkanId, timestamp);
        }

        void testSetGpuExecutionEnd(uint64_t vulkanId, uint64_t timestamp) {
            uint16_t index = vulkanId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) == vulkanId)
                m_gpuExecutionEnd[index].store(timestamp, std::memory_order_relaxed);
        }

        bool testSlotState(uint64_t vulkanId, uint64_t* generation,
                uint64_t* submitTimestamp, bool* hasSubmit, uint64_t* gpuExecutionStart,
                uint64_t* gpuExecutionEnd, bool* accounted) const {
            SlotSnapshot snapshot;
            bool found = snapshotSlot(vulkanId, &snapshot);
            *generation = snapshot.generation;
            *submitTimestamp = snapshot.submit.time_since_epoch().count();
            *hasSubmit = snapshot.submit != time_point{};
            *gpuExecutionStart = snapshot.gpuExecutionStart;
            *gpuExecutionEnd = snapshot.gpuExecutionEnd;
            *accounted = snapshot.accounted;
            return found;
        }
#endif

    private:

        struct SlotSnapshot {
            uint64_t generation = 0;
            time_point submit = {};
            uint64_t gpuExecutionStart = 0;
            uint64_t gpuExecutionEnd = 0;
            bool accounted = false;
        };

        void resetSlotLocked(uint16_t index, uint64_t id, time_point now) {
            m_submits[index].store(now, std::memory_order_relaxed);
            m_gpuExecutionStart[index].store(0, std::memory_order_relaxed);
            m_gpuExecutionEnd[index].store(0, std::memory_order_relaxed);
            m_submitAccounted[index].store(false, std::memory_order_relaxed);
            m_submitGenerations[index].store(id, std::memory_order_relaxed);
        }

        bool snapshotSlot(uint64_t vulkanId, SlotSnapshot* snapshot) const {
            if (!vulkanId)
                return false;
            uint16_t index = vulkanId % NUM_SUBMITS;
            testHook(TestHookPoint::GetterBeforeSlot);
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            snapshot->generation = m_submitGenerations[index].load(std::memory_order_relaxed);
            if (snapshot->generation != vulkanId)
                return false;
            snapshot->submit = m_submits[index].load(std::memory_order_relaxed);
            snapshot->gpuExecutionStart = m_gpuExecutionStart[index].load(std::memory_order_relaxed);
            snapshot->gpuExecutionEnd = m_gpuExecutionEnd[index].load(std::memory_order_relaxed);
            snapshot->accounted = m_submitAccounted[index].load(std::memory_order_relaxed);
            return true;
        }

        void publishGpuExecutionStart(uint64_t vulkanId, uint64_t timestamp) {
            uint16_t index = vulkanId % NUM_SUBMITS;
            std::lock_guard<dxvk::mutex> lock(m_slotMutexes[index]);
            if (m_submitGenerations[index].load(std::memory_order_relaxed) == vulkanId &&
                    !m_submitAccounted[index].load(std::memory_order_relaxed))
                m_gpuExecutionStart[index].store(timestamp, std::memory_order_relaxed);
        }

        void testHook(TestHookPoint point) const {
#ifdef VKD3D_ENABLE_TEST_HOOKS
            TestHook hook;
            void* userdata;
            {
                std::lock_guard<dxvk::mutex> lock(m_testHookMutex);
                hook = m_testHook;
                userdata = m_testHookUserdata;
                if (hook)
                    ++m_testHookInFlight;
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

        void freeQueryPoolTopOfPipe(pacer_query_pool* queryPool) {

            assert( queryPool );
            m_queryPoolsTopOfPipe.free( queryPool );

        }

        VkResult getQueryPoolResult( pacer_query_pool* queryPool, uint64_t* timestamp ) const {

            assert( queryPool );
            VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;
            VkResult res = m_device->m_vkProcs.vkGetQueryPoolResults(
              m_device->m_properties.vk_device, queryPool->pool, 0, 1, sizeof(uint64_t),
              timestamp, sizeof(uint64_t), flags
            );

            if (unlikely(res != VK_SUCCESS))
                ERR( "FramePacer: vkGetQueryPoolResults returned %u \n", res);

            return res;

        }

        void initVulkanObjects() {

            pacer_device_vk_procs* vk_procs = &m_device->m_vkProcs;
            VkDevice& device = m_device->m_properties.vk_device;

            VkQueryPoolCreateInfo queryPoolInfo = {};
            queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
            queryPoolInfo.queryCount = 1;

            // query pools for submit-start

            pacer_query_pool* queryPools = m_queryPoolsTopOfPipe.getDataUnsafe();
            for (int i=0; i<256; ++i) {
                VkResult res = vk_procs->vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPools[i].pool);

                if (res != VK_SUCCESS) {
                    ERR("FramePacer: Failed to create submit query pool \n");
                    exit(-1);
                }
            }

            // query pools for submit-end

            queryPools = m_queryPoolsBottomOfPipe.getDataUnsafe();
            for (int i=0; i<256; ++i) {
                VkResult res = vk_procs->vkCreateQueryPool(device, &queryPoolInfo, nullptr, &queryPools[i].pool);

                if (res != VK_SUCCESS) {
                    ERR("FramePacer: Failed to create submit query pool \n");
                    exit(-1);
                }
            }

            // command pool

            VkCommandPoolCreateInfo commandPoolInfo = {};
            commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolInfo.queueFamilyIndex = m_properties.queueInfo.family_index;

            if (vk_procs->vkCreateCommandPool(device, &commandPoolInfo, nullptr, &m_commandPool) != VK_SUCCESS) {
                ERR("FramePacer: Failed to create command pool \n");
                exit(-1);
            }

            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            VkCommandBufferBeginInfo beginInfo = {};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;

            // command buffers for submit-start

            queryPools = m_queryPoolsTopOfPipe.getDataUnsafe();

            for (int i=0; i<256; ++i) {
                VkResult res = vk_procs->vkAllocateCommandBuffers(
                    device, &allocInfo, &queryPools[i].buffer);

                if (res != VK_SUCCESS) {
                    ERR("FramePacer: Failed to create submit query pool \n");
                    exit(-1);
                }

                vk_procs->vkBeginCommandBuffer(queryPools[i].buffer, &beginInfo);
                // we couldn't bake the reset in like that for the beginning timestamp
                // because we would get validation error messages that we tried to access a pool
                // that wasn't reset. wondering though why this is working for the other one
//                vk_procs->vkCmdResetQueryPool(queryPools[i].buffer, queryPools[i].pool, 0, 1);
                vk_procs->vkCmdWriteTimestamp2(queryPools[i].buffer,
                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                queryPools[i].pool, 0);
                vk_procs->vkEndCommandBuffer(queryPools[i].buffer);
            }

            // command buffers for submit-end

            queryPools = m_queryPoolsBottomOfPipe.getDataUnsafe();

            for (int i=0; i<256; ++i) {
                VkResult res = vk_procs->vkAllocateCommandBuffers(
                    device, &allocInfo, &queryPools[i].buffer);

                if (res != VK_SUCCESS) {
                    ERR("FramePacer: Failed to create submit query pool \n");
                    exit(-1);
                }

                vk_procs->vkBeginCommandBuffer(queryPools[i].buffer, &beginInfo);
                vk_procs->vkCmdResetQueryPool(queryPools[i].buffer, queryPools[i].pool, 0, 1);
                vk_procs->vkCmdWriteTimestamp2(queryPools[i].buffer,
                VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                queryPools[i].pool, 0);
                vk_procs->vkEndCommandBuffer(queryPools[i].buffer);
            }
        }

        void destroyVulkanObjects() {

            pacer_device_vk_procs* vk_procs = &m_device->m_vkProcs;
            VkDevice& device = m_device->m_properties.vk_device;

            pacer_query_pool* queryPools = m_queryPoolsTopOfPipe.getDataUnsafe();
            for (int i=0; i<256; ++i) {
                vk_procs->vkFreeCommandBuffers(device, m_commandPool, 1, &queryPools[i].buffer);
            }

            queryPools = m_queryPoolsBottomOfPipe.getDataUnsafe();
            for (int i=0; i<256; ++i) {
                vk_procs->vkFreeCommandBuffers(device, m_commandPool, 1, &queryPools[i].buffer);
            }

            vk_procs->vkDestroyCommandPool(device, m_commandPool, nullptr);

            queryPools = m_queryPoolsTopOfPipe.getDataUnsafe();
            for (int i=0; i<256; ++i) {
                vk_procs->vkDestroyQueryPool(device, queryPools[i].pool, nullptr );
            }

            queryPools = m_queryPoolsBottomOfPipe.getDataUnsafe();
            for (int i=0; i<256; ++i) {
                vk_procs->vkDestroyQueryPool(device, queryPools[i].pool, nullptr );
            }

        }

        void threadFunc() {

            while (!m_stopped.load( std::memory_order_acquire)) {

                QueueItem item;
                {   std::unique_lock<dxvk::mutex> lock(m_mutex);
                    m_cond.wait( lock, [this] {
                        return m_stopped.load() || !m_queryQueue.empty();
                    });

                    if (m_stopped.load())
                        return;

                    item = std::move(m_queryQueue.front());
                    m_queryQueue.pop();
                }

                assert( item.queryPool );
                uint64_t gpuTimestamp;
                getQueryPoolResult( item.queryPool, &gpuTimestamp );
                publishGpuExecutionStart(item.submitId, gpuTimestamp);
                freeQueryPoolTopOfPipe( item.queryPool );

            }
        }

        static_assert(std::atomic<time_point>::is_always_lock_free);

        Device* m_device;

        // is accessed from multiple threads
        std::array<std::atomic<time_point>, NUM_SUBMITS> m_submits = { };
        std::array<std::atomic<uint64_t>, NUM_SUBMITS> m_gpuExecutionStart = { };
        std::array<std::atomic<uint64_t>, NUM_SUBMITS> m_gpuExecutionEnd = { };
        std::array<std::atomic<uint64_t>, NUM_SUBMITS> m_submitGenerations = { };
        std::array<std::atomic<bool>, NUM_SUBMITS> m_submitAccounted = { };
        mutable std::array<dxvk::mutex, NUM_SUBMITS> m_slotMutexes;
        std::atomic<uint64_t> m_submitCounter = { 1 };

#ifdef VKD3D_ENABLE_TEST_HOOKS
        mutable dxvk::mutex m_testHookMutex;
        mutable dxvk::condition_variable m_testHookCond;
        mutable TestHook m_testHook = nullptr;
        mutable void* m_testHookUserdata = nullptr;
        mutable uint32_t m_testHookInFlight = 0;
        mutable bool m_testHookDrainWaiting = false;
#endif

        // holding the elements in a lockfree ringbuffer has the advantage
        // that we can check if we truely cycle through the pools in order
        // alternative would be a lockfree stack, which potentially could be
        // faster but we would miss this assertive checking
        VkCommandPool m_commandPool = { VK_NULL_HANDLE };

        sync::RingbufferAllocator<pacer_query_pool, 256> m_queryPoolsTopOfPipe;
        sync::RingbufferAllocator<pacer_query_pool, 256> m_queryPoolsBottomOfPipe;

        // threading for top of pipe timestamps

        struct QueueItem {
            pacer_query_pool* queryPool;
            uint64_t submitId;
        };

        std::atomic<bool> m_stopped = { false };
        dxvk::thread m_thread;
        dxvk::condition_variable m_cond;
        dxvk::mutex m_mutex;
        std::queue<QueueItem> m_queryQueue;

    };
}
