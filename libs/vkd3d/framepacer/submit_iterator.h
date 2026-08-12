#include "command_queue.h"
#include "vulkan_queue.h"
#include "util/util_time.h"
#include "util/util_log.h"


namespace pacer {

    class SubmitIterator {

        using time_point = dxvk::high_resolution_clock::time_point;

    public:

        explicit SubmitIterator( time_point start, time_point end, CommandQueue* q )
        : m_commandQueue(q), m_vulkanQueue(q->m_vulkanQueue) {
            // we may be able to make this faster with caching, but let's first care about correctness
            // also because this is not called from a performance critical thread
            auto c_submit = [q]( uint64_t id ) {
                CommandQueue::SlotSnapshot snapshot;
                return q->snapshotSlot(id, &snapshot) ? snapshot.submit : time_point{};
            };

            uint64_t submitCounter = q->m_submitCounter.load(std::memory_order_acquire);
            m_lastIndex = submitCounter ? submitCounter - 1 : 0;

            uint64_t stopIndex = (m_lastIndex > CommandQueue::NUM_SUBMITS/2)
                ? m_lastIndex - CommandQueue::NUM_SUBMITS/2 : 0;

            while (m_lastIndex > stopIndex && c_submit(m_lastIndex) > end)
                --m_lastIndex;

            m_curIndex = m_lastIndex;
            while (m_curIndex > stopIndex && c_submit(m_curIndex) > start)
                --m_curIndex;

            m_curIndex++;
            cacheVulkanQueueId();

            _INFO( "iter spans from %" PRIu64 " to %" PRIu64 "\n", m_curIndex, m_lastIndex );
        }

#ifdef VKD3D_ENABLE_TEST_HOOKS
        explicit SubmitIterator(CommandQueue* q, uint64_t commandId)
        : m_curIndex(commandId), m_lastIndex(commandId),
          m_commandQueue(q), m_vulkanQueue(q->m_vulkanQueue) {
            cacheVulkanQueueId();
        }

        uint64_t testCachedVulkanSubmitId() const {
            return m_cachedVulkanQueueId;
        }
#endif

        bool isAtEnd() {
            return m_curIndex > m_lastIndex;
        }

        void operator++() {
            ++m_curIndex;
            cacheVulkanQueueId();
        }

        void cacheVulkanQueueId() {
            if (isAtEnd())
                return;
            CommandQueue::SlotSnapshot snapshot;
            if (!m_commandQueue->snapshotSlot(m_curIndex, &snapshot)) {
                m_cachedAppSubmit = {};
                m_cachedVulkanQueueId = 0;
                return;
            }
            m_cachedAppSubmit = snapshot.submit;
            m_cachedVulkanQueueId = snapshot.vulkanId;
            m_commandQueue->testHook(CommandQueue::TestHookPoint::IteratorAfterSnapshot);
        }

        time_point getAppSubmit() {
            return m_cachedAppSubmit;
        }

        time_point getVulkanSubmit() {
            return m_vulkanQueue->getSubmitTimestamp(m_cachedVulkanQueueId);
        }

        uint64_t getVulkanGpuExecutionStart() {
            return m_vulkanQueue->getGpuExecutionStart(m_cachedVulkanQueueId);
        }

        uint64_t getVulkanGpuExecutionEnd() {
            return m_vulkanQueue->getGpuExecutionEnd(m_cachedVulkanQueueId);
        }

        uint16_t getVulkanQueueId() {
            return m_vulkanQueue->m_properties.id;
        }

    private:

        uint64_t m_curIndex;
        uint64_t m_lastIndex;
        CommandQueue* m_commandQueue;
        VulkanQueue* m_vulkanQueue;
        uint64_t m_cachedVulkanQueueId = 0;
        time_point m_cachedAppSubmit = {};
    };

}
