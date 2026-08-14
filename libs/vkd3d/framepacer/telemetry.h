#pragma once

#include "util/util_time.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <condition_variable>
#include <mutex>

namespace pacer::telemetry {

static constexpr uint32_t SchemaVersion = 2;
static constexpr uint32_t DefaultRingCapacity = 4096;

enum class InitializationFailure : uint8_t {
    None,
    Allocation,
    FileOpen,
    WriterThread,
    WriterRuntime,
};

/* This is intentionally a result of an explicit finalization request, not a
 * lifecycle state or a finalization authority. */
enum class ExplicitFinalizeResult : uint8_t {
    Success,
    AlreadyFinalizing,
    AlreadyFinalized,
    Unavailable,
    FailedIncomplete,
};

enum class Type : uint16_t { Pacing, Submit, Present, GpuFrontier, Prediction };
enum class Phase : uint16_t { Complete, Wait, Decision, Sleep, DxgiEntry, VkPresent };
enum class QueueRole : uint8_t { Normal, OobRender, OobPresent, Unknown };
enum class CaptureClass : uint8_t { RenderCaptured, Uncaptured, Ambiguous };

/* One deliberately plain, fixed-size producer record. Unused fields remain zero.
 * The writer interprets fields by type/phase; producers never format JSON. */
struct Event {
    Type type = Type::Pacing;
    Phase phase = Phase::Complete;
    uint32_t flags = 0;
    uint64_t sequence = 0;
    uint64_t cpuTimestampNs = 0;
    uint64_t deviceId = 0;
    uint64_t epochId = 0;
    uint64_t simulationId = 0;
    uint64_t externalReflexId = 0;
    uint64_t captureGeneration = 0;
    uint64_t id0 = 0;
    uint64_t id1 = 0;
    uint64_t id2 = 0;
    uint64_t timestamp0 = 0;
    uint64_t timestamp1 = 0;
    uint64_t timestamp2 = 0;
    uint64_t timestamp3 = 0;
    uint64_t calibrationDeviationNs = 0;
    int64_t value0 = 0;
    int64_t value1 = 0;
    int64_t value2 = 0;
    int64_t value3 = 0;
    uint32_t count0 = 0;
    uint32_t count1 = 0;
    QueueRole queueRole = QueueRole::Unknown;
    CaptureClass captureClass = CaptureClass::Ambiguous;
    uint8_t reserved[6] = {};
};

static_assert(std::is_trivially_copyable_v<Event>);

uint64_t nowNs();

class Session {
public:
    Session(const std::string& path, uint32_t waitLatency,
            uint32_t capacity = DefaultRingCapacity
#ifdef VKD3D_ENABLE_TEST_HOOKS
            , bool deferWriter = false,
            InitializationFailure failure = InitializationFailure::None
#endif
            ) noexcept;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    bool enabled() const { return m_enabled.load(std::memory_order_acquire); }
    /* The configured path is a base; an enabled session owns this unique output. */
    const std::string& outputPath() const { return m_outputPath; }
    bool publish(Event event);
    /* False means that the terminal writer/close contract was not met. */
    bool shutdown();
    uint64_t dropped() const { return m_dropped.load(std::memory_order_relaxed); }
    uint64_t published() const { return m_published.load(std::memory_order_relaxed); }
    uint32_t capacity() const { return m_capacity; }
    /* Process-session publication is coordinated by the global transport. */
    bool activateGlobal() noexcept;
    void deactivateGlobal() noexcept;
#ifdef VKD3D_ENABLE_TEST_HOOKS
    void testStartWriter();
    bool testClosing() const;
    void testWaitWriterFailed();
#endif

private:
    struct FileCloser {
        void operator()(FILE *file) const {
            if (file)
                std::fclose(file);
        }
    };

    struct Slot {
        std::atomic<uint64_t> turn;
        Event event;
    };

    bool pop(Event *event);
    bool acquireProducer();
    void releaseProducer();
    void waitForProducers();
    void writerEntry() noexcept;
    void writerMain();
    void disableAfterWriterFailure() noexcept;
    void writeOutput(const std::string& output);
    void writeEvent(const Event& event, std::string& output);
    void writeRunHeader(std::string& output) const;
    void writeSummary(std::string& output) const;

    std::unique_ptr<Slot[]> m_slots;
    uint32_t m_capacity = 0;
    uint32_t m_mask = 0;
    uint32_t m_waitLatency = 0;
    std::unique_ptr<FILE, FileCloser> m_file;
    std::string m_outputPath;
    std::thread m_writer;
    std::mutex m_waitMutex;
    std::condition_variable m_waitCond;
    std::mutex m_shutdownMutex;
    std::atomic<uint64_t> m_enqueue = {0};
    uint64_t m_dequeue = 0;
    std::atomic<uint64_t> m_dropped = {0};
    std::atomic<uint64_t> m_published = {0};
    std::atomic<bool> m_stop = {false};
    std::atomic<bool> m_enabled = {false};
    std::atomic<uint64_t> m_producerState = {uint64_t(1) << 63};
    std::mutex m_producerMutex;
    std::condition_variable m_producerCond;
    uint64_t m_startNs = 0;
    uint64_t m_runId = 0;
    bool m_writerStarted = false;
    bool m_shutdownFinished = false;
    bool m_terminalSucceeded = false;
    enum class WriterState : uint8_t { NotStarted, Starting, Running, Failed, Finished };
    std::atomic<WriterState> m_writerState = {WriterState::NotStarted};
    std::atomic<bool> m_globalActive = {false};
#ifdef VKD3D_ENABLE_TEST_HOOKS
    InitializationFailure m_testFailure = InitializationFailure::None;
#endif
};

void initialize(uint32_t waitLatency) noexcept;
/* A FramePacer keeps one of these only while its telemetry-bearing work can
 * still publish. Releasing the last owner is the clean-device fallback when
 * process pre-exit authority is unavailable. */
class Owner {
public:
    Owner() = default;
    ~Owner();
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    Owner(Owner&& other) noexcept;
    Owner& operator=(Owner&& other) noexcept;
    bool acquire() noexcept;
    void release() noexcept;
    bool acquired() const { return m_acquired; }
private:
    bool m_acquired = false;
};

void shutdown();
bool isEnabled();
bool emit(Event event);
uint64_t allocateSwapchainId();
uint64_t allocateDeviceId();

#ifdef VKD3D_ENABLE_TEST_HOOKS
enum class TestFinalizationAuthority : uint8_t {
    Auto,
    LastOwnerFallback,
    ProcessPreExit,
};

enum class ProducerPausePoint : uint8_t {
    None,
    GlobalAcquired,
    SessionAcquired,
};

bool testInitialize(const std::string& path, uint32_t waitLatency,
        uint32_t capacity = DefaultRingCapacity, bool deferWriter = false,
        InitializationFailure failure = InitializationFailure::None,
        TestFinalizationAuthority authority = TestFinalizationAuthority::LastOwnerFallback);
std::string testOutputPath();
std::string testOutputPathForIdentity(const std::string& path, uint64_t processId,
        uint64_t runId);
void testReset();
void testArmProducerPause(ProducerPausePoint point);
void testWaitProducerPaused();
void testResumeProducer();
bool testGlobalClosing();
void testArmInitializationPause();
void testWaitInitializationPaused();
void testResumeInitialization();
void testWaitGlobalWriterFailed();
bool testInitializationComplete();
uint32_t testInitializerEntryCount();
void testProcessExitFinalizer();
ExplicitFinalizeResult testExplicitFinalize();
uint32_t testOwnerCount();
bool testActive();
bool testFinalized();
TestFinalizationAuthority testFinalizationAuthority();
void testArmFinalizationPause();
void testWaitFinalizationPaused();
void testResumeFinalization();
void testWaitForFinalizationWaiter();
#endif

} // namespace pacer::telemetry
