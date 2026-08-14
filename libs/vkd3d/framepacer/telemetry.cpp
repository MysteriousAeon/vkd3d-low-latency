#include "telemetry.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <inttypes.h>
#include <new>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#define get_process_id _getpid
#else
#include <unistd.h>
#define get_process_id getpid
#endif

namespace pacer::telemetry {

static std::atomic<Session *> g_session = {nullptr};
static std::mutex g_sessionMutex;
static std::condition_variable g_lifecycleCond;
static std::unique_ptr<Session> g_sessionOwner;
static std::atomic<bool> g_initialized = {false};
enum class Lifecycle : uint8_t { NeverInitialized, Unavailable, Active, Finalizing, Finalized };
static std::atomic<Lifecycle> g_lifecycle = {Lifecycle::NeverInitialized};
static uint32_t g_ownerCount;
static constexpr uint64_t ProducerClosed = uint64_t(1) << 63;
static constexpr uint64_t ProducerCountMask = ~ProducerClosed;
static std::atomic<uint64_t> g_producerState = {ProducerClosed};
static std::mutex g_producerMutex;
static std::condition_variable g_producerCond;
static std::atomic<uint64_t> g_nextSwapchainId = {1};
static std::atomic<uint64_t> g_nextDeviceId = {1};

#ifdef _WIN32
typedef void (WINAPI *WinePreExitCallback)(void *);
typedef BOOL (WINAPI *WineRegisterPreExitCallback)(WinePreExitCallback, void *, HMODULE);
static bool g_winePreExitRegistered;

static void WINAPI winePreExitFinalizer(void *) {
    shutdown();
}

static void registerWinePreExitFinalizer() noexcept {
    if (g_winePreExitRegistered)
        return;
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(winePreExitFinalizer), &module))
        return;
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    FARPROC proc = ntdll ? GetProcAddress(ntdll,
            "__wine_register_process_pre_exit_callback") : nullptr;
    WineRegisterPreExitCallback registerCallback = nullptr;
    static_assert(sizeof(registerCallback) == sizeof(proc));
    std::memcpy(&registerCallback, &proc, sizeof(registerCallback));
    if (registerCallback && registerCallback(winePreExitFinalizer, nullptr, module))
        g_winePreExitRegistered = true;
}
#endif

#ifdef VKD3D_ENABLE_TEST_HOOKS
static std::mutex g_testPauseMutex;
static std::condition_variable g_testPauseCond;
static ProducerPausePoint g_testPausePoint = ProducerPausePoint::None;
static bool g_testProducerPaused = false;
static bool g_testProducerResume = false;
static std::mutex g_testInitializationMutex;
static std::condition_variable g_testInitializationCond;
static bool g_testInitializationPause = false;
static bool g_testInitializationPaused = false;
static bool g_testInitializationResume = false;
static std::atomic<uint32_t> g_testInitializerEntries = {0};
static std::mutex g_testFinalizationMutex;
static std::condition_variable g_testFinalizationCond;
static bool g_testFinalizationPause;
static bool g_testFinalizationPaused;
static bool g_testFinalizationResume;
static uint32_t g_testFinalizationWaiters;

static void testPauseProducer(ProducerPausePoint point) {
    std::unique_lock<std::mutex> lock(g_testPauseMutex);
    if (g_testPausePoint != point)
        return;
    g_testProducerPaused = true;
    g_testPauseCond.notify_all();
    g_testPauseCond.wait(lock, [] { return g_testProducerResume; });
    g_testPausePoint = ProducerPausePoint::None;
    g_testProducerPaused = false;
    g_testProducerResume = false;
}

static void testPauseInitialization() {
    std::unique_lock<std::mutex> lock(g_testInitializationMutex);
    if (!g_testInitializationPause)
        return;
    g_testInitializationPaused = true;
    g_testInitializationCond.notify_all();
    g_testInitializationCond.wait(lock, [] { return g_testInitializationResume; });
    g_testInitializationPause = false;
    g_testInitializationPaused = false;
    g_testInitializationResume = false;
}

static void testPauseFinalization() {
    std::unique_lock<std::mutex> lock(g_testFinalizationMutex);
    if (!g_testFinalizationPause)
        return;
    g_testFinalizationPaused = true;
    g_testFinalizationCond.notify_all();
    g_testFinalizationCond.wait(lock, [] { return g_testFinalizationResume; });
    g_testFinalizationPause = false;
    g_testFinalizationPaused = false;
    g_testFinalizationResume = false;
}
#endif

struct ProcessShutdown {
    ~ProcessShutdown() {
        /* PE C++ destructors run from DLL process detach under loader lock. The
         * Wine pre-exit hook is the completeness authority; never join here. */
#ifndef _WIN32
        shutdown();
#endif
    }
};
static ProcessShutdown g_processShutdown;

uint64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            dxvk::high_resolution_clock::now().time_since_epoch()).count();
}

static uint32_t normalizeCapacity(uint32_t capacity) {
    uint32_t result = 2;
    while (result < capacity && result < (1u << 30))
        result <<= 1;
    return result;
}

static std::string outputPathForIdentity(const std::string& path, uint64_t processId,
        uint64_t runId) {
    const size_t directoryEnd = path.find_last_of("/\\\\");
    const size_t filenameStart = directoryEnd == std::string::npos ? 0 : directoryEnd + 1;
    const size_t extensionStart = path.find_last_of('.');
    const bool hasExtension = extensionStart != std::string::npos &&
            extensionStart > filenameStart;
    std::ostringstream output;
    output << path.substr(0, hasExtension ? extensionStart : path.size())
           << ".pid-" << processId << ".run-" << std::hex << std::setw(16)
           << std::setfill('0') << runId;
    if (hasExtension)
        output << path.substr(extensionStart);
    return output.str();
}

static FILE *openExclusiveOutput(const std::string& path) {
#ifdef _WIN32
    int fd = _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
            _S_IREAD | _S_IWRITE);
    if (fd < 0)
        return nullptr;
    FILE *file = _fdopen(fd, "wb");
    if (!file)
        _close(fd);
#else
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0)
        return nullptr;
    FILE *file = fdopen(fd, "wb");
    if (!file)
        close(fd);
#endif
    return file;
}

Session::Session(const std::string& path, uint32_t waitLatency, uint32_t capacity
#ifdef VKD3D_ENABLE_TEST_HOOKS
        , bool deferWriter, InitializationFailure failure
#endif
        ) noexcept
        : m_capacity(normalizeCapacity(capacity)), m_mask(m_capacity - 1),
          m_waitLatency(waitLatency), m_startNs(nowNs()),
          m_runId(m_startNs ^ (uint64_t(get_process_id()) << 32)) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
    m_testFailure = failure;
#endif
    try {
        if (path.empty() || (path[0] != '/' && !(path.size() > 2 && path[1] == ':')))
            return;
#ifdef VKD3D_ENABLE_TEST_HOOKS
        if (failure == InitializationFailure::FileOpen)
            return;
#endif
        m_outputPath = outputPathForIdentity(path, uint64_t(get_process_id()), m_runId);
        m_file.reset(openExclusiveOutput(m_outputPath));
        if (!m_file)
        {
            m_outputPath.clear();
            return;
        }
#ifdef VKD3D_ENABLE_TEST_HOOKS
        if (failure == InitializationFailure::Allocation)
            throw std::bad_alloc();
#endif
        m_slots = std::make_unique<Slot[]>(m_capacity);
        for (uint64_t i = 0; i < m_capacity; i++)
            m_slots[i].turn.store(i, std::memory_order_relaxed);
#ifdef VKD3D_ENABLE_TEST_HOOKS
        if (deferWriter) {
            m_producerState.store(0, std::memory_order_release);
            m_enabled.store(true, std::memory_order_release);
            return;
        }
        if (failure == InitializationFailure::WriterThread)
            throw std::runtime_error("injected telemetry writer failure");
#endif
        m_writerState.store(WriterState::Starting, std::memory_order_release);
        m_writer = std::thread([this] { writerEntry(); });
        m_writerStarted = true;
        {
            std::unique_lock<std::mutex> lock(m_waitMutex);
            m_waitCond.wait(lock, [this] {
                WriterState state = m_writerState.load(std::memory_order_acquire);
                return state == WriterState::Running || state == WriterState::Failed;
            });
        }
        if (m_writerState.load(std::memory_order_acquire) != WriterState::Running) {
            if (m_writer.joinable())
                m_writer.join();
            m_writerStarted = false;
            m_slots.reset();
            m_file.reset();
            return;
        }
        m_producerState.store(0, std::memory_order_release);
        m_enabled.store(true, std::memory_order_release);
    } catch (...) {
        m_producerState.fetch_or(ProducerClosed, std::memory_order_acq_rel);
        m_stop.store(true, std::memory_order_release);
        m_waitCond.notify_all();
        if (m_writer.joinable())
            m_writer.join();
        m_writerStarted = false;
        m_slots.reset();
        m_file.reset();
        m_enabled.store(false, std::memory_order_release);
    }
}

Session::~Session() {
    shutdown();
}

bool Session::acquireProducer() {
    uint64_t state = m_producerState.load(std::memory_order_acquire);
    for (;;) {
        if (state & ProducerClosed)
            return false;
        if (m_producerState.compare_exchange_weak(state, state + 1,
                std::memory_order_acquire, std::memory_order_relaxed))
            return true;
    }
}

void Session::releaseProducer() {
    uint64_t previous = m_producerState.fetch_sub(1, std::memory_order_release);
    if ((previous & ProducerClosed) && (previous & ProducerCountMask) == 1) {
        std::lock_guard<std::mutex> lock(m_producerMutex);
        m_producerCond.notify_all();
    }
}

void Session::waitForProducers() {
    std::unique_lock<std::mutex> lock(m_producerMutex);
    m_producerCond.wait(lock, [&] {
        return !(m_producerState.load(std::memory_order_acquire) & ProducerCountMask);
    });
}

bool Session::publish(Event event) {
    if (!acquireProducer())
        return false;
#ifdef VKD3D_ENABLE_TEST_HOOKS
    testPauseProducer(ProducerPausePoint::SessionAcquired);
#endif
    struct ProducerRelease {
        Session *session;
        ~ProducerRelease() { session->releaseProducer(); }
    } release = {this};

    uint64_t position = m_enqueue.load(std::memory_order_relaxed);
    for (;;) {
        Slot& slot = m_slots[position & m_mask];
        uint64_t turn = slot.turn.load(std::memory_order_acquire);
        intptr_t difference = intptr_t(turn - position);
        if (difference == 0) {
            if (m_enqueue.compare_exchange_weak(position, position + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {
                event.sequence = position + 1;
                if (!event.cpuTimestampNs)
                    event.cpuTimestampNs = nowNs();
                slot.event = event;
                slot.turn.store(position + 1, std::memory_order_release);
                m_published.fetch_add(1, std::memory_order_relaxed);
                m_waitCond.notify_one();
                return true;
            }
        } else if (difference < 0) {
            m_dropped.fetch_add(1, std::memory_order_relaxed);
            return false;
        } else {
            position = m_enqueue.load(std::memory_order_relaxed);
        }
    }
}

bool Session::pop(Event *event) {
    Slot& slot = m_slots[m_dequeue & m_mask];
    if (slot.turn.load(std::memory_order_acquire) != m_dequeue + 1)
        return false;
    *event = slot.event;
    slot.turn.store(m_dequeue + m_capacity, std::memory_order_release);
    ++m_dequeue;
    return true;
}

static const char *typeName(Type type) {
    switch (type) {
        case Type::Pacing: return "PACING";
        case Type::Submit: return "SUBMIT";
        case Type::Present: return "PRESENT";
        case Type::GpuFrontier: return "GPU_FRONTIER";
        case Type::Prediction: return "PREDICTION";
    }
    return "UNKNOWN";
}

static const char *phaseName(Phase phase) {
    switch (phase) {
        case Phase::Complete: return "COMPLETE";
        case Phase::Wait: return "WAIT";
        case Phase::Decision: return "DECISION";
        case Phase::Sleep: return "SLEEP";
        case Phase::DxgiEntry: return "DXGI_ENTRY";
        case Phase::VkPresent: return "VK_PRESENT";
    }
    return "UNKNOWN";
}

static const char *queueRoleName(QueueRole role) {
    switch (role) {
        case QueueRole::Normal: return "NORMAL";
        case QueueRole::OobRender: return "OOB_RENDER";
        case QueueRole::OobPresent: return "OOB_PRESENT";
        case QueueRole::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

static const char *captureClassName(CaptureClass value) {
    switch (value) {
        case CaptureClass::RenderCaptured: return "RENDER_CAPTURED";
        case CaptureClass::Uncaptured: return "UNCAPTURED";
        case CaptureClass::Ambiguous: return "AMBIGUOUS";
    }
    return "AMBIGUOUS";
}

void Session::writeRunHeader(std::string& out) const {
    char line[768];
    std::snprintf(line, sizeof(line),
            "{\"record_type\":\"RUN\",\"schema_version\":%u,\"run_id\":\"%016" PRIx64
            "\",\"build_id\":\"vkd3d-proton-%s+913337d603c9\",\"process_id\":%u,"
            "\"start_cpu_timestamp_ns\":%" PRIu64 ",\"cpu_clock\":\"dxvk_high_resolution_clock_ns\","
            "\"wait_latency\":%u,\"ring_capacity\":%u}\n",
            SchemaVersion, m_runId, "3.1.0", unsigned(get_process_id()),
            m_startNs, m_waitLatency, m_capacity);
    out += line;
}

void Session::writeSummary(std::string& out) const {
    char line[384];
    std::snprintf(line, sizeof(line),
            "{\"record_type\":\"SUMMARY\",\"schema_version\":%u,\"run_id\":\"%016" PRIx64
            "\",\"published_records\":%" PRIu64 ",\"dropped_records\":%" PRIu64
            ",\"end_cpu_timestamp_ns\":%" PRIu64 "}\n",
            SchemaVersion, m_runId, published(), dropped(), nowNs());
    out += line;
}

void Session::writeEvent(const Event& e, std::string& out) {
    char line[2048];
    int n = std::snprintf(line, sizeof(line),
            "{\"record_type\":\"%s\",\"schema_version\":%u,\"event_sequence\":%" PRIu64
            ",\"phase\":\"%s\",\"cpu_timestamp_ns\":%" PRIu64
            ",\"device_id\":%" PRIu64
            ",\"epoch_id\":%" PRIu64 ",\"simulation_id\":%" PRIu64
            ",\"external_reflex_id\":%" PRIu64 ",\"capture_generation\":%" PRIu64
            ",\"id0\":%" PRIu64 ",\"id1\":%" PRIu64 ",\"id2\":%" PRIu64
            ",\"timestamp0_ns\":%" PRIu64 ",\"timestamp1_ns\":%" PRIu64
            ",\"timestamp2_ns\":%" PRIu64 ",\"timestamp3_ns\":%" PRIu64
            ",\"value0\":%" PRId64 ",\"value1\":%" PRId64
            ",\"value2\":%" PRId64 ",\"value3\":%" PRId64
            ",\"count0\":%u,\"count1\":%u,\"flags\":%u,"
            "\"queue_role\":\"%s\",\"capture_class\":\"%s\"",
            typeName(e.type), SchemaVersion, e.sequence, phaseName(e.phase), e.cpuTimestampNs,
            e.deviceId,
            e.epochId, e.simulationId, e.externalReflexId, e.captureGeneration,
            e.id0, e.id1, e.id2, e.timestamp0, e.timestamp1, e.timestamp2, e.timestamp3,
            e.value0, e.value1, e.value2, e.value3, e.count0, e.count1, e.flags,
            queueRoleName(e.queueRole), captureClassName(e.captureClass));
    out.append(line, size_t(n));

    if (e.type == Type::Pacing && e.phase == Phase::Wait) {
        std::snprintf(line, sizeof(line),
                ",\"latency_sleep_entry_ns\":%" PRIu64 ",\"wait_id\":%" PRIu64
                ",\"wait_latency\":%u,\"gpu_finished_at_entry\":%" PRIu64
                ",\"cpu_finished_at_entry\":%" PRIu64 ",\"gpu_wait_begin_ns\":%" PRIu64
                ",\"gpu_wait_end_ns\":%" PRIu64 ",\"logical_depth_at_entry\":%" PRId64
                ",\"pacing_bypass\":%s,\"timeout\":%s",
                e.timestamp0, e.id0, e.count0, e.id1, e.id2, e.timestamp1, e.timestamp2,
                e.value0, (e.flags & 1) ? "true" : "false", (e.flags & 2) ? "true" : "false");
        out += line;
    } else if (e.type == Type::Pacing && e.phase == Phase::Decision) {
        std::snprintf(line, sizeof(line),
                ",\"cpu_delay_us\":%" PRId64 ",\"gpu_delay_us\":%" PRId64
                ",\"limiter_delay_us\":%" PRId64 ",\"selected_delay_us\":%" PRId64
                ",\"prediction_available\":%s,\"optimized_gpu_time_us\":%u",
                e.value0, e.value1, e.value2, e.value3,
                (e.flags & 4) ? "true" : "false", e.count0);
        out += line;
    } else if (e.type == Type::Pacing && e.phase == Phase::Sleep) {
        std::snprintf(line, sizeof(line),
                ",\"selected_delay_us\":%" PRId64 ",\"requested_sleep_us\":%" PRId64
                ",\"sleep_begin_ns\":%" PRIu64 ",\"sleep_end_ns\":%" PRIu64,
                e.value0, e.value1, e.timestamp0, e.timestamp1);
        out += line;
    } else if (e.type == Type::Submit) {
        std::snprintf(line, sizeof(line),
                ",\"command_queue_id\":%" PRIu64 ",\"command_submit_id\":%" PRIu64
                ",\"vulkan_queue_id\":%" PRIu64 ",\"vulkan_submit_id\":%" PRIu64
                ",\"publication_cpu_timestamp_ns\":%" PRIu64
                ",\"gpu_top_raw\":%" PRIu64 ",\"gpu_bottom_raw\":%" PRIu64
                ",\"gpu_top_host_ns\":%" PRIu64 ",\"gpu_bottom_host_ns\":%" PRIu64
                ",\"completion_observation_cpu_ns\":%" PRIu64
                ",\"calibration_device_raw\":%" PRIu64
                ",\"calibration_host_ns\":%" PRIu64
                ",\"calibration_max_deviation_ns\":%" PRIu64,
                e.id0, e.id1, e.id2, e.timestamp0, e.timestamp1, e.timestamp2, e.timestamp3,
                uint64_t(e.value0), uint64_t(e.value1), e.cpuTimestampNs,
                uint64_t(e.value2), uint64_t(e.value3), e.calibrationDeviationNs);
        out += line;
    } else if (e.type == Type::GpuFrontier) {
        std::snprintf(line, sizeof(line),
                ",\"frontier_id\":%" PRIu64 ",\"gpu_completion_raw\":%" PRIu64
                ",\"gpu_completion_host_ns\":%" PRIu64 ",\"publication_cpu_ns\":%" PRIu64
                ",\"published_submit_count\":%u,\"completed_submit_count\":%u",
                e.simulationId, e.timestamp0, e.timestamp1, e.cpuTimestampNs, e.count0, e.count1);
        out += line;
    } else if (e.type == Type::Present && e.phase == Phase::DxgiEntry) {
        std::snprintf(line, sizeof(line),
                ",\"swapchain_id\":%" PRIu64 ",\"dxgi_present_sequence\":%" PRIu64
                ",\"dxgi_present_entry_ns\":%" PRIu64
                ",\"present_role\":\"UNKNOWN\",\"role_source\":\"NONE\"",
                e.id0, e.id1, e.timestamp0);
        out += line;
    } else if (e.type == Type::Present && e.phase == Phase::VkPresent) {
        std::snprintf(line, sizeof(line),
                ",\"swapchain_id\":%" PRIu64 ",\"dxgi_present_sequence\":%" PRIu64
                ",\"vk_queue_present_timing\":\"UNMEASURED_OOB_NON_INTERFERENCE\""
                ",\"present_role\":\"UNKNOWN\",\"role_source\":\"NONE\"",
                e.id0, e.id1);
        out += line;
    }
    out += "}\n";
}

void Session::writeOutput(const std::string& output) {
    if (output.empty())
        return;
    if (!m_file || std::fwrite(output.data(), 1, output.size(), m_file.get()) != output.size() ||
            std::ferror(m_file.get()))
        throw std::runtime_error("telemetry output write failed");
}

void Session::writerMain() {
    std::string output;
    output.reserve(64 * 1024);
    writeRunHeader(output);
    writeOutput(output);
    output.clear();
    m_writerState.store(WriterState::Running, std::memory_order_release);
    m_waitCond.notify_all();
    for (;;) {
        Event event;
        while (pop(&event)) {
#ifdef VKD3D_ENABLE_TEST_HOOKS
            if (m_testFailure == InitializationFailure::WriterRuntime)
                throw std::runtime_error("injected telemetry writer runtime failure");
#endif
            writeEvent(event, output);
            if (output.size() >= 48 * 1024) {
                writeOutput(output);
                output.clear();
            }
        }
        if (m_stop.load(std::memory_order_acquire) &&
                m_dequeue == m_enqueue.load(std::memory_order_acquire))
            break;
        if (!output.empty()) {
            writeOutput(output);
            output.clear();
        }
        std::unique_lock<std::mutex> lock(m_waitMutex);
        m_waitCond.wait_for(lock, std::chrono::milliseconds(10));
    }
    writeSummary(output);
    writeOutput(output);
    if (std::fflush(m_file.get()) != 0)
        throw std::runtime_error("telemetry output flush failed");
}

void Session::disableAfterWriterFailure() noexcept {
    m_producerState.fetch_or(ProducerClosed, std::memory_order_acq_rel);
    m_enabled.store(false, std::memory_order_release);
    m_writerState.store(WriterState::Failed, std::memory_order_release);
    if (m_globalActive.exchange(false, std::memory_order_acq_rel))
        g_producerState.fetch_or(ProducerClosed, std::memory_order_acq_rel);
    m_waitCond.notify_all();
}

void Session::writerEntry() noexcept {
    try {
        writerMain();
        m_writerState.store(WriterState::Finished, std::memory_order_release);
        m_waitCond.notify_all();
    } catch (...) {
        disableAfterWriterFailure();
    }
}

bool Session::activateGlobal() noexcept {
    auto ready = [this] {
        WriterState state = m_writerState.load(std::memory_order_acquire);
        return enabled() && (state == WriterState::Running || state == WriterState::NotStarted);
    };
    if (!ready())
        return false;
    m_globalActive.store(true, std::memory_order_release);
    if (!ready()) {
        m_globalActive.store(false, std::memory_order_release);
        return false;
    }
    g_producerState.store(0, std::memory_order_release);
    if (!ready()) {
        g_producerState.fetch_or(ProducerClosed, std::memory_order_acq_rel);
        m_globalActive.store(false, std::memory_order_release);
        return false;
    }
    return true;
}

void Session::deactivateGlobal() noexcept {
    m_globalActive.store(false, std::memory_order_release);
}

void Session::shutdown() {
    std::lock_guard<std::mutex> shutdownLock(m_shutdownMutex);
    if (!m_writerStarted && !m_slots)
        return;
    m_producerState.fetch_or(ProducerClosed, std::memory_order_acq_rel);
    m_enabled.store(false, std::memory_order_release);
    waitForProducers();
    m_stop.store(true, std::memory_order_release);
    if (!m_writerStarted) {
        writerEntry();
        m_file.reset();
        m_slots.reset();
        return;
    }
    m_waitCond.notify_one();
    if (m_writer.joinable())
        m_writer.join();
    m_writerStarted = false;
    m_file.reset();
    m_slots.reset();
}

#ifdef VKD3D_ENABLE_TEST_HOOKS
void Session::testStartWriter() {
    if (!enabled() || m_writerStarted)
        return;
    try {
        m_writerState.store(WriterState::Starting, std::memory_order_release);
        m_writer = std::thread([this] { writerEntry(); });
        m_writerStarted = true;
        std::unique_lock<std::mutex> lock(m_waitMutex);
        m_waitCond.wait(lock, [this] {
            WriterState state = m_writerState.load(std::memory_order_acquire);
            return state == WriterState::Running || state == WriterState::Failed;
        });
    } catch (...) {
        shutdown();
    }
}

bool Session::testClosing() const {
    return m_producerState.load(std::memory_order_acquire) & ProducerClosed;
}

void Session::testWaitWriterFailed() {
    std::unique_lock<std::mutex> lock(m_waitMutex);
    m_waitCond.wait(lock, [this] {
        return m_writerState.load(std::memory_order_acquire) == WriterState::Failed;
    });
}
#endif

static bool initializeLocked(const std::string& path, uint32_t waitLatency,
        uint32_t capacity, bool deferWriter,
        InitializationFailure failure) noexcept {
    if (path.empty())
        return false;
    try {
        std::unique_ptr<Session> session;
#ifdef VKD3D_ENABLE_TEST_HOOKS
        session = std::make_unique<Session>(path, waitLatency, capacity,
                deferWriter, failure);
#else
        (void)capacity;
        (void)deferWriter;
        (void)failure;
        session = std::make_unique<Session>(path, waitLatency);
#endif
        if (!session->enabled())
            return false;
        g_sessionOwner = std::move(session);
        g_session.store(g_sessionOwner.get(), std::memory_order_release);
#ifdef VKD3D_ENABLE_TEST_HOOKS
        testPauseInitialization();
#endif
        if (!g_sessionOwner->activateGlobal()) {
            g_session.store(nullptr, std::memory_order_release);
            g_sessionOwner->shutdown();
            g_sessionOwner.reset();
            return false;
        }
        g_lifecycle.store(Lifecycle::Active, std::memory_order_release);
#ifdef _WIN32
        /* Registration pins this PE module. A failed registration leaves the
         * clean last-owner path intact but deliberately makes no exit promise. */
        registerWinePreExitFinalizer();
#endif
        return true;
    } catch (...) {
        return false;
    }
}

void initialize(uint32_t waitLatency) noexcept {
#ifdef VKD3D_ENABLE_TEST_HOOKS
    g_testInitializerEntries.fetch_add(1, std::memory_order_release);
#endif
    if (g_initialized.load(std::memory_order_acquire))
        return;
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    if (g_initialized.load(std::memory_order_relaxed))
        return;
    try {
        const char *path = std::getenv("VKD3D_FG_LATENCY_TELEMETRY");
        InitializationFailure failure = InitializationFailure::None;
#ifdef VKD3D_ENABLE_TEST_HOOKS
        const char *injectedFailure = std::getenv("VKD3D_TEST_TELEMETRY_INIT_FAILURE");
        if (injectedFailure && !std::strcmp(injectedFailure, "allocation"))
            failure = InitializationFailure::Allocation;
        else if (injectedFailure && !std::strcmp(injectedFailure, "file"))
            failure = InitializationFailure::FileOpen;
        else if (injectedFailure && !std::strcmp(injectedFailure, "thread"))
            failure = InitializationFailure::WriterThread;
#endif
        if (!initializeLocked(path ? path : "", waitLatency, DefaultRingCapacity,
                false, failure))
            g_lifecycle.store(Lifecycle::Unavailable, std::memory_order_release);
    } catch (...) {
        g_lifecycle.store(Lifecycle::Unavailable, std::memory_order_release);
    }
    g_initialized.store(true, std::memory_order_release);
}

static void releaseGlobalProducer();

static bool acquireGlobalProducer(Session **session) {
    uint64_t state = g_producerState.load(std::memory_order_acquire);
    for (;;) {
        if (state & ProducerClosed)
            return false;
        if (g_producerState.compare_exchange_weak(state, state + 1,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            *session = g_session.load(std::memory_order_acquire);
            if (*session)
                return true;
            releaseGlobalProducer();
            return false;
        }
    }
}

static void releaseGlobalProducer() {
    uint64_t previous = g_producerState.fetch_sub(1, std::memory_order_release);
    if ((previous & ProducerClosed) && (previous & ProducerCountMask) == 1) {
        std::lock_guard<std::mutex> lock(g_producerMutex);
        g_producerCond.notify_all();
    }
}

static void waitForGlobalProducers() {
    std::unique_lock<std::mutex> lock(g_producerMutex);
    g_producerCond.wait(lock, [] {
        return !(g_producerState.load(std::memory_order_acquire) & ProducerCountMask);
    });
}

/* Must be called with g_sessionMutex held. This is shared by the process
 * finalizer and the final Owner::release(), so no new owner can enter between
 * observing the last owner and sealing producer admission. */
static bool claimFinalizationLocked() {
    if (g_lifecycle.load(std::memory_order_relaxed) != Lifecycle::Active)
        return false;

    g_lifecycle.store(Lifecycle::Finalizing, std::memory_order_release);
    g_producerState.fetch_or(ProducerClosed, std::memory_order_acq_rel);
    return true;
}

static void completeFinalization() {
    std::unique_ptr<Session> session;

#ifdef VKD3D_ENABLE_TEST_HOOKS
    testPauseFinalization();
#endif
    /* Do not hold the lifecycle lock across an admitted producer wait or any
     * writer work. Admitted producers retain the published Session until this
     * wait completes. */
    waitForGlobalProducers();
    {
        std::lock_guard<std::mutex> lock(g_sessionMutex);
        g_session.store(nullptr, std::memory_order_release);
        session = std::move(g_sessionOwner);
    }
    if (session) {
        session->deactivateGlobal();
        session->shutdown();
    }
    {
        std::lock_guard<std::mutex> lock(g_sessionMutex);
        g_lifecycle.store(Lifecycle::Finalized, std::memory_order_release);
    }
    g_lifecycleCond.notify_all();
}

void shutdown() {
    {
        std::unique_lock<std::mutex> lock(g_sessionMutex);
        if (g_lifecycle.load(std::memory_order_relaxed) == Lifecycle::Finalizing) {
            /* A pre-exit caller that lost the ownership race must not return
             * to Wine before the winning finalizer has joined the writer. */
#ifdef VKD3D_ENABLE_TEST_HOOKS
            {
                std::lock_guard<std::mutex> testLock(g_testFinalizationMutex);
                ++g_testFinalizationWaiters;
                g_testFinalizationCond.notify_all();
            }
#endif
            g_lifecycleCond.wait(lock, [] {
                return g_lifecycle.load(std::memory_order_acquire) != Lifecycle::Finalizing;
            });
            return;
        }
        if (!claimFinalizationLocked())
            return;
    }
    completeFinalization();
}

bool isEnabled() {
    return g_lifecycle.load(std::memory_order_acquire) == Lifecycle::Active &&
            !(g_producerState.load(std::memory_order_relaxed) & ProducerClosed);
}

Owner::~Owner() { release(); }

Owner::Owner(Owner&& other) noexcept : m_acquired(other.m_acquired) {
    other.m_acquired = false;
}

Owner& Owner::operator=(Owner&& other) noexcept {
    if (this != &other) {
        release();
        m_acquired = other.m_acquired;
        other.m_acquired = false;
    }
    return *this;
}

bool Owner::acquire() noexcept {
    if (m_acquired)
        return true;
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    if (g_lifecycle.load(std::memory_order_relaxed) != Lifecycle::Active)
        return false;
    ++g_ownerCount;
    m_acquired = true;
    return true;
}

void Owner::release() noexcept {
    bool finalizer = false;
    {
        std::lock_guard<std::mutex> lock(g_sessionMutex);
        if (!m_acquired)
            return;
        m_acquired = false;
        if (--g_ownerCount == 0)
            finalizer = claimFinalizationLocked();
    }
    if (finalizer)
        completeFinalization();
}

bool emit(Event event) {
    Session *session = nullptr;
    if (!acquireGlobalProducer(&session))
        return false;
#ifdef VKD3D_ENABLE_TEST_HOOKS
    testPauseProducer(ProducerPausePoint::GlobalAcquired);
#endif
    bool published = session->publish(event);
    releaseGlobalProducer();
    return published;
}

uint64_t allocateSwapchainId() {
    if (!isEnabled())
        return 0;
    return g_nextSwapchainId.fetch_add(1, std::memory_order_relaxed);
}

uint64_t allocateDeviceId() {
    if (!isEnabled())
        return 0;
    return g_nextDeviceId.fetch_add(1, std::memory_order_relaxed);
}

#ifdef VKD3D_ENABLE_TEST_HOOKS
bool testInitialize(const std::string& path, uint32_t waitLatency,
        uint32_t capacity, bool deferWriter, InitializationFailure failure) {
    testReset();
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    g_initialized.store(false, std::memory_order_release);
    bool result = initializeLocked(path, waitLatency, capacity, deferWriter, failure);
    if (!result)
        g_lifecycle.store(Lifecycle::Unavailable, std::memory_order_release);
    g_initialized.store(true, std::memory_order_release);
    return result;
}

std::string testOutputPath() {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    return g_sessionOwner ? g_sessionOwner->outputPath() : "";
}

std::string testOutputPathForIdentity(const std::string& path, uint64_t processId,
        uint64_t runId) {
    return outputPathForIdentity(path, processId, runId);
}

void testReset() {
    shutdown();
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    g_initialized.store(false, std::memory_order_release);
    g_lifecycle.store(Lifecycle::NeverInitialized, std::memory_order_release);
    g_producerState.store(ProducerClosed, std::memory_order_release);
    g_ownerCount = 0;
    g_testInitializerEntries.store(0, std::memory_order_release);
    {
        std::lock_guard<std::mutex> testLock(g_testFinalizationMutex);
        g_testFinalizationPause = false;
        g_testFinalizationPaused = false;
        g_testFinalizationResume = false;
        g_testFinalizationWaiters = 0;
    }
}

void testArmProducerPause(ProducerPausePoint point) {
    std::lock_guard<std::mutex> lock(g_testPauseMutex);
    g_testPausePoint = point;
    g_testProducerPaused = false;
    g_testProducerResume = false;
}

void testWaitProducerPaused() {
    std::unique_lock<std::mutex> lock(g_testPauseMutex);
    g_testPauseCond.wait(lock, [] { return g_testProducerPaused; });
}

void testResumeProducer() {
    std::lock_guard<std::mutex> lock(g_testPauseMutex);
    g_testProducerResume = true;
    g_testPauseCond.notify_all();
}

bool testGlobalClosing() {
    return g_producerState.load(std::memory_order_acquire) & ProducerClosed;
}

void testArmInitializationPause() {
    std::lock_guard<std::mutex> lock(g_testInitializationMutex);
    g_testInitializationPause = true;
    g_testInitializationPaused = false;
    g_testInitializationResume = false;
}

void testWaitInitializationPaused() {
    std::unique_lock<std::mutex> lock(g_testInitializationMutex);
    g_testInitializationCond.wait(lock, [] { return g_testInitializationPaused; });
}

void testResumeInitialization() {
    std::lock_guard<std::mutex> lock(g_testInitializationMutex);
    g_testInitializationResume = true;
    g_testInitializationCond.notify_all();
}

void testWaitGlobalWriterFailed() {
    Session *session = g_session.load(std::memory_order_acquire);
    if (session)
        session->testWaitWriterFailed();
}

bool testInitializationComplete() {
    return g_initialized.load(std::memory_order_acquire);
}

uint32_t testInitializerEntryCount() {
    return g_testInitializerEntries.load(std::memory_order_acquire);
}

void testProcessExitFinalizer() {
    shutdown();
}

uint32_t testOwnerCount() {
    std::lock_guard<std::mutex> lock(g_sessionMutex);
    return g_ownerCount;
}

bool testFinalized() {
    return g_lifecycle.load(std::memory_order_acquire) == Lifecycle::Finalized;
}

void testArmFinalizationPause() {
    std::lock_guard<std::mutex> lock(g_testFinalizationMutex);
    g_testFinalizationPause = true;
    g_testFinalizationPaused = false;
    g_testFinalizationResume = false;
    g_testFinalizationWaiters = 0;
}

void testWaitFinalizationPaused() {
    std::unique_lock<std::mutex> lock(g_testFinalizationMutex);
    g_testFinalizationCond.wait(lock, [] { return g_testFinalizationPaused; });
}

void testResumeFinalization() {
    std::lock_guard<std::mutex> lock(g_testFinalizationMutex);
    g_testFinalizationResume = true;
    g_testFinalizationCond.notify_all();
}

void testWaitForFinalizationWaiter() {
    std::unique_lock<std::mutex> lock(g_testFinalizationMutex);
    g_testFinalizationCond.wait(lock, [] { return g_testFinalizationWaiters; });
}
#endif

} // namespace pacer::telemetry
