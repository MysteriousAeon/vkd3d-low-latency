#include "framepacer/telemetry.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace pacer::telemetry;

static std::atomic<int> failures;

static void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "failure: %s\n", message);
        failures.fetch_add(1, std::memory_order_relaxed);
    }
}

static std::string tempPath(const char *name) {
#ifdef _WIN32
    char directory[MAX_PATH];
    DWORD length = GetTempPathA(sizeof(directory), directory);
    return std::string(directory, length) + name;
#else
    return std::string("/tmp/") + name;
#endif
}

static std::vector<std::string> readLines(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line))
        lines.push_back(line);
    return lines;
}

static unsigned countSummary(const std::vector<std::string>& lines) {
    return std::count_if(lines.begin(), lines.end(), [](const std::string& line) {
        return line.find("\"record_type\":\"SUMMARY\"") != std::string::npos;
    });
}

static uint64_t sequenceOf(const std::string& line) {
    const std::string key = "\"event_sequence\":";
    size_t position = line.find(key);
    return position == std::string::npos ? 0
            : std::strtoull(line.c_str() + position + key.size(), nullptr, 10);
}

static void setTelemetryPath(const std::string& path) {
#ifdef _WIN32
    _putenv_s("VKD3D_FG_LATENCY_TELEMETRY", path.c_str());
#else
    if (path.empty())
        unsetenv("VKD3D_FG_LATENCY_TELEMETRY");
    else
        setenv("VKD3D_FG_LATENCY_TELEMETRY", path.c_str(), 1);
#endif
}

static void testDisabled() {
    std::string path = tempPath("vkd3d-telemetry-disabled.jsonl");
    std::remove(path.c_str());
    Session session("", 3);
    check(!session.enabled(), "empty telemetry path enabled a session");
    check(!session.publish({}), "disabled session accepted an event");
    std::ifstream file(path);
    check(!file.good(), "disabled session created an output file");
}

static void testEnabledSchema() {
    std::string path = tempPath("vkd3d-telemetry-enabled.jsonl");
    std::remove(path.c_str());
    {
        Session session(path, 3, 64);
        check(session.enabled(), "absolute telemetry path did not enable session");
        Event event;
        event.type = Type::Pacing;
        event.deviceId = 41;
        event.phase = Phase::Wait;
        event.simulationId = 7;
        session.publish(event);
        event = {};
        event.type = Type::Submit;
        event.queueRole = QueueRole::OobRender;
        event.captureClass = CaptureClass::RenderCaptured;
        event.calibrationDeviationNs = UINT64_MAX;
        session.publish(event);
        event = {};
        event.type = Type::GpuFrontier;
        event.count0 = 1;
        event.count1 = 1;
        session.publish(event);
        event = {};
        event.type = Type::Present;
        event.phase = Phase::DxgiEntry;
        session.publish(event);
        event = {};
        event.type = Type::Present;
        event.phase = Phase::VkPresent;
        session.publish(event);
    }
    auto lines = readLines(path);
    check(lines.size() == 7, "enabled output did not contain header, five events, summary");
    check(!lines.empty() && lines.front().find("\"record_type\":\"RUN\"") != std::string::npos,
            "run header missing");
    check(!lines.empty() && lines.back().find("\"record_type\":\"SUMMARY\"") != std::string::npos,
            "final summary missing");
    check(lines.size() > 1 && lines[1].find("\"device_id\":41") != std::string::npos,
            "event schema omitted the process-global device identity");
    check(lines.size() > 2 &&
            lines[2].find("\"calibration_max_deviation_ns\":18446744073709551615")
                    != std::string::npos,
            "event schema narrowed the full-width calibration deviation");
    check(lines.size() > 3 &&
            lines[3].find("\"published_submit_count\":1") != std::string::npos &&
            lines[3].find("\"completed_submit_count\":1") != std::string::npos,
            "GPU_FRONTIER schema emitted uninitialized submit counts");
    check(lines.size() > 5 &&
            lines[5].find("\"vk_queue_present_timing\":\"UNMEASURED_OOB_NON_INTERFERENCE\"")
                    != std::string::npos &&
            lines[5].find("vk_queue_present_begin_ns") == std::string::npos,
            "vkQueuePresent telemetry claimed timing inside the OOB marker interval");
    for (const auto& line : lines)
        check(!line.empty() && line.front() == '{' && line.back() == '}', "JSONL framing corrupt");
}

static void testOverflow() {
    std::string path = tempPath("vkd3d-telemetry-overflow.jsonl");
    std::remove(path.c_str());
    Session session(path, 3, 8, true);
    Event event;
    event.type = Type::Pacing;
    unsigned accepted = 0;
    for (unsigned i = 0; i < 1000; i++)
        accepted += session.publish(event);
    check(accepted == 8, "bounded ring accepted more than its capacity while writer paused");
    check(session.dropped() == 992, "overflow dropped counter is not exact");
    session.testStartWriter();
    session.shutdown();
    auto lines = readLines(path);
    check(lines.size() == 10, "overflow output lost framing records or accepted events");
    check(lines.back().find("\"dropped_records\":992") != std::string::npos,
            "overflow summary omitted dropped count");
}

static void testMultiProducer() {
    std::string path = tempPath("vkd3d-telemetry-stress.jsonl");
    std::remove(path.c_str());
    constexpr unsigned Producers = 4;
    constexpr unsigned PerProducer = 4000;
    {
        Session session(path, 3, 32768);
        std::vector<std::thread> threads;
        for (unsigned producer = 0; producer < Producers; producer++) {
            threads.emplace_back([&, producer] {
                for (unsigned i = 0; i < PerProducer; i++) {
                    Event event;
                    event.type = Type::Submit;
                    event.id0 = producer + 1;
                    event.id1 = i + 1;
                    check(session.publish(event), "stress ring unexpectedly overflowed");
                }
            });
        }
        for (auto& thread : threads)
            thread.join();
        check(session.published() == Producers * PerProducer,
                "stress publication count mismatch");
        check(session.dropped() == 0, "stress run dropped records");
    }
    auto lines = readLines(path);
    check(lines.size() == Producers * PerProducer + 2, "stress JSONL record count mismatch");
    uint64_t expected = 1;
    for (size_t i = 1; i + 1 < lines.size(); i++)
        check(sequenceOf(lines[i]) == expected++, "event sequence is not unique and monotonic");
}

static void testSmallRingManyTurns() {
    std::string path = tempPath("vkd3d-telemetry-many-turns.jsonl");
    std::remove(path.c_str());
    constexpr unsigned Producers = 4;
    constexpr unsigned PerProducer = 5000;
    {
        Session session(path, 3, 8, true);
        std::atomic<bool> start = {false};
        std::vector<std::thread> threads;
        for (unsigned producer = 0; producer < Producers; producer++) {
            threads.emplace_back([&, producer] {
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                for (unsigned i = 0; i < PerProducer; i++) {
                    Event event;
                    event.type = Type::Submit;
                    event.id0 = producer + 1;
                    event.id1 = i + 1;
                    while (!session.publish(event))
                        std::this_thread::yield();
                }
            });
        }
        start.store(true, std::memory_order_release);
        while (!session.dropped())
            std::this_thread::yield();
        session.testStartWriter();
        for (auto& thread : threads)
            thread.join();
        check(session.published() == Producers * PerProducer,
                "small ring lost an eventually accepted event across reuse turns");
        check(session.dropped() != 0,
                "small ring contention did not exercise the full/retry path");
    }
    auto lines = readLines(path);
    check(lines.size() == Producers * PerProducer + 2,
            "small ring many-turn capture record count mismatch");
    uint64_t expected = 1;
    for (size_t i = 1; i + 1 < lines.size(); i++)
        check(sequenceOf(lines[i]) == expected++,
                "small ring sequence broke across slot reuse");
}

static void testSessionShutdownQuiescence() {
    std::string path = tempPath("vkd3d-telemetry-session-quiescence.jsonl");
    std::remove(path.c_str());
    Session session(path, 3, 8, true);
    testArmProducerPause(ProducerPausePoint::SessionAcquired);
    std::atomic<bool> published = {false};
    std::atomic<bool> shutdownStarted = {false};
    std::atomic<bool> shutdownComplete = {false};
    std::thread producer([&] { published.store(session.publish({}), std::memory_order_release); });
    testWaitProducerPaused();
    std::thread stopper([&] {
        shutdownStarted.store(true, std::memory_order_release);
        session.shutdown();
        shutdownComplete.store(true, std::memory_order_release);
    });
    while (!shutdownStarted.load(std::memory_order_acquire) || !session.testClosing())
        std::this_thread::yield();
    check(!shutdownComplete.load(std::memory_order_acquire),
            "session shutdown did not wait for an acquired producer");
    testResumeProducer();
    producer.join();
    stopper.join();
    check(published.load(std::memory_order_acquire),
            "acquired session producer was not allowed to finish publication");
    auto lines = readLines(path);
    check(lines.size() == 3, "session quiescence did not drain accepted publication");
    check(lines.back().find("\"published_records\":1") != std::string::npos,
            "session quiescence SUMMARY undercounted accepted producer");
}

static void testGlobalShutdownQuiescence() {
    std::string path = tempPath("vkd3d-telemetry-global-quiescence.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 8, true), "global quiescence initialization failed");
    testArmProducerPause(ProducerPausePoint::GlobalAcquired);
    std::atomic<bool> published = {false};
    std::atomic<bool> shutdownStarted = {false};
    std::atomic<bool> shutdownComplete = {false};
    std::thread producer([&] { published.store(emit({}), std::memory_order_release); });
    testWaitProducerPaused();
    std::thread stopper([&] {
        shutdownStarted.store(true, std::memory_order_release);
        shutdown();
        shutdownComplete.store(true, std::memory_order_release);
    });
    while (!shutdownStarted.load(std::memory_order_acquire) || !testGlobalClosing())
        std::this_thread::yield();
    check(!shutdownComplete.load(std::memory_order_acquire),
            "global shutdown did not wait for an acquired producer");
    testResumeProducer();
    producer.join();
    stopper.join();
    check(published.load(std::memory_order_acquire),
            "globally acquired producer was not allowed to finish publication");
    auto lines = readLines(path);
    check(lines.size() == 3, "global quiescence did not drain accepted publication");
    check(lines.back().find("\"published_records\":1") != std::string::npos,
            "global quiescence SUMMARY undercounted accepted producer");
    testReset();
}

static void testInitializationFailures() {
    std::string path = tempPath("vkd3d-telemetry-init-failure.jsonl");
    for (InitializationFailure failure : {
            InitializationFailure::Allocation,
            InitializationFailure::FileOpen,
            InitializationFailure::WriterThread}) {
        std::remove(path.c_str());
        bool result = testInitialize(path, 3, 8, false, failure);
        check(!result && !isEnabled(), "injected initialization failure enabled telemetry");
        check(!emit({}), "injected initialization failure accepted an event");
    }
    Session badPath(tempPath("missing-vkd3d-dir/output.jsonl"), 3);
    check(!badPath.enabled(), "file-open failure enabled a direct session");
    testReset();
}

static void testPostStartWriterFailure() {
    std::string path = tempPath("vkd3d-telemetry-writer-runtime-failure.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 8, false, InitializationFailure::WriterRuntime),
            "post-start writer failure session did not reach ready state");
    Event event;
    event.deviceId = 1;
    check(emit(event), "post-start writer failure did not accept its trigger event");
    testWaitGlobalWriterFailed();
    check(!isEnabled(), "post-start writer failure left the global fast path enabled");
    check(!emit(event), "post-start writer failure accepted a later event");
    shutdown();
    auto lines = readLines(path);
    check(lines.size() == 1 &&
            lines[0].find("\"record_type\":\"RUN\"") != std::string::npos,
            "post-start writer failure did not leave a safely framed startup record");
    testReset();
}

static void testInitializationPublication() {
    std::string path = tempPath("vkd3d-telemetry-init-publication.jsonl");
    std::remove(path.c_str());
    testReset();
    setTelemetryPath(path);
    testArmInitializationPause();
    std::atomic<bool> firstReturned = {false};
    std::atomic<bool> secondStarted = {false};
    std::atomic<bool> secondReturned = {false};
    std::thread first([&] {
        initialize(3);
        firstReturned.store(true, std::memory_order_release);
    });
    testWaitInitializationPaused();
    check(!testInitializationComplete(),
            "initialization was published complete before terminal session publication");
    std::thread second([&] {
        secondStarted.store(true, std::memory_order_release);
        initialize(3);
        secondReturned.store(true, std::memory_order_release);
    });
    while (!secondStarted.load(std::memory_order_acquire) ||
            testInitializerEntryCount() < 2)
        std::this_thread::yield();
    check(!firstReturned.load(std::memory_order_acquire) &&
            !secondReturned.load(std::memory_order_acquire),
            "a concurrent initializer returned while publication was paused");
    testResumeInitialization();
    first.join();
    second.join();
    Event event;
    event.deviceId = 1;
    check(testInitializationComplete() && isEnabled() && emit(event),
            "terminal initialization publication lost the first successful-session event");
    shutdown();
    auto lines = readLines(path);
    check(lines.size() == 3 && lines[1].find("\"event_sequence\":1") != std::string::npos,
            "successful initialization did not preserve the first event");
    setTelemetryPath("");
    testReset();
}

static void testConcurrentInitializationAndIds() {
    std::string path = tempPath("vkd3d-telemetry-concurrent-init.jsonl");
    std::remove(path.c_str());
    testReset();
    setTelemetryPath(path);
    std::vector<std::thread> initializers;
    for (unsigned i = 0; i < 8; i++)
        initializers.emplace_back([] { initialize(3); });
    for (auto& thread : initializers)
        thread.join();
    check(isEnabled(), "concurrent initializers did not publish one ready session");
    uint64_t first = allocateSwapchainId();
    uint64_t secondSwapchain = allocateSwapchainId();
    uint64_t firstDevice = allocateDeviceId();
    uint64_t secondDevice = allocateDeviceId();
    check(first != 0 && secondSwapchain != 0 && first != secondSwapchain,
            "process-global swapchain identities collided across owners");
    check(firstDevice != 0 && secondDevice != 0 && firstDevice != secondDevice,
            "process-global device identities collided across D3D12 devices");
    check(emit({}), "ready session lost its first post-enable event");
    shutdown();
    check(allocateSwapchainId() == 0,
            "disabled telemetry allocated a swapchain identity");
    setTelemetryPath("");
    testReset();
}

static void testOwnerLifecycle() {
    std::string path = tempPath("vkd3d-telemetry-owner-lifecycle.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64), "owner lifecycle initialization failed");
    Owner ownerA, ownerB;
    check(ownerA.acquire() && ownerB.acquire() && testOwnerCount() == 2,
            "two live FramePacer owners were not recorded");
    check(emit({}), "first owner telemetry publication failed");
    ownerA.release();
    check(isEnabled() && testOwnerCount() == 1,
            "releasing one of two owners finalized telemetry");
    check(emit({}), "second owner could not publish after first owner release");
    ownerB.release();
    check(testFinalized() && !isEnabled(), "final owner did not permanently finalize telemetry");
    auto lines = readLines(path);
    check(countSummary(lines) == 1 && !lines.empty() &&
            lines.back().find("\"record_type\":\"SUMMARY\"") != std::string::npos,
            "final owner did not leave exactly one terminal SUMMARY");
    Owner later;
    check(!later.acquire(), "telemetry reopened after final owner shutdown");
    check(readLines(path).size() == lines.size(), "later owner reopened or truncated telemetry output");
    shutdown();
    check(countSummary(readLines(path)) == 1, "module fallback produced a second SUMMARY");
    testReset();
}

static void testProcessFinalizerWithLiveOwners() {
    std::string path = tempPath("vkd3d-telemetry-process-finalizer.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 8, true), "process-finalizer initialization failed");
    Owner ownerA, ownerB;
    check(ownerA.acquire() && ownerB.acquire(), "live owners failed to acquire telemetry");
    testArmProducerPause(ProducerPausePoint::GlobalAcquired);
    std::atomic<bool> finalized = {false};
    std::thread producer([] { check(emit({}), "admitted producer was rejected by process finalizer"); });
    testWaitProducerPaused();
    std::thread finalizer([&] {
        testProcessExitFinalizer();
        finalized.store(true, std::memory_order_release);
    });
    while (!testGlobalClosing()) std::this_thread::yield();
    check(!finalized.load(std::memory_order_acquire),
            "process finalizer bypassed an admitted producer");
    testResumeProducer();
    producer.join();
    finalizer.join();
    check(testFinalized(), "process callback did not permanently finalize telemetry");
    check(countSummary(readLines(path)) == 1, "process callback did not produce one SUMMARY");
    ownerA.release();
    ownerB.release();
    shutdown();
    check(countSummary(readLines(path)) == 1,
            "owner or module fallback repeated process callback finalization");
    testReset();
}

static void testProcessFinalizerOwnerRace() {
    std::string path = tempPath("vkd3d-telemetry-finalizer-owner-race.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64), "finalizer race initialization failed");
    Owner owner;
    check(owner.acquire(), "race owner acquisition failed");
    testArmFinalizationPause();
    std::atomic<bool> processReturned = {false};
    std::thread lastOwner([&] { owner.release(); });
    testWaitFinalizationPaused();
    std::thread processFinalizer([&] {
        testProcessExitFinalizer();
        processReturned.store(true, std::memory_order_release);
    });
    testWaitForFinalizationWaiter();
    check(!processReturned.load(std::memory_order_acquire),
            "process finalizer returned while last-owner finalization was incomplete");
    testResumeFinalization();
    lastOwner.join();
    processFinalizer.join();
    check(testFinalized() && countSummary(readLines(path)) == 1,
            "process callback versus last owner race was not exactly once");
    testReset();
}

static void testOwnerAcquireFinalOwnerRace() {
    std::string path = tempPath("vkd3d-telemetry-owner-acquire-final-owner-race.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64), "owner acquire/final-owner initialization failed");
    Owner finalOwner, contender;
    check(finalOwner.acquire(), "final owner acquisition failed");
    testArmFinalizationPause();
    std::thread release([&] { finalOwner.release(); });

    /* This pause is after the atomic last-owner/finalization claim and before
     * any blocking drain. A contender must therefore see the permanent seal,
     * rather than becoming an owner whose telemetry is finalized underneath it. */
    testWaitFinalizationPaused();
    check(!contender.acquire(),
            "new owner acquired after the final owner claimed telemetry finalization");
    check(testOwnerCount() == 0,
            "failed post-finalization owner acquisition changed the owner count");
    check(!isEnabled(), "telemetry stayed active after final-owner claim");
    testResumeFinalization();
    release.join();
    check(testFinalized() && countSummary(readLines(path)) == 1,
            "owner acquire/final-owner race did not leave one terminal SUMMARY");
    testReset();
}

static void testOwnerConstructionUnwind() {
    std::string path = tempPath("vkd3d-telemetry-owner-construction-unwind.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64), "construction-unwind initialization failed");
    try {
        Owner owner;
        check(owner.acquire(), "construction-unwind owner acquisition failed");
        throw std::runtime_error("injected construction failure");
    } catch (const std::runtime_error&) {
    }
    check(testFinalized() && countSummary(readLines(path)) == 1,
            "owner construction unwind leaked ownership or duplicated finalization");
    testReset();
}

int main() {
    testDisabled();
    testEnabledSchema();
    testOverflow();
    testMultiProducer();
    testSmallRingManyTurns();
    testSessionShutdownQuiescence();
    testGlobalShutdownQuiescence();
    testInitializationFailures();
    testPostStartWriterFailure();
    testInitializationPublication();
    testConcurrentInitializationAndIds();
    testOwnerLifecycle();
    testProcessFinalizerWithLiveOwners();
    testProcessFinalizerOwnerRace();
    testOwnerAcquireFinalOwnerRace();
    testOwnerConstructionUnwind();
    int failureCount = failures.load(std::memory_order_relaxed);
    if (failureCount)
        std::fprintf(stderr, "%d telemetry test failure(s).\n", failureCount);
    else
        std::printf("framepacer telemetry transport tests passed.\n");
    return failureCount != 0;
}
