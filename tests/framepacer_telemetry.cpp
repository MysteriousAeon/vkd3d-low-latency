#include "framepacer/telemetry.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <chrono>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
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

static std::string readBytes(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static unsigned countSummary(const std::vector<std::string>& lines) {
    return std::count_if(lines.begin(), lines.end(), [](const std::string& line) {
        return line.find("\"record_type\":\"SUMMARY\"") != std::string::npos;
    });
}

static unsigned countRun(const std::vector<std::string>& lines) {
    return std::count_if(lines.begin(), lines.end(), [](const std::string& line) {
        return line.find("\"record_type\":\"RUN\"") != std::string::npos;
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
    std::string output;
    {
        Session session(path, 3, 64);
        check(session.enabled(), "absolute telemetry path did not enable session");
        output = session.outputPath();
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
    auto lines = readLines(output);
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
    std::string output = session.outputPath();
    Event event;
    event.type = Type::Pacing;
    unsigned accepted = 0;
    for (unsigned i = 0; i < 1000; i++)
        accepted += session.publish(event);
    check(accepted == 8, "bounded ring accepted more than its capacity while writer paused");
    check(session.dropped() == 992, "overflow dropped counter is not exact");
    session.testStartWriter();
    session.shutdown();
    auto lines = readLines(output);
    check(lines.size() == 10, "overflow output lost framing records or accepted events");
    check(lines.back().find("\"dropped_records\":992") != std::string::npos,
            "overflow summary omitted dropped count");
}

static void testMultiProducer() {
    std::string path = tempPath("vkd3d-telemetry-stress.jsonl");
    std::remove(path.c_str());
    constexpr unsigned Producers = 4;
    constexpr unsigned PerProducer = 4000;
    std::string output;
    {
        Session session(path, 3, 32768);
        output = session.outputPath();
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
    auto lines = readLines(output);
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
    std::string output;
    {
        Session session(path, 3, 8, true);
        output = session.outputPath();
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
    auto lines = readLines(output);
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
    std::string output = session.outputPath();
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
    auto lines = readLines(output);
    check(lines.size() == 3, "session quiescence did not drain accepted publication");
    check(lines.back().find("\"published_records\":1") != std::string::npos,
            "session quiescence SUMMARY undercounted accepted producer");
}

static void testGlobalShutdownQuiescence() {
    std::string path = tempPath("vkd3d-telemetry-global-quiescence.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 8, true), "global quiescence initialization failed");
    std::string output = testOutputPath();
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
    auto lines = readLines(output);
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
    std::string output = testOutputPath();
    Event event;
    event.deviceId = 1;
    check(emit(event), "post-start writer failure did not accept its trigger event");
    testWaitGlobalWriterFailed();
    check(!isEnabled(), "post-start writer failure left the global fast path enabled");
    check(!emit(event), "post-start writer failure accepted a later event");
    shutdown();
    auto lines = readLines(output);
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
    std::string output = testOutputPath();
    shutdown();
    auto lines = readLines(output);
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
    check(testInitialize(path, 3, 64, false, InitializationFailure::None,
            TestFinalizationAuthority::LastOwnerFallback),
            "owner lifecycle initialization failed");
    std::string output = testOutputPath();
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
    auto lines = readLines(output);
    check(countSummary(lines) == 1 && !lines.empty() &&
            lines.back().find("\"record_type\":\"SUMMARY\"") != std::string::npos,
            "final owner did not leave exactly one terminal SUMMARY");
    Owner later;
    check(!later.acquire(), "telemetry reopened after final owner shutdown");
    check(readLines(output).size() == lines.size(), "later owner reopened or truncated telemetry output");
    shutdown();
    check(countSummary(readLines(output)) == 1, "module fallback produced a second SUMMARY");
    testReset();
}

static void testProcessFinalizerWithLiveOwners() {
    std::string path = tempPath("vkd3d-telemetry-process-finalizer.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 8, true, InitializationFailure::None,
            TestFinalizationAuthority::ProcessPreExit),
            "process-finalizer initialization failed");
    std::string output = testOutputPath();
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
    check(countSummary(readLines(output)) == 1, "process callback did not produce one SUMMARY");
    ownerA.release();
    ownerB.release();
    shutdown();
    check(countSummary(readLines(output)) == 1,
            "owner or module fallback repeated process callback finalization");
    testReset();
}

static void testProcessPreExitTransientZeroOwners() {
    std::string path = tempPath("vkd3d-telemetry-process-pre-exit-zero-owner-gap.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64, false, InitializationFailure::None,
            TestFinalizationAuthority::ProcessPreExit),
            "process-pre-exit zero-owner initialization failed");
    std::string output = testOutputPath();
    check(testFinalizationAuthority() == TestFinalizationAuthority::ProcessPreExit,
            "process-pre-exit authority was not elected");

    Owner ownerA, ownerB;
    check(ownerA.acquire(), "first owner failed to acquire process-pre-exit telemetry");
    Event first;
    first.id0 = 0xa11;
    check(emit(first), "first owner failed to emit telemetry");
    ownerA.release();
    check(testActive() && isEnabled() && testOwnerCount() == 0 && !testGlobalClosing(),
            "process-pre-exit authority finalized during a transient zero-owner gap");
    check(countSummary(readLines(output)) == 0,
            "transient zero-owner gap produced a SUMMARY");

    check(ownerB.acquire() && testOwnerCount() == 1,
            "second owner could not acquire the still-active telemetry session");
    Event second;
    second.id0 = 0xb22;
    check(emit(second), "second owner telemetry event was rejected after zero-owner gap");
    testProcessExitFinalizer();

    auto lines = readLines(output);
    check(countRun(lines) == 1, "transient zero-owner gap created more than one RUN");
    check(std::any_of(lines.begin(), lines.end(), [](const std::string& line) {
        return line.find("\"id0\":2850") != std::string::npos;
    }), "second owner telemetry event did not survive process finalization");
    check(countSummary(lines) == 1 && !lines.empty() &&
            lines.back().find("\"record_type\":\"SUMMARY\"") != std::string::npos,
            "process finalization did not leave one terminal SUMMARY");
    check(lines.back().find("\"dropped_records\":0") != std::string::npos,
            "deterministic transient zero-owner test dropped telemetry");
    ownerB.release();
    testReset();
}

static void testProcessPreExitAcquireRace() {
    std::string path = tempPath("vkd3d-telemetry-process-pre-exit-acquire-race.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64, false, InitializationFailure::None,
            TestFinalizationAuthority::ProcessPreExit),
            "process-pre-exit acquire-race initialization failed");
    std::string output = testOutputPath();
    check(testActive() && testOwnerCount() == 0,
            "process-pre-exit acquire-race did not start active with zero owners");
    testArmFinalizationPause();
    std::thread finalizer([] { testProcessExitFinalizer(); });
    testWaitFinalizationPaused();

    Owner contender;
    check(!contender.acquire(), "owner acquired after process pre-exit sealed telemetry");
    check(testOwnerCount() == 0,
            "failed acquisition after process pre-exit changed owner count");
    testResumeFinalization();
    finalizer.join();

    auto lines = readLines(output);
    check(testFinalized() && countSummary(lines) == 1 && !lines.empty() &&
            lines.back().find("\"record_type\":\"SUMMARY\"") != std::string::npos,
            "process pre-exit versus acquisition race did not leave one terminal SUMMARY");
    testReset();
}

static void testProcessFinalizerOwnerRace() {
    std::string path = tempPath("vkd3d-telemetry-finalizer-owner-race.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64, false, InitializationFailure::None,
            TestFinalizationAuthority::LastOwnerFallback),
            "finalizer race initialization failed");
    std::string output = testOutputPath();
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
    check(testFinalized() && countSummary(readLines(output)) == 1,
            "process callback versus last owner race was not exactly once");
    testReset();
}

static void testOwnerAcquireFinalOwnerRace() {
    std::string path = tempPath("vkd3d-telemetry-owner-acquire-final-owner-race.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64, false, InitializationFailure::None,
            TestFinalizationAuthority::LastOwnerFallback),
            "owner acquire/final-owner initialization failed");
    std::string output = testOutputPath();
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
    check(testFinalized() && countSummary(readLines(output)) == 1,
            "owner acquire/final-owner race did not leave one terminal SUMMARY");
    testReset();
}

static void testOwnerConstructionUnwind() {
    std::string path = tempPath("vkd3d-telemetry-owner-construction-unwind.jsonl");
    std::remove(path.c_str());
    check(testInitialize(path, 3, 64, false, InitializationFailure::None,
            TestFinalizationAuthority::LastOwnerFallback),
            "construction-unwind initialization failed");
    std::string output = testOutputPath();
    try {
        Owner owner;
        check(owner.acquire(), "construction-unwind owner acquisition failed");
        throw std::runtime_error("injected construction failure");
    } catch (const std::runtime_error&) {
    }
    check(testFinalized() && countSummary(readLines(output)) == 1,
            "owner construction unwind leaked ownership or duplicated finalization");
    testReset();
}

static void testOutputIdentityPathShapes() {
    check(testOutputPathForIdentity("/tmp/foo.jsonl", 42, 0xabc) ==
            "/tmp/foo.pid-42.run-0000000000000abc.jsonl",
            "extension output identity did not preserve the configured directory and extension");
    check(testOutputPathForIdentity("/tmp/foo", 42, 0xabc) ==
            "/tmp/foo.pid-42.run-0000000000000abc",
            "extensionless output identity was malformed");
    check(testOutputPathForIdentity("/tmp/telemetry directory/foo.jsonl", 42, 0xabc) ==
            "/tmp/telemetry directory/foo.pid-42.run-0000000000000abc.jsonl",
            "space-containing output identity was malformed");
    check(testOutputPathForIdentity("/tmp/foo.jsonl", 42, 0xabc) !=
            testOutputPathForIdentity("/tmp/foo.jsonl", 42, 0xabd),
            "sequential sessions with a reused PID resolved to one output identity");
}

static void testDistinctOutputsDoNotTruncate() {
    std::string path = tempPath("vkd3d-telemetry-output-identity.jsonl");
    std::remove(path.c_str());
    std::string firstOutput;
    {
        Session first(path, 3, 64);
        check(first.enabled(), "first output-identity session did not initialize");
        firstOutput = first.outputPath();
        Event event;
        event.deviceId = 1;
        check(first.publish(event), "first output-identity session did not publish");
    }
    const std::string firstBytes = readBytes(firstOutput);
    std::string secondOutput;
    {
        Session second(path, 3, 64);
        check(second.enabled(), "second output-identity session did not initialize");
        secondOutput = second.outputPath();
        Event event;
        event.deviceId = 1;
        check(second.publish(event), "second output-identity session did not publish");
    }
    check(!firstOutput.empty() && !secondOutput.empty() && firstOutput != secondOutput,
            "distinct sessions sharing one configured path resolved to one output file");
    check(readBytes(firstOutput) == firstBytes,
            "second session changed the completed first session output");
    auto firstLines = readLines(firstOutput);
    auto secondLines = readLines(secondOutput);
    check(firstLines.size() == 3 && secondLines.size() == 3 &&
            countSummary(firstLines) == 1 && countSummary(secondLines) == 1,
            "separate output-identity sessions did not preserve complete framing");
}

#ifndef _WIN32
enum class JsonType { Null, Boolean, Number, String, Array, Object };

struct JsonValue {
    JsonType type = JsonType::Null;
    std::string text;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    const JsonValue *member(const char *name) const {
        auto it = object.find(name);
        return it == object.end() ? nullptr : &it->second;
    }
};

/* Test-only JSON decoder. Parse complete JSON values, not just the fields this
 * test inspects, so malformed event records cannot hide in a valid JSONL file. */
class JsonParser {
public:
    explicit JsonParser(const std::string& input) : m_input(input) {}

    JsonValue parse() {
        JsonValue value = parseValue();
        skipSpace();
        if (m_position != m_input.size())
            fail("trailing data");
        return value;
    }

private:
    [[noreturn]] void fail(const char *reason) const {
        throw std::runtime_error(reason);
    }

    void skipSpace() {
        while (m_position < m_input.size() && std::isspace(
                static_cast<unsigned char>(m_input[m_position])))
            ++m_position;
    }

    char take() {
        if (m_position == m_input.size())
            fail("unexpected end of JSON");
        return m_input[m_position++];
    }

    void expect(char expected) {
        if (take() != expected)
            fail("unexpected JSON token");
    }

    bool consume(char expected) {
        skipSpace();
        if (m_position == m_input.size() || m_input[m_position] != expected)
            return false;
        ++m_position;
        return true;
    }

    static unsigned hex(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        throw std::runtime_error("invalid unicode escape");
    }

    unsigned unicodeEscape() {
        if (m_position + 4 > m_input.size())
            fail("truncated unicode escape");
        unsigned value = 0;
        for (unsigned i = 0; i < 4; ++i)
            value = (value << 4) | hex(m_input[m_position++]);
        return value;
    }

    static void appendUtf8(std::string& output, unsigned codepoint) {
        if (codepoint <= 0x7f)
            output += char(codepoint);
        else if (codepoint <= 0x7ff) {
            output += char(0xc0 | (codepoint >> 6));
            output += char(0x80 | (codepoint & 0x3f));
        } else if (codepoint <= 0xffff) {
            output += char(0xe0 | (codepoint >> 12));
            output += char(0x80 | ((codepoint >> 6) & 0x3f));
            output += char(0x80 | (codepoint & 0x3f));
        } else {
            output += char(0xf0 | (codepoint >> 18));
            output += char(0x80 | ((codepoint >> 12) & 0x3f));
            output += char(0x80 | ((codepoint >> 6) & 0x3f));
            output += char(0x80 | (codepoint & 0x3f));
        }
    }

    std::string parseString() {
        expect('"');
        std::string output;
        while (m_position < m_input.size()) {
            unsigned char value = static_cast<unsigned char>(take());
            if (value == '"')
                return output;
            if (value < 0x20)
                fail("control character in JSON string");
            if (value != '\\') {
                output += char(value);
                continue;
            }
            switch (take()) {
                case '"': output += '"'; break;
                case '\\': output += '\\'; break;
                case '/': output += '/'; break;
                case 'b': output += '\b'; break;
                case 'f': output += '\f'; break;
                case 'n': output += '\n'; break;
                case 'r': output += '\r'; break;
                case 't': output += '\t'; break;
                case 'u': {
                    unsigned codepoint = unicodeEscape();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (m_position + 2 > m_input.size() || m_input[m_position++] != '\\' ||
                                m_input[m_position++] != 'u')
                            fail("unpaired unicode surrogate");
                        unsigned low = unicodeEscape();
                        if (low < 0xdc00 || low > 0xdfff)
                            fail("unpaired unicode surrogate");
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail("unpaired unicode surrogate");
                    }
                    appendUtf8(output, codepoint);
                    break;
                }
                default: fail("invalid JSON string escape");
            }
        }
        fail("unterminated JSON string");
    }

    JsonValue parseNumber() {
        const size_t start = m_position;
        if (m_input[m_position] == '-')
            ++m_position;
        if (m_position == m_input.size())
            fail("truncated JSON number");
        if (m_input[m_position] == '0')
            ++m_position;
        else {
            if (!std::isdigit(static_cast<unsigned char>(m_input[m_position])))
                fail("invalid JSON number");
            while (m_position < m_input.size() && std::isdigit(
                    static_cast<unsigned char>(m_input[m_position])))
                ++m_position;
        }
        if (m_position < m_input.size() && m_input[m_position] == '.') {
            ++m_position;
            if (m_position == m_input.size() || !std::isdigit(
                    static_cast<unsigned char>(m_input[m_position])))
                fail("invalid JSON fraction");
            while (m_position < m_input.size() && std::isdigit(
                    static_cast<unsigned char>(m_input[m_position])))
                ++m_position;
        }
        if (m_position < m_input.size() && (m_input[m_position] == 'e' ||
                m_input[m_position] == 'E')) {
            ++m_position;
            if (m_position < m_input.size() && (m_input[m_position] == '+' ||
                    m_input[m_position] == '-'))
                ++m_position;
            if (m_position == m_input.size() || !std::isdigit(
                    static_cast<unsigned char>(m_input[m_position])))
                fail("invalid JSON exponent");
            while (m_position < m_input.size() && std::isdigit(
                    static_cast<unsigned char>(m_input[m_position])))
                ++m_position;
        }
        JsonValue value;
        value.type = JsonType::Number;
        value.text = m_input.substr(start, m_position - start);
        return value;
    }

    JsonValue parseArray() {
        expect('[');
        JsonValue value;
        value.type = JsonType::Array;
        if (consume(']'))
            return value;
        for (;;) {
            value.array.push_back(parseValue());
            if (consume(']'))
                return value;
            expect(',');
        }
    }

    JsonValue parseObject() {
        expect('{');
        JsonValue value;
        value.type = JsonType::Object;
        if (consume('}'))
            return value;
        for (;;) {
            skipSpace();
            if (m_position == m_input.size() || m_input[m_position] != '"')
                fail("JSON object key is not a string");
            std::string key = parseString();
            skipSpace();
            expect(':');
            auto insertion = value.object.emplace(std::move(key), parseValue());
            if (!insertion.second)
                fail("duplicate JSON object key");
            if (consume('}'))
                return value;
            expect(',');
        }
    }

    JsonValue parseValue() {
        skipSpace();
        if (m_position == m_input.size())
            fail("missing JSON value");
        char value = m_input[m_position];
        if (value == '{') return parseObject();
        if (value == '[') return parseArray();
        if (value == '"') {
            JsonValue result;
            result.type = JsonType::String;
            result.text = parseString();
            return result;
        }
        if (value == '-' || std::isdigit(static_cast<unsigned char>(value)))
            return parseNumber();
        const char *literal = value == 't' ? "true" : value == 'f' ? "false" :
                value == 'n' ? "null" : nullptr;
        if (!literal || m_input.compare(m_position, std::strlen(literal), literal))
            fail("invalid JSON value");
        m_position += std::strlen(literal);
        JsonValue result;
        result.type = value == 'n' ? JsonType::Null : JsonType::Boolean;
        result.text = literal;
        return result;
    }

    const std::string& m_input;
    size_t m_position = 0;
};

static std::vector<std::string> outputPathsForBase(const std::string& path) {
    const size_t separator = path.find_last_of('/');
    const std::string directory = separator == std::string::npos ? "." : path.substr(0, separator);
    const std::string filename = path.substr(separator == std::string::npos ? 0 : separator + 1);
    const size_t extension = filename.find_last_of('.');
    const bool hasExtension = extension != std::string::npos && extension != 0;
    const std::string prefix = filename.substr(0, hasExtension ? extension : filename.size()) + ".pid-";
    const std::string suffix = hasExtension ? filename.substr(extension) : "";
    std::vector<std::string> paths;
    DIR *stream = opendir(directory.c_str());
    if (!stream)
        return paths;
    while (dirent *entry = readdir(stream)) {
        std::string name = entry->d_name;
        if (name.rfind(prefix, 0) == 0 &&
                (suffix.empty() || (name.size() >= suffix.size() &&
                        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)))
            paths.push_back(directory + "/" + name);
    }
    closedir(stream);
    return paths;
}

static bool writeAll(int fd, const std::string& value) {
    size_t written = 0;
    while (written < value.size()) {
        ssize_t result = write(fd, value.data() + written, value.size() - written);
        if (result > 0) {
            written += size_t(result);
            continue;
        }
        if (result < 0 && errno == EINTR)
            continue;
        return false;
    }
    return true;
}

static bool readLineWithTimeout(int fd, std::string *line) {
    line->clear();
    constexpr int TimeoutMs = 5000;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(TimeoutMs);
    while (line->size() < 4096) {
        auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
            return false;
        int remaining = int(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
                .count()) + 1;
        pollfd descriptor = {fd, POLLIN, 0};
        int ready = poll(&descriptor, 1, remaining);
        if (ready == 0)
            return false;
        if (ready < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        char value;
        ssize_t result = read(fd, &value, 1);
        if (result == 1) {
            if (value == '\n')
                return true;
            line->push_back(value);
            continue;
        }
        if (result < 0 && errno == EINTR)
            continue;
        return false;
    }
    return false;
}

static bool parseUnsigned(const std::string& text, uint64_t *value) {
    if (text.empty())
        return false;
    uint64_t result = 0;
    for (char character : text) {
        if (character < '0' || character > '9' ||
                result > (UINT64_MAX - unsigned(character - '0')) / 10)
            return false;
        result = result * 10 + unsigned(character - '0');
    }
    *value = result;
    return true;
}

struct ChildSession {
    std::string path;
    uint64_t processId = 0;
    std::string runId;
};

static bool outputIdentity(const std::string& path, ChildSession *session) {
    const size_t pidStart = path.rfind(".pid-");
    const size_t runStart = path.rfind(".run-");
    const size_t runEnd = path.rfind(".jsonl");
    if (pidStart == std::string::npos || runStart == std::string::npos ||
            runEnd == std::string::npos || pidStart >= runStart || runStart >= runEnd ||
            runEnd + 6 != path.size())
        return false;
    session->path = path;
    session->runId = path.substr(runStart + 5, runEnd - (runStart + 5));
    return !session->runId.empty() && parseUnsigned(path.substr(pidStart + 5,
            runStart - (pidStart + 5)), &session->processId);
}

static bool jsonUnsignedMember(const JsonValue& record, const char *name, uint64_t *value) {
    const JsonValue *member = record.member(name);
    return member && member->type == JsonType::Number && parseUnsigned(member->text, value);
}

static bool jsonStringMember(const JsonValue& record, const char *name, std::string *value) {
    const JsonValue *member = record.member(name);
    if (!member || member->type != JsonType::String)
        return false;
    *value = member->text;
    return true;
}

static bool validateChildOutput(const std::string& path, ChildSession *session) {
    bool valid = outputIdentity(path, session);
    check(valid, "child output filename did not contain a valid session identity");
    if (!valid)
        return false;

    std::vector<JsonValue> records;
    for (const std::string& line : readLines(path)) {
        if (line.empty())
            continue;
        try {
            JsonValue record = JsonParser(line).parse();
            valid = record.type == JsonType::Object;
            check(valid, "child telemetry JSONL record was not a JSON object");
            if (valid)
                records.push_back(std::move(record));
        } catch (const std::exception&) {
            check(false, "child telemetry JSONL contained malformed JSON");
            valid = false;
        }
    }
    check(records.size() == 3, "child telemetry output did not preserve its three framing records");
    valid &= records.size() == 3;

    size_t runCount = 0;
    size_t summaryCount = 0;
    size_t summaryIndex = records.size();
    const JsonValue *run = nullptr;
    const JsonValue *summary = nullptr;
    for (size_t i = 0; i < records.size(); ++i) {
        std::string recordType;
        bool hasRecordType = jsonStringMember(records[i], "record_type", &recordType);
        check(hasRecordType, "child telemetry JSONL record omitted its string record_type");
        valid &= hasRecordType;
        if (recordType == "RUN") {
            ++runCount;
            run = &records[i];
        } else if (recordType == "SUMMARY") {
            ++summaryCount;
            summary = &records[i];
            summaryIndex = i;
        }

        const JsonValue *attributedRun = records[i].member("run_id");
        if (attributedRun) {
            bool ownRun = attributedRun->type == JsonType::String &&
                    attributedRun->text == session->runId;
            check(ownRun, "child telemetry output contained another session run_id");
            valid &= ownRun;
        }
        const JsonValue *attributedProcess = records[i].member("process_id");
        if (attributedProcess) {
            uint64_t processId = 0;
            bool ownProcess = attributedProcess->type == JsonType::Number &&
                    parseUnsigned(attributedProcess->text, &processId) &&
                    processId == session->processId;
            check(ownProcess, "child telemetry output contained another session process_id");
            valid &= ownProcess;
        }
    }
    check(runCount == 1, "child telemetry output did not contain exactly one RUN");
    check(summaryCount == 1, "child telemetry output did not contain exactly one SUMMARY");
    check(summaryIndex + 1 == records.size(),
            "child telemetry output contained a record after SUMMARY");
    valid &= runCount == 1 && summaryCount == 1 && summaryIndex + 1 == records.size();

    uint64_t runProcessId = 0;
    std::string runId;
    std::string summaryRunId;
    bool runIdentity = run && jsonUnsignedMember(*run, "process_id", &runProcessId) &&
            jsonStringMember(*run, "run_id", &runId) && runProcessId == session->processId &&
            runId == session->runId;
    bool summaryIdentity = summary && jsonStringMember(*summary, "run_id", &summaryRunId) &&
            summaryRunId == runId;
    check(runIdentity, "filename PID/run_id did not match parsed RUN identity");
    check(summaryIdentity, "SUMMARY run_id did not match parsed RUN identity");
    return valid && runIdentity && summaryIdentity;
}

static int telemetryChild(const char *path, const char *role, int signalFd, int releaseFd) {
    std::string output;
    {
        Session session(path, 3, 64);
        if (!session.enabled())
            return 1;
        Event event;
        event.deviceId = 1;
        if (!session.publish(event))
            return 2;
        output = session.outputPath();
        if (role[0] == 'a') {
            if (!writeAll(signalFd, output + "\n"))
                return 3;
            char release;
            ssize_t result;
            do {
                result = read(releaseFd, &release, 1);
            } while (result < 0 && errno == EINTR);
            if (result != 1)
                return 4;
        }
    }
    return role[0] == 'b' && !writeAll(signalFd, output + "\n") ? 5 : 0;
}

static void testMultiProcessOutputIdentity(const char *self) {
    std::string path = tempPath("vkd3d-telemetry-multiprocess-") +
            std::to_string(getpid()) + ".jsonl";
    for (const auto& output : outputPathsForBase(path))
        std::remove(output.c_str());

    int readyA[2] = {-1, -1};
    int releaseA[2] = {-1, -1};
    int doneB[2] = {-1, -1};
    bool pipesReady = pipe(readyA) == 0 && pipe(releaseA) == 0 && pipe(doneB) == 0;
    check(pipesReady, "could not create child-process synchronization pipes");
    if (!pipesReady)
        return;

    pid_t first = fork();
    if (first == 0) {
        close(readyA[0]);
        close(releaseA[1]);
        close(doneB[0]);
        close(doneB[1]);
        std::string readyFd = std::to_string(readyA[1]);
        std::string releaseFd = std::to_string(releaseA[0]);
        execl(self, self, "--telemetry-child", path.c_str(), "a", readyFd.c_str(),
                releaseFd.c_str(), nullptr);
        _exit(127);
    }
    check(first > 0, "could not create telemetry child A");
    close(readyA[1]);
    close(releaseA[0]);
    if (first <= 0) {
        close(releaseA[1]);
        close(doneB[0]);
        close(doneB[1]);
        return;
    }

    std::string firstOutput;
    bool ready = readLineWithTimeout(readyA[0], &firstOutput);
    close(readyA[0]);
    check(ready && !firstOutput.empty(),
            "child A did not prove an active telemetry session before child B launch");
    if (!ready || firstOutput.empty()) {
        close(releaseA[1]);
        int status = 0;
        waitpid(first, &status, 0);
        close(doneB[1]);
        close(doneB[0]);
        return;
    }

    struct stat beforeB = {};
    bool firstExistsBeforeB = stat(firstOutput.c_str(), &beforeB) == 0 && S_ISREG(beforeB.st_mode);
    check(firstExistsBeforeB, "child A active output did not exist before child B initialized");

    pid_t second = fork();
    if (second == 0) {
        close(doneB[0]);
        close(releaseA[1]);
        std::string doneFd = std::to_string(doneB[1]);
        execl(self, self, "--telemetry-child", path.c_str(), "b", doneFd.c_str(), "-1", nullptr);
        _exit(127);
    }
    check(second > 0, "could not create telemetry child B after child A became ready");
    close(doneB[1]);

    std::string secondOutput;
    bool done = second > 0 && readLineWithTimeout(doneB[0], &secondOutput);
    close(doneB[0]);
    check(done && !secondOutput.empty(),
            "child B did not finalize telemetry while child A was held alive");
    int firstStatus = 0;
    check(waitpid(first, &firstStatus, WNOHANG) == 0,
            "child A exited before child B completed and before release");

    struct stat afterB = {};
    bool firstSurvivedB = firstExistsBeforeB && stat(firstOutput.c_str(), &afterB) == 0 &&
            beforeB.st_dev == afterB.st_dev && beforeB.st_ino == afterB.st_ino;
    check(firstSurvivedB, "child B destructively replaced child A active output");
    auto outputsDuringOverlap = outputPathsForBase(path);
    check(outputsDuringOverlap.size() == 2 &&
            std::find(outputsDuringOverlap.begin(), outputsDuringOverlap.end(), firstOutput) !=
                    outputsDuringOverlap.end() &&
            std::find(outputsDuringOverlap.begin(), outputsDuringOverlap.end(), secondOutput) !=
                    outputsDuringOverlap.end(),
            "both distinct child outputs did not coexist while child A was held");

    ChildSession secondSession;
    bool secondValid = done && !secondOutput.empty() &&
            validateChildOutput(secondOutput, &secondSession);
    int secondStatus = 0;
    if (second > 0)
        check(waitpid(second, &secondStatus, 0) == second && WIFEXITED(secondStatus) &&
                WEXITSTATUS(secondStatus) == 0, "child B failed after its completion signal");

    check(writeAll(releaseA[1], "R"), "could not release child A after child B completion");
    close(releaseA[1]);
    check(waitpid(first, &firstStatus, 0) == first && WIFEXITED(firstStatus) &&
            WEXITSTATUS(firstStatus) == 0, "child A failed after explicit release");

    ChildSession firstSession;
    bool firstValid = validateChildOutput(firstOutput, &firstSession);
    check(firstValid && secondValid && firstOutput != secondOutput &&
            (firstSession.processId != secondSession.processId ||
                    firstSession.runId != secondSession.runId),
            "child outputs did not have distinct parsed session identities");
}
#endif

int main(int argc, char **argv) {
#ifndef _WIN32
    if (argc == 6 && !std::strcmp(argv[1], "--telemetry-child"))
        return telemetryChild(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]));
#endif
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
    testProcessPreExitTransientZeroOwners();
    testProcessPreExitAcquireRace();
    testProcessFinalizerOwnerRace();
    testOwnerAcquireFinalOwnerRace();
    testOwnerConstructionUnwind();
    testOutputIdentityPathShapes();
    testDistinctOutputsDoNotTruncate();
#ifndef _WIN32
    testMultiProcessOutputIdentity(argv[0]);
#endif
    int failureCount = failures.load(std::memory_order_relaxed);
    if (failureCount)
        std::fprintf(stderr, "%d telemetry test failure(s).\n", failureCount);
    else
        std::printf("framepacer telemetry transport tests passed.\n");
    return failureCount != 0;
}
