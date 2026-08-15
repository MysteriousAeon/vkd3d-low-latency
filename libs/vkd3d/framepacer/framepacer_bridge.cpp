#include "framepacer_bridge.h"
#include "device.h"
#include "command_queue.h"
#include "submit_iterator.h"
#include "vulkan_queue.h"
#include "nvapi_pacing_adapter.h"
#include "sleep_value_filter.h"
#include "util/util_log.h"
#include "util/util_debug.h"
#include "vkd3d_dxgi1_2.h"
#include "vkd3d_platform.h"

#include <exception>

using namespace pacer;

#define DEVICE(x) ((Device*) x)
#define COMMAND_QUEUE(x) ((CommandQueue*) x)
#define VULKAN_QUEUE(x) ((VulkanQueue*) x)
static std::atomic<bool> g_NvApi_sleepEnabled;
static SleepValueFilter g_sleepValueFilter;

bool pacer_is_running( pacer_device_handle device ) {
    assert(device);

    uint64_t accountingState = DEVICE(device)->m_pacer->getAccountingState();
    if (FramePacer::isReflexAccountingState(accountingState))
        return true;

    return DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.isActive();
}

pacer_device_handle pacer_create_device( struct pacer_device_properties* properties, struct pacer_device_vk_procs* vk_procs ) {
    try {
        Debug debug("pacer_create_device");
        if (!properties->khrCalibratedTimestamps) {
            ERR( "VK_KHR_calibrated_timestamps extension is required for frame-pacer device \n" );
            return nullptr;
        }
#ifdef VKD3D_ENABLE_TEST_HOOKS
        char test_hook[2];
        if (vkd3d_get_env_var("VKD3D_TEST_FAIL_PACER_DEVICE_CREATION",
                test_hook, sizeof(test_hook)) && test_hook[0] == '1')
            g_testFailNvApiPacingAdapterConstruction.store(true,
                    std::memory_order_release);
#endif
        return (pacer_device_handle) new Device(properties, vk_procs);
    } catch (const std::exception& exception) {
        ERR( "Failed to create frame-pacer device: %s \n", exception.what() );
        return nullptr;
    } catch (...) {
        ERR( "Failed to create frame-pacer device due to an unknown exception \n" );
        return nullptr;
    }
}

void pacer_destroy_device( pacer_device_handle handle ) {
    Debug debug("pacer_destroy_device");
    delete DEVICE(handle);
}

pacer_queues pacer_register_queues( pacer_device_handle handle,
        void* command_queue, D3D12_COMMAND_LIST_TYPE type,
        void* vkd3d_queue, struct pacer_vulkan_queue_info vulkan_queue_info) {
    assert(handle);
    return DEVICE(handle)->registerQueues(command_queue, type, vkd3d_queue, vulkan_queue_info);
}

void pacer_register_swapchain( pacer_device_handle handle, void* vkd3d_swapchain, void* vkd3d_command_queue, DXGI_SWAP_CHAIN_DESC1 desc, vkd3d_native_sync_handle* latency_event) {
    assert(handle);
    DEVICE(handle)->registerSwapchain(vkd3d_swapchain, vkd3d_command_queue, desc, latency_event);
}

void pacer_unregister_swapchain( pacer_device_handle handle, void* vkd3d_swapchain ) {
    assert(handle);
    DEVICE(handle)->unregisterSwapchain(vkd3d_swapchain);
    DEVICE(handle)->m_pacer->m_simulationLedger.unregisterSwapchain(vkd3d_swapchain);
}

uint64_t pacer_command_queue_notify_submit( pacer_command_queue_handle command_queue ) {
    assert(command_queue);
    return COMMAND_QUEUE(command_queue)->notifySubmit();
}

uint64_t pacer_command_queue_notify_legacy_submit(
        pacer_command_queue_handle command_queue) {
    assert(command_queue);
    return COMMAND_QUEUE(command_queue)->notifyLegacySubmit();
}

static pacer_command_capture_token export_capture_token(
        const CaptureToken& token) {
    return {token.accountingEpoch, token.captureGeneration,
            token.simulationId, token.publicationLeaseId,
            token.externalReflexId};
}

static CaptureToken import_capture_token(
        const pacer_command_capture_token& token) {
    return {token.accounting_epoch, token.capture_generation,
            token.simulation_id, token.publication_lease_id,
            token.external_reflex_id};
}

pacer_command_capture_lease pacer_command_queue_acquire_capture(
        pacer_command_queue_handle command_queue) {
    pacer_command_capture_lease lease = {};
    if (!command_queue) {
        lease.acquire_result = PACER_CAPTURE_NOT_REFLEX;
        return lease;
    }
    try {
        CaptureLeaseAcquisition acquisition =
                COMMAND_QUEUE(command_queue)->acquireCaptureLease();
        lease.acquire_result = static_cast<uint32_t>(acquisition.result);
        lease.token = export_capture_token(acquisition.token);
        lease.active = acquisition.result == CaptureAcquireResult::Acquired;
    } catch (...) {
        COMMAND_QUEUE(command_queue)->m_device->m_pacer->forceReflexPacingBypass(
                telemetry::FailureReason::BridgeCaptureException);
        lease.acquire_result = PACER_CAPTURE_NO_OPEN;
    }
    return lease;
}

uint64_t pacer_command_queue_commit_capture(
        pacer_command_queue_handle command_queue,
        pacer_command_capture_lease *lease) {
    uint64_t result = 0;
    assert(command_queue);
    if (!lease || !lease->active)
        return 0;
    CaptureToken token = import_capture_token(lease->token);
    /* The caller-local guard is consumed even when ledger validation rejects
     * the token. The ledger lease map prevents any double decrement. */
    lease->active = false;
    try {
        result = COMMAND_QUEUE(command_queue)->commitCapturedSubmit(token);
    } catch (...) {
        FirstFailureContext context;
        context.simulationId = token.simulationId;
        context.captureGeneration = token.captureGeneration;
        context.externalReflexId = token.externalReflexId;
        context.contextId = token.publicationLeaseId;
        COMMAND_QUEUE(command_queue)->m_device->m_pacer->forceReflexPacingBypass(
                telemetry::FailureReason::BridgeCaptureException, context);
    }
    return result;
}

void pacer_command_queue_retire_capture(
        pacer_command_queue_handle command_queue,
        pacer_command_capture_lease *lease,
        pacer_capture_retire_reason reason) {
    assert(command_queue);
    if (!lease || !lease->active)
        return;
    CaptureToken token = import_capture_token(lease->token);
    lease->active = false;
    try {
        (void)reason;
        COMMAND_QUEUE(command_queue)->retireCaptureLease(token,
                CaptureRetireReason::BenignAbort);
    } catch (...) {
        FirstFailureContext context;
        context.simulationId = token.simulationId;
        context.captureGeneration = token.captureGeneration;
        context.externalReflexId = token.externalReflexId;
        context.contextId = token.publicationLeaseId;
        COMMAND_QUEUE(command_queue)->m_device->m_pacer->forceReflexPacingBypass(
                telemetry::FailureReason::BridgeCaptureException, context);
    }
}

bool pacer_command_queue_notify_vulkan_submit( pacer_command_queue_handle command_queue, uint64_t command_submit_id, uint64_t vulkan_submit_id ) {
    assert(command_queue);
    return COMMAND_QUEUE(command_queue)->notifyVulkanSubmit(command_submit_id, vulkan_submit_id);
}


void pacer_queue_notify_gpu_execution_end( struct pacer_queues pacer_queues, uint64_t command_submit_id, uint64_t vulkan_submit_id, struct pacer_query_pool* query_pool ) {
    CommandQueue* commandQueue = COMMAND_QUEUE(pacer_queues.command_queue);
    VulkanQueue* vulkanQueue = VULKAN_QUEUE(pacer_queues.vulkan_queue);
    assert( commandQueue );
    assert( vulkanQueue );
    uint64_t gpuTimestamp = vulkanQueue->notifyGpuExecutionEnd(vulkan_submit_id, query_pool);
    commandQueue->notifyVulkanGpuExecutionEnd(command_submit_id, vulkan_submit_id, gpuTimestamp);
    vulkanQueue->finishSubmit(vulkan_submit_id);
}

void pacer_queue_notify_submit_failed( struct pacer_queues pacer_queues,
        uint64_t command_submit_id, uint64_t vulkan_submit_id ) {
    CommandQueue* commandQueue = COMMAND_QUEUE(pacer_queues.command_queue);
    VulkanQueue* vulkanQueue = VULKAN_QUEUE(pacer_queues.vulkan_queue);

    if (commandQueue)
        commandQueue->notifySubmitFailed(command_submit_id, vulkan_submit_id);
    if (vulkanQueue && vulkan_submit_id)
        vulkanQueue->abandonSubmit(vulkan_submit_id);
}

void pacer_command_queue_set_observed_oob_role(
        pacer_command_queue_handle command_queue, uint32_t type) {
    if (!command_queue)
        return;
    telemetry::QueueRole role = telemetry::QueueRole::Unknown;
    if (type == 0)
        role = telemetry::QueueRole::OobRender;
    else if (type == 1)
        role = telemetry::QueueRole::OobPresent;
    COMMAND_QUEUE(command_queue)->setTelemetryQueueRole(role);
}


uint64_t pacer_vulkan_queue_notify_submit( pacer_vulkan_queue_handle vulkan_queue ) {
    assert(vulkan_queue);
    return VULKAN_QUEUE(vulkan_queue)->notifySubmit();
}

struct pacer_query_pool* pacer_vulkan_queue_alloc_query_pool( pacer_vulkan_queue_handle vulkan_queue ) {
    assert(vulkan_queue);
    return VULKAN_QUEUE(vulkan_queue)->allocQueryPool();
}

void pacer_vulkan_queue_free_query_pool( pacer_vulkan_queue_handle vulkan_queue,  struct pacer_query_pool* query_pool) {
    assert(vulkan_queue);
    VULKAN_QUEUE(vulkan_queue)->freeQueryPool( query_pool );
}

struct pacer_query_pool* pacer_vulkan_queue_alloc_query_pool_top_of_pipe( pacer_vulkan_queue_handle vulkan_queue ) {
    assert(vulkan_queue);
    return VULKAN_QUEUE(vulkan_queue)->allocQueryPoolTopOfPipe();
}

void pacer_vulkan_queue_push_query_pool_top_of_pipe( pacer_vulkan_queue_handle vulkan_queue, struct pacer_query_pool* query_pool, uint64_t vulkan_submit_id, bool push_into_queue ) {
    assert(vulkan_queue);
    VULKAN_QUEUE(vulkan_queue)->pushQueryPoolTopOfPipe(query_pool, vulkan_submit_id, push_into_queue);
}

void NvAPI_setSleepMode( pacer_device_handle handle, bool enable, UINT32 minimum_interval_us ) {
    g_NvApi_sleepEnabled.store( enable, std::memory_order_release );
    DEVICE(handle)->m_pacer->setReflexMode(enable);

    // games (UE5!) seem to spam garbage which we need to filter
    g_sleepValueFilter.push(minimum_interval_us);
    uint32_t minInterval = g_sleepValueFilter.getMinInterval();

    if (handle)
        DEVICE(handle)->m_pacer->setFpsLimit(minInterval);
}

void NvAPI_setLatencyMarker( pacer_device_handle handle, uint64_t frameId, VkLatencyMarkerNV marker ) {
    assert(handle);
    try {
        DEVICE(handle)->m_nvApi_pacingAdapter->setLatencyMarker(frameId, marker);
    } catch (...) {
        DEVICE(handle)->m_pacer->forceReflexPacingBypass(
                telemetry::FailureReason::BridgeMarkerException);
    }
}

void NvAPI_sleep( pacer_device_handle handle ) {
    assert(handle);
    if (g_NvApi_sleepEnabled)
        DEVICE(handle)->m_nvApi_pacingAdapter->sleepAndBeginFrame();
}

static PresentAttemptToken import_present_attempt(
        const pacer_present_attempt_token& attempt) {
    return {attempt.accounting_epoch, attempt.simulation_id,
            attempt.attempt_generation, attempt.thread_id};
}

static pacer_present_attempt_token export_present_attempt(
        const PresentAttemptToken& attempt) {
    return {attempt.accountingEpoch, attempt.simulationId,
            attempt.attemptGeneration, attempt.threadId};
}

pacer_present_attempt_token pacer_begin_present_attempt(
        pacer_device_handle device) {
    assert(device);
    return export_present_attempt(DEVICE(device)->m_pacer->capturePresentAttempt());
}

uint64_t pacer_notify_present( pacer_device_handle device, void* vkd3d_swapchain,
        uint64_t presentation_sequence, pacer_present_attempt_token rawAttempt,
        uint64_t dxgiPresentEntryNs ) {
    assert(device);
    assert(vkd3d_swapchain);
    CommandQueue* commandQueue = DEVICE(device)->m_primaryCommandQueue;
    PresentAttemptToken attempt = import_present_attempt(rawAttempt);
    uint64_t currentState = DEVICE(device)->m_pacer->getAccountingState();
    if (telemetry::isEnabled()) {
    telemetry::Event presentEvent;
    presentEvent.type = telemetry::Type::Present;
    presentEvent.deviceId = DEVICE(device)->m_telemetryId;
    presentEvent.phase = telemetry::Phase::DxgiEntry;
    presentEvent.epochId = attempt.accountingEpoch;
    presentEvent.simulationId = attempt.simulationId;
    presentEvent.externalReflexId = DEVICE(device)->m_pacer->getExternalFrameId(
            attempt.accountingEpoch, attempt.simulationId);
    presentEvent.id0 = DEVICE(device)->getSwapchainTelemetryId(vkd3d_swapchain);
    presentEvent.id1 = presentation_sequence;
    presentEvent.timestamp0 = dxgiPresentEntryNs;
    telemetry::emit(presentEvent);
    }

    if (!attempt || attempt.threadId != dxvk::this_thread::get_id()) {
        FirstFailureContext context;
        const uint32_t callerThreadId = dxvk::this_thread::get_id();
        context.simulationId = attempt.simulationId;
        context.contextId = attempt.attemptGeneration;
        context.contextValue0 = attempt.accountingEpoch;
        context.contextValue1 = presentation_sequence;
        context.contextValue2 = DEVICE(device)->getSwapchainTelemetryId(vkd3d_swapchain);
        context.contextCount0 = attempt.threadId;
        context.contextCount1 = callerThreadId;
        if (attempt)
            context.flags |= telemetry::FirstFailurePresentTokenProvided;
        if (attempt.threadId == callerThreadId)
            context.flags |= telemetry::FirstFailureCallerTokenThreadMatch;
        DEVICE(device)->m_pacer->forceReflexPacingBypass(!attempt
                ? telemetry::FailureReason::InvalidPresentAttemptZeroToken
                : telemetry::FailureReason::InvalidPresentAttemptThreadMismatch,
                context);
        DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.releaseSemaphore(
                vkd3d_swapchain, 1);
    } else if (FramePacer::isReflexAccountingState(attempt.accountingEpoch)) {
        if (currentState == attempt.accountingEpoch && commandQueue &&
                DEVICE(device)->m_activeSwapchain == vkd3d_swapchain) {
            if (DEVICE(device)->m_pacer->notifyReflexPresent(attempt,
                    vkd3d_swapchain, presentation_sequence))
                DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.sleep(
                        vkd3d_swapchain, attempt.accountingEpoch);
            else
                DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.releaseSemaphore(
                        vkd3d_swapchain, 1);
        } else {
            DEVICE(device)->m_pacer->cancelReflexPresent(attempt);
            DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.releaseSemaphore(
                    vkd3d_swapchain, 1);
        }
    } else if (currentState == attempt.accountingEpoch && commandQueue &&
            DEVICE(device)->m_activeSwapchain == vkd3d_swapchain) {
        commandQueue->notifyLegacyPresent(vkd3d_swapchain,
                attempt.accountingEpoch);
    } else {
        DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.releaseSemaphore(
                vkd3d_swapchain, 1);
    }

    // return a frame_id when we'll make use of it (present_timing, present_wait, etc.)
    return 0;
}

void pacer_notify_vk_present(pacer_device_handle device, void* vkd3d_swapchain,
        uint64_t presentation_sequence, uint64_t accounting_epoch,
        uint64_t simulation_id, uint64_t external_reflex_id) {
    if (!device || !vkd3d_swapchain)
        return;
    telemetry::Event event;
    event.type = telemetry::Type::Present;
    event.deviceId = DEVICE(device)->m_telemetryId;
    event.phase = telemetry::Phase::VkPresent;
    event.epochId = accounting_epoch;
    event.simulationId = simulation_id;
    event.externalReflexId = external_reflex_id ? external_reflex_id
            : DEVICE(device)->m_pacer->getExternalFrameId(
                    accounting_epoch, simulation_id);
    event.id0 = DEVICE(device)->getSwapchainTelemetryId(vkd3d_swapchain);
    event.id1 = presentation_sequence;
    telemetry::emit(event);
}

uint64_t pacer_telemetry_now_ns(void) {
    return telemetry::nowNs();
}

bool pacer_telemetry_enabled(void) {
    return telemetry::isEnabled();
}

void pacer_notify_aborted_present( pacer_device_handle device,
        void* vkd3d_swapchain, pacer_present_attempt_token rawAttempt ) {
    assert(device);
    assert(vkd3d_swapchain);

    PresentAttemptToken attempt = import_present_attempt(rawAttempt);
    if (!attempt || attempt.threadId != dxvk::this_thread::get_id()) {
        FirstFailureContext context;
        const uint32_t callerThreadId = dxvk::this_thread::get_id();
        context.simulationId = attempt.simulationId;
        context.contextId = attempt.attemptGeneration;
        context.contextValue0 = attempt.accountingEpoch;
        context.contextValue2 = DEVICE(device)->getSwapchainTelemetryId(vkd3d_swapchain);
        context.contextCount0 = attempt.threadId;
        context.contextCount1 = callerThreadId;
        if (attempt)
            context.flags |= telemetry::FirstFailurePresentTokenProvided;
        if (attempt.threadId == callerThreadId)
            context.flags |= telemetry::FirstFailureCallerTokenThreadMatch;
        DEVICE(device)->m_pacer->forceReflexPacingBypass(
                telemetry::FailureReason::InvalidAbortedPresentToken, context);
    }
    else if (FramePacer::isReflexAccountingState(attempt.accountingEpoch))
        DEVICE(device)->m_pacer->cancelReflexPresent(attempt);

    DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.releaseSemaphore( vkd3d_swapchain, 1 );
}

#ifdef VKD3D_ENABLE_TEST_HOOKS
namespace pacer {
    extern std::atomic<bool> g_testLastTeardownWorkerStopped;
}

void pacer_test_set_wait_latency(void *device, uint32_t wait_latency) {
    assert(device);
    DEVICE(device)->m_pacer->m_frameSync.m_waitLatency = wait_latency;
}

void pacer_test_set_latency_sleep_bypass(void *device, bool bypass) {
    assert(device);
    DEVICE(device)->m_pacer->m_testSleepBypass = bypass;
}

void pacer_test_reset_latency_sleep_decision(void *device) {
    assert(device);
    DEVICE(device)->m_pacer->m_testSleepDecision = {};
    DEVICE(device)->m_pacer->m_testSleepEntryCount.store(0,
            std::memory_order_release);
}

void pacer_test_get_latency_sleep_decision(void *device,
        vkd3d_test_latency_sleep_decision *decision) {
    assert(device);
    assert(decision);
    *decision = DEVICE(device)->m_pacer->m_testSleepDecision;
}

void pacer_test_sleep_for_frame(void *device, uint64_t frame_id) {
    assert(device);
    DEVICE(device)->m_pacer->sleep(frame_id, {});
}

uint64_t pacer_test_get_sleep_entry_count(void *device) {
    assert(device);
    return DEVICE(device)->m_pacer->m_testSleepEntryCount.load(
            std::memory_order_acquire);
}

void pacer_test_pause_after_sleep(void *device, bool pause) {
    assert(device);
    DEVICE(device)->m_nvApi_pacingAdapter->testPauseAfterSleep(pause);
}

bool pacer_test_sleep_returned(void *device) {
    assert(device);
    return DEVICE(device)->m_nvApi_pacingAdapter->testSleepReturned();
}

void pacer_test_get_nvapi_adapter_state(void *device,
        vkd3d_test_nvapi_adapter_state *state) {
    assert(device);
    assert(state);
    DEVICE(device)->m_nvApi_pacingAdapter->testGetState(state);
}

void pacer_test_set_prediction(void *device, uint64_t frame_id,
        int32_t optimized_gpu_time) {
    FramePacer *framepacer = DEVICE(device)->m_pacer.get();
    framepacer->testSetPrediction(frame_id, optimized_gpu_time);
}

void pacer_test_get_prediction_state(void *device,
        vkd3d_test_prediction_state *state) {
    FramePacer *framepacer = DEVICE(device)->m_pacer.get();
    assert(state);
    framepacer->testGetPredictionState(state);
}

bool pacer_test_get_last_teardown_worker_stopped(void) {
    return pacer::g_testLastTeardownWorkerStopped.load(std::memory_order_acquire);
}

void pacer_test_set_frame_state(void *device, uint64_t frame_id,
        uint64_t external_frame_id, int32_t marker_value) {
    FramePacer *framepacer = DEVICE(device)->m_pacer.get();
    uint64_t accountingState = framepacer->getAccountingState();
    framepacer->m_frameMapping.registerMapping(accountingState, frame_id,
            external_frame_id);
    framepacer->m_latencyMarkers.updateMarkers(accountingState, frame_id,
            [&](LatencyMarkers& markers) { markers.renderStart = marker_value; });
}

void pacer_test_get_frame_state(void *device, uint64_t frame_id,
        vkd3d_test_frame_state *state) {
    FramePacer *framepacer = DEVICE(device)->m_pacer.get();
    LatencyMarkers markers;

    assert(state);
    state->accounting_state = framepacer->getAccountingState();
    state->frame_id = frame_id;
    state->external_frame_id = framepacer->m_frameMapping.getFrameId(
            state->accounting_state, frame_id);
    markers = framepacer->m_latencyMarkers.getMarkers(
            state->accounting_state, frame_id);
    state->marker_present = markers.simulationId == frame_id;
    state->marker_value = markers.renderStart;
}

void pacer_test_queue_waitable_task(void *device, uint64_t accounting_state,
        uint64_t cpu_id) {
    DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.testQueueTask(
            accounting_state, cpu_id);
}

void pacer_test_wait_waitable_task_dequeued(void *device) {
    DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.testWaitDequeued();
}

void pacer_test_resume_waitable_task(void *device) {
    DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.testResumeTask();
}

void pacer_test_wait_waitable_task_idle(void *device) {
    DEVICE(device)->m_pacer->m_waitableDxgiSwapchain.testWaitIdle();
}

void pacer_test_get_waitable_task_stats(void *device,
        vkd3d_test_waitable_task_stats *stats) {
    WaitableDXGISwapchain& waitable =
            DEVICE(device)->m_pacer->m_waitableDxgiSwapchain;
    assert(stats);
    stats->marker_reads = waitable.testMarkerReads();
    stats->marker_writes = waitable.testMarkerWrites();
    stats->rejected_tasks = waitable.testRejectedTasks();
}

bool pacer_test_device_construction_failure_cleanup(
        pacer_device_properties *properties, pacer_device_vk_procs *vk_procs) {
    pacer_device_handle device;

    g_testFailNvApiPacingAdapterConstruction.store(true,
            std::memory_order_release);
    try {
        device = pacer_create_device(properties, vk_procs);
    } catch (...) {
        g_testFailNvApiPacingAdapterConstruction.store(false,
                std::memory_order_release);
        return false;
    }
    g_testFailNvApiPacingAdapterConstruction.store(false,
            std::memory_order_release);

    if (device) {
        pacer_destroy_device(device);
        return false;
    }

    return g_testLastTeardownWorkerStopped.load(
            std::memory_order_acquire);
}

void pacer_test_get_framepacer_snapshot(void *device, void *command_queue_handle,
        void *vulkan_queue_handle, uint64_t command_submit_id,
        uint64_t vulkan_submit_id, uint64_t first_internal_frame_id,
        uint64_t second_internal_frame_id,
        vkd3d_test_framepacer_snapshot *snapshot) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    FramePacer *framepacer;

    assert(device);
    assert(command_queue);
    assert(vulkan_queue);
    assert(snapshot);

    framepacer = DEVICE(device)->m_pacer.get();
    snapshot->accounting_state = framepacer->getAccountingState();
    snapshot->reflex_accounting = FramePacer::isReflexAccountingState(
            snapshot->accounting_state);
    snapshot->cpu_finished = framepacer->m_frameSync.cpuFinished.load();
    snapshot->gpu_finished = framepacer->m_frameSync.gpuFinished.load();
    snapshot->command_submit_count = command_queue->testSubmitCount();
    snapshot->command_vulkan_submit_id = command_queue->testVulkanSubmitId(command_submit_id);
    snapshot->vulkan_gpu_timestamp = vulkan_queue->testGpuExecutionEnd(vulkan_submit_id);
    snapshot->first_internal_frame_id = first_internal_frame_id;
    snapshot->first_external_frame_id = framepacer->m_frameMapping.getFrameId(
            snapshot->accounting_state, first_internal_frame_id);
    snapshot->second_internal_frame_id = second_internal_frame_id;
    snapshot->second_external_frame_id = framepacer->m_frameMapping.getFrameId(
            snapshot->accounting_state, second_internal_frame_id);
    snapshot->submit_simulation_id = command_queue->testSubmitSimulation(command_submit_id);
    snapshot->presentation_count = framepacer->m_simulationLedger.testPresentationCount(
            snapshot->submit_simulation_id);
    snapshot->submit_pending = command_queue->testPresentPending(command_submit_id);
    snapshot->tracking_bypassed = framepacer->m_simulationLedger.shouldBypassPacing();
}

void pacer_test_get_capture_snapshot(void *device, uint64_t capture_generation,
        vkd3d_test_capture_snapshot *snapshot) {
    assert(device);
    assert(snapshot);
    CaptureSnapshot source = DEVICE(device)->m_pacer->m_simulationLedger
            .testCaptureSnapshot(capture_generation);
    snapshot->accounting_epoch = source.accountingEpoch;
    snapshot->simulation_id = source.simulationId;
    snapshot->capture_generation = source.captureGeneration;
    snapshot->capture_state = static_cast<uint32_t>(source.state);
    snapshot->failure_reason = static_cast<uint32_t>(source.failureReason);
    snapshot->in_flight_publications = source.inFlightPublications;
    snapshot->open_capture_count = source.openCaptureCount;
    snapshot->active_lease_count = source.activeLeaseCount;
    snapshot->capture_record_count = source.captureRecordCount;
    snapshot->end_association_count = source.endAssociationCount;
    snapshot->highest_started_external_id = source.highestStartedExternalId;
    snapshot->published_submits = source.publishedSubmits;
    snapshot->accounted_submits = source.accountedSubmits;
    snapshot->completed_submits = source.completedSubmits;
    snapshot->submission_seal_requested = source.submissionSealRequested;
    snapshot->cpu_seal_requested = source.cpuSealRequested;
    snapshot->submissions_sealed = source.submissionsSealed;
    snapshot->cpu_sealed = source.cpuSealed;
    snapshot->tracking_failed = source.trackingFailed;
    snapshot->cpu_finished = DEVICE(device)->m_pacer->m_frameSync.cpuFinished.load();
    snapshot->gpu_finished = DEVICE(device)->m_pacer->m_frameSync.gpuFinished.load();
}

bool pacer_test_accept_present(void *device, uint64_t external_reflex_id) {
    static char testSwapchain;
    FramePacer *framepacer = DEVICE(device)->m_pacer.get();

    NvAPI_setLatencyMarker(static_cast<pacer_device_handle>(device), external_reflex_id,
            VK_LATENCY_MARKER_PRESENT_START_NV);
    PresentAttemptToken attempt = framepacer->capturePresentAttempt();
    return attempt && framepacer->notifyReflexPresent(
            attempt, &testSwapchain, 1);
}

void pacer_test_set_command_ring_hook(void *command_queue_handle,
        vkd3d_test_command_ring_hook_callback callback, void *userdata) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    command_queue->setTestHook(callback, userdata);
}

void pacer_test_pause_command_ring_hook_dispatch(void *command_queue_handle) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    command_queue->pauseTestHookDispatch();
}

void pacer_test_wait_command_ring_hook_dispatch(void *command_queue_handle) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    command_queue->waitTestHookDispatchArrived();
}

void pacer_test_wait_command_ring_hook_cleared(void *command_queue_handle) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    command_queue->waitTestHookCleared();
}

void pacer_test_wait_command_ring_hook_drain(void *command_queue_handle) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    command_queue->waitTestHookDrain();
}

void pacer_test_resume_command_ring_hook_dispatch(void *command_queue_handle) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    command_queue->resumeTestHookDispatch();
}

void pacer_test_set_vulkan_ring_hook(void *vulkan_queue_handle,
        vkd3d_test_vulkan_ring_hook_callback callback, void *userdata) {
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    assert(vulkan_queue);
    vulkan_queue->setTestHook(callback, userdata);
}

void pacer_test_wait_vulkan_ring_hook_drain(void *vulkan_queue_handle) {
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    assert(vulkan_queue);
    vulkan_queue->waitTestHookDrain();
}

void pacer_test_set_vulkan_submit_timestamp(void *vulkan_queue_handle,
        uint64_t vulkan_id, uint64_t timestamp) {
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    assert(vulkan_queue);
    vulkan_queue->testSetSubmitTimestamp(vulkan_id, timestamp);
}

void pacer_test_set_vulkan_gpu_execution_start(void *vulkan_queue_handle,
        uint64_t vulkan_id, uint64_t timestamp) {
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    assert(vulkan_queue);
    vulkan_queue->testSetGpuExecutionStart(vulkan_id, timestamp);
}

void pacer_test_set_vulkan_gpu_execution_end(void *vulkan_queue_handle,
        uint64_t vulkan_id, uint64_t timestamp) {
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    assert(vulkan_queue);
    vulkan_queue->testSetGpuExecutionEnd(vulkan_id, timestamp);
}

void pacer_test_get_vulkan_slot_snapshot(void *vulkan_queue_handle,
        uint64_t vulkan_id, vkd3d_test_vulkan_slot_snapshot *snapshot) {
    VulkanQueue *vulkan_queue = VULKAN_QUEUE(vulkan_queue_handle);
    assert(vulkan_queue);
    assert(snapshot);
    snapshot->found = vulkan_queue->testSlotState(vulkan_id,
            &snapshot->generation, &snapshot->submit_timestamp,
            &snapshot->has_submit_timestamp,
            &snapshot->gpu_execution_start, &snapshot->gpu_execution_end,
            &snapshot->submit_accounted);
}

void pacer_test_get_command_slot_snapshot(void *command_queue_handle,
        uint64_t command_id, vkd3d_test_command_slot_snapshot *snapshot) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    assert(snapshot);
    command_queue->testSlotState(command_id, &snapshot->generation,
            &snapshot->vulkan_id, &snapshot->present_pending,
            &snapshot->present_epoch, &snapshot->submit_timestamp,
            &snapshot->has_submit_timestamp);
    snapshot->ledger_completion_accounted =
            command_queue->testSubmitCompletionAccounted(command_id);
}

bool pacer_test_set_command_present_state(void *command_queue_handle,
        uint64_t command_id, uint64_t present_epoch) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    return command_queue->testSetPresentState(command_id, present_epoch);
}

bool pacer_test_set_command_submit_timestamp(void *command_queue_handle,
        uint64_t command_id, uint64_t timestamp) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    return command_queue->testSetSubmitTimestamp(command_id, timestamp);
}

void pacer_test_get_submit_iterator_snapshot(void *command_queue_handle,
        uint64_t command_id, vkd3d_test_submit_iterator_snapshot *snapshot) {
    CommandQueue *command_queue = COMMAND_QUEUE(command_queue_handle);
    assert(command_queue);
    assert(snapshot);
    SubmitIterator iterator(command_queue, command_id);
    snapshot->vulkan_id = iterator.testCachedVulkanSubmitId();
    snapshot->vulkan_gpu_execution_start = iterator.getVulkanGpuExecutionStart();
    snapshot->vulkan_gpu_execution_end = iterator.getVulkanGpuExecutionEnd();
    snapshot->has_app_submit = iterator.getAppSubmit() !=
            dxvk::high_resolution_clock::time_point{};
    auto vulkan_submit = iterator.getVulkanSubmit();
    snapshot->vulkan_submit_timestamp = vulkan_submit.time_since_epoch().count();
    snapshot->has_vulkan_submit = vulkan_submit !=
            dxvk::high_resolution_clock::time_point{};
}
#endif
