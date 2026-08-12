#include "framepacer/framepacer_bridge.h"
#include "vkd3d_dxgi1_2.h"
#include "vkd3d_test_hooks.h"

#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

extern "C"
{
#include "config_flags.h"
union vkd3d_config_flags vkd3d_config_flags = {};
}

static constexpr uint64_t fixed_gpu_timestamp = 0x12345678ull;
static std::atomic<uintptr_t> next_handle = { 1 };
static unsigned int failures;

#define check(condition, ...) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "framepacer_accounting: "); \
        std::fprintf(stderr, __VA_ARGS__); \
        failures++; \
    } \
} while (0)

template<typename T>
static T make_handle()
{
    return (T)next_handle.fetch_add(1, std::memory_order_relaxed);
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_create_query_pool(VkDevice,
        const VkQueryPoolCreateInfo *, const VkAllocationCallbacks *, VkQueryPool *pool)
{
    *pool = make_handle<VkQueryPool>();
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_destroy_query_pool(VkDevice, VkQueryPool,
        const VkAllocationCallbacks *)
{
}

static VKAPI_ATTR void VKAPI_CALL fake_reset_query_pool(VkDevice, VkQueryPool,
        uint32_t, uint32_t)
{
}

static VKAPI_ATTR void VKAPI_CALL fake_cmd_reset_query_pool(VkCommandBuffer,
        VkQueryPool, uint32_t, uint32_t)
{
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_get_query_pool_results(VkDevice,
        VkQueryPool, uint32_t, uint32_t, size_t data_size, void *data,
        VkDeviceSize, VkQueryResultFlags)
{
    check(data_size >= sizeof(fixed_gpu_timestamp),
            "query result buffer is too small (%zu).\n", data_size);
    std::memcpy(data, &fixed_gpu_timestamp, sizeof(fixed_gpu_timestamp));
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_create_command_pool(VkDevice,
        const VkCommandPoolCreateInfo *, const VkAllocationCallbacks *, VkCommandPool *pool)
{
    *pool = make_handle<VkCommandPool>();
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_destroy_command_pool(VkDevice, VkCommandPool,
        const VkAllocationCallbacks *)
{
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_allocate_command_buffers(VkDevice,
        const VkCommandBufferAllocateInfo *info, VkCommandBuffer *buffers)
{
    uint32_t i;

    for (i = 0; i < info->commandBufferCount; i++)
        buffers[i] = make_handle<VkCommandBuffer>();
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_free_command_buffers(VkDevice, VkCommandPool,
        uint32_t, const VkCommandBuffer *)
{
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_begin_command_buffer(VkCommandBuffer,
        const VkCommandBufferBeginInfo *)
{
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_end_command_buffer(VkCommandBuffer)
{
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL fake_cmd_write_timestamp2(VkCommandBuffer,
        VkPipelineStageFlags2, VkQueryPool, uint32_t)
{
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_get_calibrated_timestamps(VkDevice,
        uint32_t count, const VkCalibratedTimestampInfoKHR *, uint64_t *timestamps,
        uint64_t *max_deviation)
{
    check(count == 1, "expected one calibrated timestamp, got %u.\n", count);
    timestamps[0] = fixed_gpu_timestamp - 1000;
    *max_deviation = 1;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL fake_get_calibrateable_time_domains(
        VkPhysicalDevice, uint32_t *count, VkTimeDomainKHR *domains)
{
    if (domains)
        domains[0] = VK_TIME_DOMAIN_DEVICE_KHR;
    *count = 1;
    return VK_SUCCESS;
}

enum class event_type
{
    present,
    gpu_completion,
};

struct fixture
{
    struct submit_pair
    {
        uint64_t command;
        uint64_t vulkan;
    };

    pacer_device_handle device = nullptr;
    pacer_queues queues = {};
    vkd3d_native_sync_handle latency_event = {};
    uint64_t command_submit_id = 0;
    uint64_t vulkan_submit_id = 0;
    uint64_t presentation_sequence = 0;
    char command_queue_token = 0;
    char vulkan_queue_token = 0;
    char swapchain_token = 0;

    explicit fixture(bool reflex_enabled = true, bool initial_submit = true)
    {
        pacer_device_properties properties = {};
        pacer_device_vk_procs procs = {};
        pacer_vulkan_queue_info queue_info = {};
        DXGI_SWAP_CHAIN_DESC1 desc = {};

        properties.vk_device = make_handle<VkDevice>();
        properties.timestamp_period = 1.0f;
        properties.khrCalibratedTimestamps = true;
        procs.vkCreateQueryPool = fake_create_query_pool;
        procs.vkDestroyQueryPool = fake_destroy_query_pool;
        procs.vkResetQueryPool = fake_reset_query_pool;
        procs.vkCmdResetQueryPool = fake_cmd_reset_query_pool;
        procs.vkGetQueryPoolResults = fake_get_query_pool_results;
        procs.vkCreateCommandPool = fake_create_command_pool;
        procs.vkDestroyCommandPool = fake_destroy_command_pool;
        procs.vkAllocateCommandBuffers = fake_allocate_command_buffers;
        procs.vkFreeCommandBuffers = fake_free_command_buffers;
        procs.vkBeginCommandBuffer = fake_begin_command_buffer;
        procs.vkEndCommandBuffer = fake_end_command_buffer;
        procs.vkCmdWriteTimestamp2 = fake_cmd_write_timestamp2;
        procs.vkGetCalibratedTimestampsKHR = fake_get_calibrated_timestamps;
        procs.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR =
                fake_get_calibrateable_time_domains;

        device = pacer_create_device(&properties, &procs);
        check(device != nullptr, "failed to create pacer device.\n");

        queue_info.timestamp_valid_bits = 64;
        queues = pacer_register_queues(device, &command_queue_token,
                D3D12_COMMAND_LIST_TYPE_DIRECT, &vulkan_queue_token, queue_info);
        check(queues.command_queue != nullptr && queues.vulkan_queue != nullptr,
                "failed to register production queues.\n");

        desc.Width = 1920;
        desc.Height = 1080;
        pacer_register_swapchain(device, &swapchain_token, &command_queue_token,
                desc, &latency_event);
        pacer_test_set_wait_latency(device, 1);
        pacer_test_set_latency_sleep_bypass(device, true);
        NvAPI_setSleepMode(device, reflex_enabled, 0);

        if (reflex_enabled)
        {
            /* Establish the production adapter's initial sleep/simulation state. */
            NvAPI_sleep(device);
            NvAPI_setLatencyMarker(device, 101, VK_LATENCY_MARKER_SIMULATION_START_NV);
            NvAPI_setLatencyMarker(device, 101, VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
        }
        pacer_test_reset_latency_sleep_decision(device);

        if (initial_submit)
        {
            command_submit_id = pacer_command_queue_notify_submit(queues.command_queue);
            vulkan_submit_id = pacer_vulkan_queue_notify_submit(queues.vulkan_queue);
            pacer_command_queue_notify_vulkan_submit(queues.command_queue,
                    command_submit_id, vulkan_submit_id);
            check(command_submit_id == 1 && vulkan_submit_id == 1,
                    "fresh fixture did not produce submit pair 1/1 (%" PRIu64 "/%" PRIu64 ").\n",
                    command_submit_id, vulkan_submit_id);
        }
    }

    ~fixture()
    {
        if (device)
        {
            pacer_unregister_swapchain(device, &swapchain_token);
            pacer_destroy_device(device);
        }
    }

    vkd3d_test_framepacer_snapshot snapshot(uint64_t command_id,
            uint64_t vulkan_id, uint64_t first_simulation = 2,
            uint64_t second_simulation = 3) const
    {
        vkd3d_test_framepacer_snapshot result = {};
        pacer_test_get_framepacer_snapshot(device, queues.command_queue,
                queues.vulkan_queue, command_id, vulkan_id,
                first_simulation, second_simulation, &result);
        return result;
    }

    vkd3d_test_framepacer_snapshot snapshot() const
    {
        return snapshot(command_submit_id, vulkan_submit_id);
    }

    submit_pair submit()
    {
        submit_pair result;
        result.command = pacer_command_queue_notify_submit(queues.command_queue);
        result.vulkan = pacer_vulkan_queue_notify_submit(queues.vulkan_queue);
        if (result.command && result.vulkan)
            check(pacer_command_queue_notify_vulkan_submit(queues.command_queue,
                    result.command, result.vulkan),
                    "failed to publish submit pair %" PRIu64 "/%" PRIu64 ".\n",
                    result.command, result.vulkan);
        return result;
    }

    void begin_simulation(uint64_t external_id)
    {
        NvAPI_sleep(device);
        NvAPI_setLatencyMarker(device, external_id,
                VK_LATENCY_MARKER_SIMULATION_START_NV);
        NvAPI_setLatencyMarker(device, external_id,
                VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
    }

    void enable_reflex(uint64_t external_id)
    {
        NvAPI_setSleepMode(device, true, 0);
        begin_simulation(external_id);
    }

    void disable_reflex()
    {
        NvAPI_setSleepMode(device, false, 0);
    }

    void present(uint64_t external_id = 101)
    {
        NvAPI_setLatencyMarker(device, external_id, VK_LATENCY_MARKER_PRESENT_START_NV);
        pacer_present_attempt_token attempt = pacer_begin_present_attempt(device);
        pacer_notify_present(device, &swapchain_token, ++presentation_sequence,
                attempt);
        NvAPI_setLatencyMarker(device, external_id, VK_LATENCY_MARKER_PRESENT_END_NV);
    }

    void legacy_present()
    {
        pacer_present_attempt_token attempt = pacer_begin_present_attempt(device);
        pacer_notify_present(device, &swapchain_token, ++presentation_sequence,
                attempt);
    }

    pacer_present_attempt_token begin_present_attempt(uint64_t external_id)
    {
        NvAPI_setLatencyMarker(device, external_id,
                VK_LATENCY_MARKER_PRESENT_START_NV);
        return pacer_begin_present_attempt(device);
    }

    void accept_present(pacer_present_attempt_token attempt,
            void *swapchain = nullptr)
    {
        pacer_notify_present(device, swapchain ? swapchain : &swapchain_token,
                ++presentation_sequence, attempt);
    }

    void abort_present(pacer_present_attempt_token attempt)
    {
        pacer_notify_aborted_present(device, &swapchain_token, attempt);
    }

    void complete_gpu(uint64_t command_id, uint64_t vulkan_id)
    {
        pacer_query_pool *query_pool = pacer_vulkan_queue_alloc_query_pool(queues.vulkan_queue);
        check(query_pool != nullptr, "failed to allocate production query-pool record.\n");
        pacer_queue_notify_gpu_execution_end(queues, command_id,
                vulkan_id, query_pool);
    }

    void complete_gpu()
    {
        complete_gpu(command_submit_id, vulkan_submit_id);
    }
};

static void check_common_snapshot(const char *name,
        const vkd3d_test_framepacer_snapshot& snapshot)
{
    check(snapshot.command_submit_count == 1,
            "%s: expected exactly one command submit, got %" PRIu64 ".\n",
            name, snapshot.command_submit_count);
    check(snapshot.command_vulkan_submit_id == 1,
            "%s: command submit is not mapped to Vulkan submit 1 (got %" PRIu64 ").\n",
            name, snapshot.command_vulkan_submit_id);
}

static void run_fg_case(const char *name, const event_type events[3])
{
    fixture f;
    vkd3d_test_framepacer_snapshot states[4] = {};
    vkd3d_test_latency_sleep_decision decision = {};
    unsigned int i;

    states[0] = f.snapshot();
    check_common_snapshot(name, states[0]);
    check(states[0].cpu_finished == 1 && states[0].gpu_finished == 1,
            "%s: fresh counters are not 1/1 (%" PRIu64 "/%" PRIu64 ").\n",
            name, states[0].cpu_finished, states[0].gpu_finished);
    check(states[0].submit_pending && !states[0].vulkan_gpu_timestamp &&
            states[0].submit_simulation_id == 2,
            "%s: fresh submit was not published for simulation 2.\n", name);

    for (i = 0; i < 3; i++)
    {
        if (events[i] == event_type::present)
            f.present();
        else
            f.complete_gpu();
        states[i + 1] = f.snapshot();
        check_common_snapshot(name, states[i + 1]);
    }

    check(states[3].cpu_finished - states[0].cpu_finished == 1,
            "%s: expected CPU delta 1, got %" PRIu64 ".\n", name,
            states[3].cpu_finished - states[0].cpu_finished);
    check(states[3].gpu_finished - states[0].gpu_finished == 1,
            "%s: expected GPU delta 1, got %" PRIu64 ".\n", name,
            states[3].gpu_finished - states[0].gpu_finished);
    check(states[3].first_external_frame_id == 101 &&
            states[3].second_external_frame_id == 0,
            "%s: dense simulation was not mapped to Reflex ID 101 (map 2->%" PRIu64
            ", 3->%" PRIu64 ").\n", name, states[3].first_external_frame_id,
            states[3].second_external_frame_id);
    check(!states[3].submit_pending && states[3].presentation_count == 2,
            "%s: submit was not accounted once or Presents were not retained as children.\n", name);
    check(states[3].vulkan_gpu_timestamp == fixed_gpu_timestamp,
            "%s: fixed GPU timestamp was not published (got %#" PRIx64 ").\n",
            name, states[3].vulkan_gpu_timestamp);

    pacer_test_reset_latency_sleep_decision(f.device);
    NvAPI_sleep(f.device);
    pacer_test_get_latency_sleep_decision(f.device, &decision);
    check(decision.call_count == 1 && decision.frame_id == 3 && decision.wait_id == 2,
            "%s: unexpected next sleep decision (%" PRIu64 ", frame %" PRIu64
            ", wait %" PRIu64 ").\n", name, decision.call_count,
            decision.frame_id, decision.wait_id);
    check(decision.cpu_finished == 2 && decision.gpu_finished == 2,
            "%s: sleep snapshot does not match accounting (%" PRIu64 "/%" PRIu64 ").\n",
            name, decision.cpu_finished, decision.gpu_finished);
    check(decision.wait_satisfied && decision.would_start_frame,
            "%s: next sleep branch did not reflect the GPU accounting.\n", name);

    std::printf("%s: CPU +1, GPU +1, next wait %" PRIu64 " %s.\n",
            name, decision.wait_id,
            decision.wait_satisfied ? "ready" : "blocked");

    if (events[0] == event_type::gpu_completion)
    {
        check(states[1].cpu_finished == 1 && states[1].gpu_finished == 1 &&
                !states[1].submit_pending &&
                states[1].vulkan_gpu_timestamp == fixed_gpu_timestamp,
                "%s: bad state after GPU completion.\n", name);
        check(states[2].cpu_finished == 2 && states[2].gpu_finished == 2 &&
                !states[2].submit_pending && states[2].first_external_frame_id == 101 &&
                states[2].presentation_count == 1,
                "%s: bad state after Present 1.\n", name);
    }
    else
    {
        check(states[1].cpu_finished == 2 && states[1].gpu_finished == 1 &&
                states[1].submit_pending && states[1].first_external_frame_id == 101 &&
                states[1].presentation_count == 1,
                "%s: bad state after Present 1.\n", name);
        if (events[1] == event_type::gpu_completion)
            check(states[2].cpu_finished == 2 && states[2].gpu_finished == 2 &&
                    !states[2].submit_pending && states[2].presentation_count == 1,
                    "%s: bad state after GPU completion.\n", name);
        else
            check(states[2].cpu_finished == 2 && states[2].gpu_finished == 1 &&
                    states[2].submit_pending && states[2].second_external_frame_id == 0 &&
                    states[2].presentation_count == 2,
                    "%s: bad state after Present 2.\n", name);
    }
}

static void run_control_case()
{
    fixture f;
    vkd3d_test_framepacer_snapshot initial, after_present, after_completion;
    vkd3d_test_latency_sleep_decision decision = {};

    initial = f.snapshot();
    f.present();
    after_present = f.snapshot();
    f.complete_gpu();
    after_completion = f.snapshot();

    check_common_snapshot("control", initial);
    check_common_snapshot("control", after_present);
    check_common_snapshot("control", after_completion);
    check(after_present.cpu_finished == 2 && after_present.gpu_finished == 1 &&
            after_present.submit_pending && after_present.first_external_frame_id == 101,
            "control: bad state after Present.\n");
    check(after_completion.cpu_finished - initial.cpu_finished == 1 &&
            after_completion.gpu_finished - initial.gpu_finished == 1,
            "control: expected CPU/GPU deltas 1/1, got %" PRIu64 "/%" PRIu64 ".\n",
            after_completion.cpu_finished - initial.cpu_finished,
            after_completion.gpu_finished - initial.gpu_finished);
    check(!after_completion.submit_pending &&
            after_completion.vulkan_gpu_timestamp == fixed_gpu_timestamp,
            "control: completion did not consume the pending Present.\n");
    check(after_completion.first_external_frame_id == 101 &&
            after_completion.second_external_frame_id == 0,
            "control: unexpected internal/external mappings.\n");

    pacer_test_reset_latency_sleep_decision(f.device);
    NvAPI_sleep(f.device);
    pacer_test_get_latency_sleep_decision(f.device, &decision);
    check(decision.call_count == 1 && decision.frame_id == 3 && decision.wait_id == 2 &&
            decision.cpu_finished == 2 && decision.gpu_finished == 2 &&
            decision.wait_satisfied && decision.would_start_frame,
            "control: unexpected next sleep decision.\n");

    std::printf("control: CPU +1, GPU +1, next wait %" PRIu64 " ready.\n",
            decision.wait_id);
}

static void run_multiple_submit_case()
{
    fixture f;
    fixture::submit_pair second = f.submit();
    vkd3d_test_framepacer_snapshot after_first, after_second;

    check(second.command == 2 && second.vulkan == 2,
            "multiple-submit: unexpected second generations %" PRIu64 "/%" PRIu64 ".\n",
            second.command, second.vulkan);
    f.present();
    f.complete_gpu();
    after_first = f.snapshot();
    check(after_first.cpu_finished == 2 && after_first.gpu_finished == 1 &&
            !after_first.submit_pending,
            "multiple-submit: first completion advanced the simulation early.\n");

    f.complete_gpu(second.command, second.vulkan);
    after_second = f.snapshot(second.command, second.vulkan);
    check(after_second.cpu_finished == 2 && after_second.gpu_finished == 2 &&
            !after_second.submit_pending && after_second.submit_simulation_id == 2,
            "multiple-submit: final completion did not advance exactly one simulation.\n");
    std::printf("multiple-submit: two completions produced one GPU watermark.\n");
}

static void run_out_of_order_simulation_case()
{
    static constexpr uint64_t second_external_id = UINT64_C(0xf000000000000065);
    fixture f;
    fixture::submit_pair second;
    vkd3d_test_framepacer_snapshot after_second, after_first;

    f.present();
    f.begin_simulation(second_external_id);
    second = f.submit();
    f.present(second_external_id);

    after_second = f.snapshot(second.command, second.vulkan);
    check(after_second.cpu_finished == 3 && after_second.gpu_finished == 1 &&
            after_second.first_external_frame_id == 101 &&
            after_second.second_external_frame_id == second_external_id,
            "out-of-order: CPU simulations were not sealed contiguously.\n");

    f.complete_gpu(second.command, second.vulkan);
    after_second = f.snapshot(second.command, second.vulkan);
    check(after_second.gpu_finished == 1 && !after_second.submit_pending,
            "out-of-order: simulation 3 bypassed incomplete simulation 2.\n");

    f.complete_gpu();
    after_first = f.snapshot();
    check(after_first.gpu_finished == 3 && !after_first.submit_pending,
            "out-of-order: contiguous GPU watermark did not catch up to 3.\n");
    std::printf("out-of-order: GPU watermark held at 1, then advanced contiguously to 3.\n");
}

static void run_submit_generation_cases()
{
    static constexpr uint64_t ring_size = 2048;

    {
        fixture f;
        uint64_t id = 0;

        for (uint64_t expected = 2; expected <= ring_size; expected++)
        {
            id = pacer_command_queue_notify_submit(f.queues.command_queue);
            check(id == expected,
                    "submit-generation: expected command generation %" PRIu64
                    ", got %" PRIu64 ".\n", expected, id);
        }
        id = pacer_command_queue_notify_submit(f.queues.command_queue);
        check(id == 0, "submit-generation: unaccounted ring slot was reused.\n");

        vkd3d_test_framepacer_snapshot state = f.snapshot();
        check(state.submit_pending && state.tracking_bypassed &&
                state.cpu_finished == 1 && state.gpu_finished == 1,
                "submit-generation: tracking failure fabricated progress or did not bypass.\n");
    }

    {
        fixture f;
        uint64_t id = 0;

        f.complete_gpu();
        for (uint64_t expected = 2; expected <= ring_size + 1; expected++)
        {
            id = pacer_command_queue_notify_submit(f.queues.command_queue);
            check(id == expected,
                    "submit-generation: safely accounted slot did not accept generation %" PRIu64
                    " (got %" PRIu64 ").\n", expected, id);
        }
    }

    {
        fixture f;
        uint64_t id = 0;

        f.complete_gpu();
        for (uint64_t expected = 2; expected <= ring_size + 1; expected++)
        {
            id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
            check(id == expected,
                    "timestamp-generation: expected Vulkan generation %" PRIu64
                    ", got %" PRIu64 ".\n", expected, id);
        }
        id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        check(id == 0, "timestamp-generation: unaccounted timestamp slot was reused.\n");
    }

    std::printf("submit-generation: delayed generations cannot overwrite live records.\n");
}

static void run_mode_transition_case()
{
    static constexpr uint64_t reflex_id = 501;
    fixture f(false);
    vkd3d_test_framepacer_snapshot legacy_first, reflex, legacy_second;

    f.legacy_present();
    f.complete_gpu();
    legacy_first = f.snapshot();
    check(legacy_first.cpu_finished == 2 && legacy_first.gpu_finished == 2,
            "mode-transition: initial legacy frame did not finish at 2/2.\n");

    f.enable_reflex(reflex_id);
    fixture::submit_pair reflex_submit = f.submit();
    f.present(reflex_id);
    f.complete_gpu(reflex_submit.command, reflex_submit.vulkan);
    reflex = f.snapshot(reflex_submit.command, reflex_submit.vulkan, 3, 4);
    check(reflex.cpu_finished == 2 && reflex.gpu_finished == 2 &&
            reflex.submit_simulation_id == 2 &&
            f.snapshot(reflex_submit.command, reflex_submit.vulkan).first_external_frame_id == reflex_id,
            "mode-transition: Reflex generation did not use a fresh trusted domain.\n");

    f.disable_reflex();
    fixture::submit_pair final_submit = f.submit();
    f.legacy_present();
    f.complete_gpu(final_submit.command, final_submit.vulkan);
    legacy_second = f.snapshot(final_submit.command, final_submit.vulkan, 3, 4);
    check(legacy_second.cpu_finished == 2 && legacy_second.gpu_finished == 2 &&
            !legacy_second.reflex_accounting,
            "mode-transition: final legacy generation was not trustworthy at 2/2.\n");
    std::printf("mode-transition: legacy, Reflex, and legacy used isolated generations.\n");
}

static void run_delayed_epoch_completion_case()
{
    static constexpr uint64_t old_reflex_id = 601;
    static constexpr uint64_t new_reflex_id = 602;
    fixture f(false);
    vkd3d_test_framepacer_snapshot before_old_completion, after_old_completion;
    vkd3d_test_framepacer_snapshot before_new_epoch_completion, after_new_epoch_completion;

    f.legacy_present();
    f.complete_gpu();

    f.enable_reflex(old_reflex_id);
    fixture::submit_pair old_first = f.submit();
    fixture::submit_pair old_second = f.submit();
    f.present(old_reflex_id);
    f.disable_reflex();

    fixture::submit_pair legacy_submit = f.submit();
    f.legacy_present();
    f.complete_gpu(legacy_submit.command, legacy_submit.vulkan);
    before_old_completion = f.snapshot(legacy_submit.command, legacy_submit.vulkan);
    f.complete_gpu(old_first.command, old_first.vulkan);
    after_old_completion = f.snapshot(legacy_submit.command, legacy_submit.vulkan);
    check(before_old_completion.cpu_finished == 2 &&
            before_old_completion.gpu_finished == 2 &&
            after_old_completion.cpu_finished == 2 &&
            after_old_completion.gpu_finished == 2,
            "delayed-epoch: old Reflex completion changed the active legacy domain.\n");

    f.enable_reflex(new_reflex_id);
    fixture::submit_pair new_submit = f.submit();
    f.present(new_reflex_id);
    before_new_epoch_completion = f.snapshot(new_submit.command,
            new_submit.vulkan, 5, 6);
    f.complete_gpu(old_second.command, old_second.vulkan);
    after_old_completion = f.snapshot(new_submit.command,
            new_submit.vulkan, 5, 6);
    check(before_new_epoch_completion.cpu_finished == 2 &&
            before_new_epoch_completion.gpu_finished == 1 &&
            after_old_completion.cpu_finished == 2 &&
            after_old_completion.gpu_finished == 1,
            "delayed-epoch: old Reflex completion changed the newer Reflex domain.\n");

    f.complete_gpu(new_submit.command, new_submit.vulkan);
    after_new_epoch_completion = f.snapshot(new_submit.command,
            new_submit.vulkan, 5, 6);
    check(after_new_epoch_completion.cpu_finished == 2 &&
            after_new_epoch_completion.gpu_finished == 2 &&
            f.snapshot(new_submit.command, new_submit.vulkan).first_external_frame_id == new_reflex_id,
            "delayed-epoch: newer Reflex epoch did not publish its anchored watermark.\n");
    std::printf("delayed-epoch: stale callbacks were inert in legacy and newer Reflex epochs.\n");
}

static void run_cancelled_present_case(const char *name)
{
    fixture f;
    vkd3d_test_framepacer_snapshot state;

    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_START_NV);
    pacer_present_attempt_token attempt = pacer_begin_present_attempt(f.device);
    pacer_notify_aborted_present(f.device, &f.swapchain_token, attempt);
    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_END_NV);
    f.complete_gpu();
    state = f.snapshot();
    check(state.cpu_finished == 2 && state.gpu_finished == 1 &&
            !state.submit_pending && state.presentation_count == 0 &&
            state.tracking_bypassed,
            "%s: canceled attempt left a gap or fabricated GPU progress.\n", name);
    std::printf("%s: attempt sealed as canceled; GPU watermark held.\n", name);
}

static void run_inactive_present_case()
{
    fixture f;
    char inactive_swapchain = 0;
    vkd3d_test_framepacer_snapshot state;

    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_START_NV);
    pacer_present_attempt_token attempt = pacer_begin_present_attempt(f.device);
    pacer_notify_present(f.device, &inactive_swapchain, 1, attempt);
    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_END_NV);
    f.complete_gpu();
    state = f.snapshot();
    check(state.cpu_finished == 2 && state.gpu_finished == 1 &&
            !state.submit_pending && state.presentation_count == 0 &&
            state.tracking_bypassed,
            "inactive-present: skipped inactive swapchain left a gap or false progress.\n");
    std::printf("inactive-present: pacer early-return terminalized the attempt.\n");
}

static void run_present_end_without_notify_case()
{
    fixture f;
    vkd3d_test_framepacer_snapshot state;

    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_START_NV);
    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_END_NV);
    f.complete_gpu();
    state = f.snapshot();
    check(state.cpu_finished == 2 && state.gpu_finished == 1 &&
            state.tracking_bypassed && state.presentation_count == 0,
            "present-end-cancel: marker end did not terminalize a skipped Present.\n");
    std::printf("present-end-cancel: skipped attempt terminalized at PRESENT_END.\n");
}

static void run_partial_tracking_failure_case()
{
    fixture f;
    uint64_t unowned_submit = UINT64_MAX;
    vkd3d_test_framepacer_snapshot state;

    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);

    std::thread unowned([&]() {
        unowned_submit = pacer_command_queue_notify_submit(f.queues.command_queue);
    });
    unowned.join();
    check(unowned_submit == 0,
            "partial-tracking: unowned submit unexpectedly acquired a simulation.\n");

    f.present();
    f.complete_gpu();
    state = f.snapshot();
    check(state.cpu_finished == 2 && state.gpu_finished == 1 &&
            !state.submit_pending && state.tracking_bypassed,
            "partial-tracking: tracked subset fabricated trusted GPU progress.\n");
    std::printf("partial-tracking: incomplete simulation held the trusted GPU watermark.\n");
}

static void run_reflex_gap_to_legacy_case()
{
    static constexpr uint64_t reflex_id = 701;
    fixture f(false);
    vkd3d_test_framepacer_snapshot reflex_ahead, legacy_complete, after_stale;
    vkd3d_test_latency_sleep_decision decision = {};

    f.enable_reflex(reflex_id);
    fixture::submit_pair reflex_submit = f.submit();
    f.present(reflex_id);
    reflex_ahead = f.snapshot(reflex_submit.command, reflex_submit.vulkan);
    check(reflex_ahead.cpu_finished == 2 && reflex_ahead.gpu_finished == 1,
            "reflex-gap-legacy: Reflex CPU was not ahead of GPU at transition.\n");

    f.disable_reflex();
    fixture::submit_pair legacy_submit = f.submit();
    f.legacy_present();
    f.complete_gpu(legacy_submit.command, legacy_submit.vulkan);
    legacy_complete = f.snapshot(legacy_submit.command, legacy_submit.vulkan);
    check(legacy_complete.cpu_finished == 2 && legacy_complete.gpu_finished == 2 &&
            !legacy_complete.reflex_accounting,
            "reflex-gap-legacy: legacy did not establish an isolated 2/2 domain.\n");

    pacer_test_reset_latency_sleep_decision(f.device);
    pacer_test_sleep_for_frame(f.device, 3);
    pacer_test_get_latency_sleep_decision(f.device, &decision);
    check(decision.call_count == 1 && decision.wait_id == 2 &&
            decision.wait_satisfied && decision.would_start_frame &&
            decision.entry_accounting_state == decision.exit_accounting_state,
            "reflex-gap-legacy: legacy pacing was not satisfied by its own domain.\n");

    f.complete_gpu(reflex_submit.command, reflex_submit.vulkan);
    after_stale = f.snapshot(legacy_submit.command, legacy_submit.vulkan);
    check(after_stale.cpu_finished == 2 && after_stale.gpu_finished == 2,
            "reflex-gap-legacy: delayed Reflex completion changed legacy progress.\n");
    std::printf("reflex-gap-legacy: abandoned Reflex holes were isolated from legacy pacing.\n");
}

static void run_stale_present_after_disable_case()
{
    fixture f;
    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_START_NV);
    f.disable_reflex();
    pacer_present_attempt_token stale = pacer_begin_present_attempt(f.device);
    check(!!(stale.accounting_epoch & 1) && stale.simulation_id == 2 &&
            stale.attempt_generation,
            "stale-present-disable: marker-created Reflex token was not retained.\n");
    f.accept_present(stale);
    vkd3d_test_framepacer_snapshot after_stale = f.snapshot();
    check(after_stale.cpu_finished == 1 && after_stale.gpu_finished == 1 &&
            !after_stale.reflex_accounting,
            "stale-present-disable: stale Reflex callback advanced legacy.\n");

    fixture::submit_pair legacy_submit = f.submit();
    f.legacy_present();
    f.complete_gpu(legacy_submit.command, legacy_submit.vulkan);
    vkd3d_test_framepacer_snapshot legacy = f.snapshot(
            legacy_submit.command, legacy_submit.vulkan);
    check(legacy.cpu_finished == 2 && legacy.gpu_finished == 2,
            "stale-present-disable: subsequent legacy accounting was not trustworthy.\n");
    std::printf("stale-present-disable: terminal callback retained its old Reflex owner.\n");
}

static void run_stale_present_after_reenable_case(bool abort_stale)
{
    static constexpr uint64_t new_reflex_id = 702;
    fixture f;
    pacer_present_attempt_token stale = f.begin_present_attempt(101);

    f.disable_reflex();
    f.enable_reflex(new_reflex_id);
    fixture::submit_pair current_submit = f.submit();
    pacer_present_attempt_token current = f.begin_present_attempt(new_reflex_id);

    if (abort_stale)
        f.abort_present(stale);
    else
        f.accept_present(stale);

    vkd3d_test_framepacer_snapshot before_current = f.snapshot(
            current_submit.command, current_submit.vulkan);
    check(before_current.cpu_finished == 1 && before_current.gpu_finished == 1 &&
            before_current.submit_pending && !before_current.tracking_bypassed,
            "stale-present-reenable: stale %s mutated the newer Reflex epoch.\n",
            abort_stale ? "abort" : "accept");

    f.accept_present(current);
    NvAPI_setLatencyMarker(f.device, new_reflex_id,
            VK_LATENCY_MARKER_PRESENT_END_NV);
    f.complete_gpu(current_submit.command, current_submit.vulkan);
    vkd3d_test_framepacer_snapshot completed = f.snapshot(
            current_submit.command, current_submit.vulkan);
    check(completed.cpu_finished == 2 && completed.gpu_finished == 2 &&
            completed.presentation_count == 1 && !completed.tracking_bypassed,
            "stale-present-reenable: current attempt did not remain intact.\n");
    std::printf("stale-present-reenable: stale %s could not retire the current attempt.\n",
            abort_stale ? "abort" : "accept");
}

static void run_unidentified_terminal_case()
{
    fixture f;
    pacer_present_attempt_token current = f.begin_present_attempt(101);
    pacer_present_attempt_token unidentified = {};

    f.abort_present(unidentified);
    f.accept_present(current);
    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_PRESENT_END_NV);
    f.complete_gpu();
    vkd3d_test_framepacer_snapshot state = f.snapshot();
    check(state.cpu_finished == 2 && state.gpu_finished == 2 &&
            state.presentation_count == 1 && state.tracking_bypassed,
            "unidentified-terminal: unrelated current attempt was canceled.\n");
    std::printf("unidentified-terminal: pacing bypassed without cross-attempt cancellation.\n");
}

static void run_sealed_unknown_owner_case()
{
    fixture f;

    NvAPI_setLatencyMarker(f.device, 101, VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    uint64_t unowned = pacer_command_queue_notify_submit(f.queues.command_queue);
    check(unowned == 0,
            "sealed-unknown-owner: submit after RENDERSUBMIT_END found an owner.\n");
    f.present();
    f.complete_gpu();
    vkd3d_test_framepacer_snapshot state = f.snapshot();
    check(state.cpu_finished == 2 && state.gpu_finished == 1 &&
            state.tracking_bypassed,
            "sealed-unknown-owner: tracked subset published GPU progress.\n");
    std::printf("sealed-unknown-owner: sealed GPU-incomplete simulation was poisoned.\n");
}

static void run_multiple_open_unknown_owner_case()
{
    static constexpr uint64_t second_reflex_id = 703;
    fixture f;
    f.begin_simulation(second_reflex_id);
    uint64_t unowned = UINT64_MAX;

    std::thread unknown_owner([&]() {
        unowned = pacer_command_queue_notify_submit(f.queues.command_queue);
    });
    unknown_owner.join();
    check(unowned == 0,
            "multiple-open-unknown: unscoped submit unexpectedly found an owner.\n");

    f.present(101);
    f.present(second_reflex_id);
    f.complete_gpu();
    vkd3d_test_framepacer_snapshot first = f.snapshot();
    check(first.cpu_finished == 3 && first.gpu_finished == 1 &&
            first.tracking_bypassed,
            "multiple-open-unknown: not all plausible simulations were poisoned.\n");
    std::printf("multiple-open-unknown: all simulations beyond the GPU frontier were poisoned.\n");
}

static void run_generation_aware_sleep_case()
{
    fixture f(false);
    vkd3d_test_latency_sleep_decision decision = {};

    pacer_test_set_latency_sleep_bypass(f.device, false);
    pacer_test_reset_latency_sleep_decision(f.device);
    std::thread sleeper([&]() {
        pacer_test_sleep_for_frame(f.device, 4);
    });
    while (!pacer_test_get_sleep_entry_count(f.device))
        std::this_thread::yield();

    NvAPI_setSleepMode(f.device, true, 0);
    sleeper.join();
    pacer_test_get_latency_sleep_decision(f.device, &decision);
    check(decision.call_count == 1 &&
            decision.entry_accounting_state != decision.exit_accounting_state &&
            !decision.would_start_frame,
            "generation-sleep: newer timeline satisfied an old pacing decision.\n");
    std::printf("generation-sleep: mode transition woke and retired the old waiter.\n");
}

static void run_stale_nvapi_sleep_publication_case()
{
    fixture f;
    vkd3d_test_nvapi_adapter_state current = {}, after = {};
    vkd3d_test_prediction_state current_prediction = {}, after_prediction = {};

    pacer_test_set_latency_sleep_bypass(f.device, false);
    pacer_test_reset_latency_sleep_decision(f.device);
    pacer_test_pause_after_sleep(f.device, true);
    std::thread stale_sleep([&]() {
        NvAPI_sleep(f.device);
    });
    while (!pacer_test_get_sleep_entry_count(f.device))
        std::this_thread::yield();

    f.disable_reflex();
    NvAPI_setSleepMode(f.device, true, 0);
    while (!pacer_test_sleep_returned(f.device))
        std::this_thread::yield();
    NvAPI_setLatencyMarker(f.device, 811,
            VK_LATENCY_MARKER_SIMULATION_START_NV);
    pacer_test_set_prediction(f.device, 100, 51001);
    pacer_test_get_nvapi_adapter_state(f.device, &current);
    pacer_test_get_prediction_state(f.device, &current_prediction);

    pacer_test_pause_after_sleep(f.device, false);
    stale_sleep.join();
    pacer_test_get_nvapi_adapter_state(f.device, &after);
    pacer_test_get_prediction_state(f.device, &after_prediction);
    check(current.accounting_epoch == after.accounting_epoch &&
            current.simulation_id == after.simulation_id &&
            current.drift == after.drift &&
            current.pending_sleep == after.pending_sleep &&
            current.last_end_sleep == after.last_end_sleep &&
            current_prediction.accounting_state == after_prediction.accounting_state &&
            current_prediction.finished_frame_id == after_prediction.finished_frame_id &&
            current_prediction.predicted_gpu_time == after_prediction.predicted_gpu_time,
            "stale-adapter-sleep: generation A changed generation C adapter state.\n");
    std::printf("stale-adapter-sleep: blocked generation A could not publish into generation C.\n");
}

static void run_post_sleep_commit_toctou_case()
{
    fixture f;
    vkd3d_test_nvapi_adapter_state before = {}, after = {};

    pacer_test_pause_after_sleep(f.device, true);
    std::thread stale_commit([&]() {
        NvAPI_sleep(f.device);
    });
    while (!pacer_test_sleep_returned(f.device))
        std::this_thread::yield();

    f.disable_reflex();
    NvAPI_setSleepMode(f.device, true, 0);
    NvAPI_setLatencyMarker(f.device, 901,
            VK_LATENCY_MARKER_SIMULATION_START_NV);
    pacer_test_get_nvapi_adapter_state(f.device, &before);

    pacer_test_pause_after_sleep(f.device, false);
    stale_commit.join();
    pacer_test_get_nvapi_adapter_state(f.device, &after);
    check(before.accounting_epoch == after.accounting_epoch &&
            before.simulation_id == after.simulation_id &&
            before.drift == after.drift &&
            before.pending_sleep == after.pending_sleep &&
            before.last_end_sleep == after.last_end_sleep,
            "sleep-commit-toctou: generation A published after transition to generation C.\n");
    std::printf("sleep-commit-toctou: post-sleep generation A commit was rejected atomically.\n");
}

static void run_prediction_generation_isolation_case()
{
    static constexpr int32_t stale_prediction = 41001;
    static constexpr int32_t current_prediction = 73002;
    fixture f;
    vkd3d_test_prediction_state state = {};

    pacer_test_set_prediction(f.device, 100, stale_prediction);
    pacer_test_get_prediction_state(f.device, &state);
    check(state.finished_frame_id == 100 &&
            state.predicted_gpu_time == stale_prediction,
            "prediction-generation: Reflex A prediction was not established.\n");

    for (unsigned int i = 0; i < 4; i++)
    {
        f.disable_reflex();
        NvAPI_setSleepMode(f.device, true, 0);
        pacer_test_get_prediction_state(f.device, &state);
        check(state.finished_frame_id == 0 && state.predicted_gpu_time == 0,
                "prediction-generation: transition %u inherited Reflex A cache.\n", i);

        pacer_test_set_prediction(f.device, 16, current_prediction + i);
        pacer_test_get_prediction_state(f.device, &state);
        check(state.finished_frame_id == 16 &&
                state.predicted_gpu_time == current_prediction + (int32_t)i,
                "prediction-generation: Reflex C did not use its own prediction.\n");
    }

    std::printf("prediction-generation: prediction cache reset across repeated accounting transitions.\n");
}

static void run_mode_publication_race_case()
{
    fixture f(false);
    std::atomic<bool> done = { false };

    std::thread toggler([&]() {
        for (unsigned int i = 0; i < 1000; i++)
        {
            NvAPI_setSleepMode(f.device, true, 0);
            NvAPI_setSleepMode(f.device, false, 0);
        }
        done.store(true, std::memory_order_release);
    });

    do
    {
        vkd3d_test_framepacer_snapshot state = f.snapshot();
        check(state.reflex_accounting == !!(state.accounting_state & 1),
                "mode-publication: observed a mixed accounting/pacing mode.\n");
    } while (!done.load(std::memory_order_acquire));
    toggler.join();
    std::printf("mode-publication: readers observed one authoritative mode state.\n");
}

static void run_waitable_teardown_case()
{
    fixture *f = new fixture(false);
    delete f;
    check(pacer_test_get_last_teardown_worker_stopped(),
            "waitable-teardown: worker was live during accounting destruction.\n");
    std::printf("waitable-teardown: worker was explicitly joined before teardown.\n");
}

static void run_generation_tagged_frame_state_case()
{
    fixture f;
    vkd3d_test_frame_state state = {};

    pacer_test_set_frame_state(f.device, 2, 801, 101);
    pacer_test_get_frame_state(f.device, 2, &state);
    check(state.marker_present && state.marker_value == 101 &&
            state.external_frame_id == 801,
            "generation-state: Reflex generation A state was not established.\n");

    f.disable_reflex();
    pacer_test_get_frame_state(f.device, 2, &state);
    check(!state.marker_present && !state.external_frame_id,
            "generation-state: legacy generation B observed Reflex A frame 2.\n");

    NvAPI_setSleepMode(f.device, true, 0);
    pacer_test_get_frame_state(f.device, 2, &state);
    check(!state.marker_present && !state.external_frame_id,
            "generation-state: Reflex generation C inherited reused frame 2.\n");

    for (unsigned int i = 0; i < 6; i++)
    {
        pacer_test_set_frame_state(f.device, 2, 900 + i, 200 + i);
        NvAPI_setSleepMode(f.device, !(state.accounting_state & 1), 0);
        pacer_test_get_frame_state(f.device, 2, &state);
        check(!state.marker_present && !state.external_frame_id,
                "generation-state: transition %u inherited reused frame 2.\n", i);
    }

    std::printf("generation-state: reused marker and mapping IDs stayed generation-local.\n");
}

static void run_stale_waitable_task_case()
{
    fixture f(false);
    vkd3d_test_frame_state before2 = {}, before4 = {}, after2 = {}, after4 = {};
    vkd3d_test_waitable_task_stats stats = {};
    uint64_t generation = f.snapshot().accounting_state;

    pacer_test_set_frame_state(f.device, 2, 701, 11);
    pacer_test_queue_waitable_task(f.device, generation, 3);
    pacer_test_wait_waitable_task_dequeued(f.device);

    NvAPI_setSleepMode(f.device, true, 0);
    pacer_test_set_frame_state(f.device, 2, 702, 22);
    pacer_test_set_frame_state(f.device, 4, 704, 44);
    pacer_test_get_frame_state(f.device, 2, &before2);
    pacer_test_get_frame_state(f.device, 4, &before4);

    pacer_test_resume_waitable_task(f.device);
    pacer_test_wait_waitable_task_idle(f.device);
    pacer_test_get_waitable_task_stats(f.device, &stats);
    pacer_test_get_frame_state(f.device, 2, &after2);
    pacer_test_get_frame_state(f.device, 4, &after4);

    check(stats.rejected_tasks == 1 && !stats.marker_reads &&
            !stats.marker_writes,
            "stale-waitable: stale task touched generation-sensitive state.\n");
    check(before2.marker_present && before2.marker_value == after2.marker_value &&
            before2.external_frame_id == after2.external_frame_id &&
            before4.marker_present && before4.marker_value == after4.marker_value &&
            before4.external_frame_id == after4.external_frame_id,
            "stale-waitable: stale task changed newer-generation marker contents.\n");
    std::printf("stale-waitable: dequeued stale task was rejected before marker access.\n");
}

static void run_device_construction_raii_case()
{
    pacer_device_properties properties = {};
    pacer_device_vk_procs procs = {};

    properties.vk_device = make_handle<VkDevice>();
    properties.timestamp_period = 1.0f;
    properties.khrCalibratedTimestamps = true;
    procs.vkCreateQueryPool = fake_create_query_pool;
    procs.vkDestroyQueryPool = fake_destroy_query_pool;
    procs.vkResetQueryPool = fake_reset_query_pool;
    procs.vkCmdResetQueryPool = fake_cmd_reset_query_pool;
    procs.vkGetQueryPoolResults = fake_get_query_pool_results;
    procs.vkCreateCommandPool = fake_create_command_pool;
    procs.vkDestroyCommandPool = fake_destroy_command_pool;
    procs.vkAllocateCommandBuffers = fake_allocate_command_buffers;
    procs.vkFreeCommandBuffers = fake_free_command_buffers;
    procs.vkBeginCommandBuffer = fake_begin_command_buffer;
    procs.vkEndCommandBuffer = fake_end_command_buffer;
    procs.vkCmdWriteTimestamp2 = fake_cmd_write_timestamp2;
    procs.vkGetCalibratedTimestampsKHR = fake_get_calibrated_timestamps;
    procs.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR =
            fake_get_calibrateable_time_domains;

    check(pacer_test_device_construction_failure_cleanup(&properties, &procs),
            "device-raii: C bridge leaked an exception or failed to stop the FramePacer worker.\n");
    std::printf("device-raii: C bridge contained adapter construction failure and joined the worker.\n");
}

static vkd3d_test_capture_snapshot capture_snapshot(fixture& f,
        uint64_t generation = 0)
{
    vkd3d_test_capture_snapshot snapshot = {};
    pacer_test_get_capture_snapshot(f.device, generation, &snapshot);
    return snapshot;
}

static fixture::submit_pair commit_capture_lease(fixture& f,
        pacer_command_capture_lease *lease)
{
    fixture::submit_pair result = {};
    result.command = pacer_command_queue_commit_capture(
            f.queues.command_queue, lease);
    if (result.command)
    {
        result.vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        check(result.vulkan && pacer_command_queue_notify_vulkan_submit(
                f.queues.command_queue, result.command, result.vulkan),
                "capture lease: failed to map command submit %" PRIu64 ".\n",
                result.command);
    }
    return result;
}

static void run_capture_start_a_execute_b_end_a_case()
{
    fixture f(false, false);
    std::atomic<unsigned int> phase = {0};
    fixture::submit_pair submit = {};

    std::thread marker_thread([&]() {
        NvAPI_setSleepMode(f.device, true, 0);
        NvAPI_sleep(f.device);
        NvAPI_setLatencyMarker(f.device, 1001,
                VK_LATENCY_MARKER_SIMULATION_START_NV);
        NvAPI_setLatencyMarker(f.device, 1001,
                VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
        phase.store(1, std::memory_order_release);
        while (phase.load(std::memory_order_acquire) != 2)
            std::this_thread::yield();
        NvAPI_setLatencyMarker(f.device, 1001,
                VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    });

    while (phase.load(std::memory_order_acquire) != 1)
        std::this_thread::yield();
    std::thread execute_thread([&]() {
        pacer_command_capture_lease lease =
                pacer_command_queue_acquire_capture(f.queues.command_queue);
        check(lease.active && lease.acquire_result == PACER_CAPTURE_ACQUIRED &&
                lease.token.external_reflex_id == 1001,
                "capture-1: thread B did not acquire thread A's capture.\n");
        submit = commit_capture_lease(f, &lease);
    });
    execute_thread.join();
    phase.store(2, std::memory_order_release);
    marker_thread.join();

    vkd3d_test_capture_snapshot state = capture_snapshot(f);
    check(submit.command && state.simulation_id == 2 &&
            state.published_submits == 1 && state.in_flight_publications == 0 &&
            state.submission_seal_requested && state.submissions_sealed,
            "capture-1: START A / Execute B / END A did not close exact ownership.\n");
    std::printf("capture-1: START A, Execute B, END A.\n");
}

static void run_capture_close_then_publish_case()
{
    fixture f(true, false);
    pacer_command_capture_lease lease =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    vkd3d_test_capture_snapshot closing = capture_snapshot(f);
    check(lease.active && closing.in_flight_publications == 1 &&
            closing.end_association_count == 1 &&
            closing.submission_seal_requested && !closing.submissions_sealed,
            "capture-2: duplicate END or pre-close lease corrupted exact ownership.\n");

    fixture::submit_pair submit = commit_capture_lease(f, &lease);
    vkd3d_test_capture_snapshot sealed = capture_snapshot(f);
    check(submit.command && sealed.published_submits == 1 &&
            sealed.in_flight_publications == 0 && sealed.submissions_sealed &&
            sealed.end_association_count == 0 && !sealed.tracking_failed,
            "capture-2: pre-close lease did not publish after END.\n");
    std::printf("capture-2: lease acquired before END published after closure.\n");
}

static void run_capture_execute_after_end_case()
{
    fixture f(true, false);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    pacer_command_capture_lease lease =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    check(!lease.active && lease.acquire_result == PACER_CAPTURE_NO_OPEN &&
            capture_snapshot(f).active_lease_count == 0,
            "capture-3: Execute beginning after END attached to a capture.\n");
    std::printf("capture-3: Execute after END was rejected.\n");
}

static void run_capture_commit_and_abort_case()
{
    fixture f(true, false);
    pacer_command_capture_lease committed =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    pacer_command_capture_lease aborted =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    fixture::submit_pair submit = commit_capture_lease(f, &committed);
    pacer_command_queue_retire_capture(f.queues.command_queue, &aborted,
            PACER_CAPTURE_RETIRE_BENIGN_ABORT);
    vkd3d_test_capture_snapshot state = capture_snapshot(f);
    check(submit.command && !committed.active && !aborted.active &&
            state.published_submits == 1 && state.in_flight_publications == 0 &&
            state.active_lease_count == 0 && !state.tracking_failed,
            "capture-4: commit/abort C queue wiring did not retire both leases.\n");
    std::printf("capture-4: commit and benign abort used the C queue path.\n");
}

static void run_capture_ambiguous_case()
{
    fixture f(true, false);
    f.begin_simulation(1002);
    pacer_command_capture_lease lease = {};
    std::thread third([&]() {
        lease = pacer_command_queue_acquire_capture(f.queues.command_queue);
    });
    third.join();
    vkd3d_test_capture_snapshot state = capture_snapshot(f);
    check(!lease.active && lease.acquire_result == PACER_CAPTURE_AMBIGUOUS &&
            state.open_capture_count == 0 && state.active_lease_count == 0 &&
            state.tracking_failed,
            "capture-5: ambiguous third-thread Execute guessed an owner.\n");
    std::printf("capture-5: two open captures rejected ambiguous Execute.\n");
}

static void run_capture_abort_exactly_once_case()
{
    fixture f(true, false);
    pacer_command_capture_lease lease =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    pacer_command_queue_retire_capture(f.queues.command_queue, &lease,
            PACER_CAPTURE_RETIRE_BENIGN_ABORT);
    pacer_command_queue_retire_capture(f.queues.command_queue, &lease,
            PACER_CAPTURE_RETIRE_BENIGN_ABORT);
    vkd3d_test_capture_snapshot state = capture_snapshot(f);
    check(!lease.active && state.in_flight_publications == 0 &&
            state.active_lease_count == 0 && state.open_capture_count == 1 &&
            !state.tracking_failed,
            "capture-6: benign post-acquisition failure was not retired exactly once.\n");
    std::printf("capture-6: post-acquisition failure retired exactly once.\n");
}

static void run_capture_present_final_lease_case()
{
    fixture f(true, false);
    pacer_command_capture_lease lease =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    f.present();
    vkd3d_test_capture_snapshot at_present = capture_snapshot(f);
    vkd3d_test_framepacer_snapshot progress_at_present = f.snapshot(0, 0);
    check(at_present.cpu_sealed && at_present.submission_seal_requested &&
            at_present.in_flight_publications == 1 &&
            !at_present.submissions_sealed &&
            progress_at_present.gpu_finished == 1,
            "capture-7: Present crossed the in-flight publication blocker.\n");

    fixture::submit_pair submit = commit_capture_lease(f, &lease);
    vkd3d_test_capture_snapshot at_commit = capture_snapshot(f);
    vkd3d_test_framepacer_snapshot progress_at_commit =
            f.snapshot(submit.command, submit.vulkan);
    check(at_commit.published_submits == 1 &&
            at_commit.in_flight_publications == 0 &&
            at_commit.submissions_sealed && at_commit.completed_submits == 0 &&
            progress_at_commit.gpu_finished == 1,
            "capture-7: commit fabricated completion progress.\n");

    f.complete_gpu(submit.command, submit.vulkan);
    vkd3d_test_capture_snapshot completed = capture_snapshot(f);
    check(completed.completed_submits == 1 &&
            f.snapshot(submit.command, submit.vulkan).gpu_finished == 2,
            "capture-7: only real S1 completion failed to release GPU progress.\n");
    std::printf("capture-7: Present raced the final lease without advancing GPU early.\n");
}

static void run_capture_external_reuse_case()
{
    fixture f(true, false);
    fixture::submit_pair first = f.submit();
    f.present();
    uint64_t first_generation = capture_snapshot(f).capture_generation;
    f.begin_simulation(101);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    vkd3d_test_capture_snapshot state = capture_snapshot(f);
    check(first.command && state.capture_generation == first_generation &&
            state.open_capture_count == 0 && state.tracking_failed,
            "capture-8: reused external ID allowed delayed END to close a newer capture.\n");
    std::printf("capture-8: external-ID reuse was rejected by the high-watermark.\n");
}

static void run_capture_invalid_markers_case()
{
    fixture f(true, false);
    NvAPI_setLatencyMarker(f.device, 0,
            VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);
    NvAPI_setLatencyMarker(f.device, 9999,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    vkd3d_test_capture_snapshot capture = capture_snapshot(f);
    vkd3d_test_framepacer_snapshot state = f.snapshot(0, 0);
    check(capture.open_capture_count == 0 &&
            capture.active_lease_count == 0 && state.tracking_bypassed,
            "capture-9: zero/mismatched marker did not fail conservatively.\n");
    std::printf("capture-9: zero and mismatched START/END failed conservatively.\n");
}

static void run_capture_historical_thread_case()
{
    fixture f(true, false);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    pacer_command_capture_lease lease =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    check(!lease.active && lease.acquire_result == PACER_CAPTURE_NO_OPEN,
            "capture-10: historical marker thread retained render ownership.\n");
    std::printf("capture-10: historical thread owner could not attach after close.\n");
}

static void run_capture_external_metadata_identity_case()
{
    fixture f(true, false);
    pacer_command_capture_lease lease =
            pacer_command_queue_acquire_capture(f.queues.command_queue);
    uint64_t generation = lease.token.capture_generation;

    lease.token.external_reflex_id = 0xdeadbeef;
    fixture::submit_pair submit = commit_capture_lease(f, &lease);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    vkd3d_test_capture_snapshot state = capture_snapshot(f, generation);
    check(submit.command && f.snapshot(submit.command, submit.vulkan).submit_simulation_id == 2 &&
            state.simulation_id == 2 && state.published_submits == 1 &&
            !state.tracking_failed,
            "capture-11: copied external metadata became authoritative lease identity.\n");
    std::printf("capture-11: external metadata did not participate in lease ownership.\n");
}

static void run_capture_end_after_present_case()
{
    fixture f(true, false);
    fixture::submit_pair submit = f.submit();

    f.present();
    vkd3d_test_capture_snapshot before_end = capture_snapshot(f);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    NvAPI_setLatencyMarker(f.device, 101,
            VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
    vkd3d_test_capture_snapshot after_end = capture_snapshot(f);
    check(submit.command && before_end.submissions_sealed &&
            !before_end.tracking_failed && !after_end.tracking_failed &&
            !f.snapshot(submit.command, submit.vulkan).tracking_bypassed,
            "capture-12: exact END after Present poisoned trusted tracking.\n");
    std::printf("capture-12: exact END after Present was consumed idempotently.\n");
}

static void run_capture_long_session_case()
{
    static constexpr uint64_t first_external_id = 10000;
    static constexpr uint32_t capture_count = 100000;
    fixture f(false, false);
    vkd3d_test_capture_snapshot state = {};

    f.enable_reflex(first_external_id);

    for (uint32_t i = 0; i < capture_count; i++)
    {
        uint64_t external_id = first_external_id + i;
        fixture::submit_pair submit = f.submit();
        NvAPI_setLatencyMarker(f.device, external_id,
                VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
        f.present(external_id);
        if (submit.command)
            f.complete_gpu(submit.command, submit.vulkan);
        if (i + 1 < capture_count)
        {
            f.begin_simulation(external_id + 1);
            if (i == 128)
            {
                NvAPI_setLatencyMarker(f.device, first_external_id,
                        VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
                state = capture_snapshot(f);
                check(state.open_capture_count == 1 &&
                        !f.snapshot(0, 0).tracking_bypassed,
                        "capture-13: delayed old END affected a newer open capture.\n");
            }
        }
    }

    state = capture_snapshot(f);
    check(state.capture_record_count <= 64 &&
            state.end_association_count == 0 &&
            state.highest_started_external_id ==
                    first_external_id + capture_count - 1 &&
            !f.snapshot(0, 0).tracking_bypassed,
            "capture-13: 100k session was unbounded or bypassed (%u captures, %u associations).\n",
            state.capture_record_count, state.end_association_count);
    std::printf("capture-13: 100000 same-epoch captures stayed bounded and trusted.\n");
}

static void run_capture_marker_order_cases()
{
    {
        fixture f(true, false);
        f.begin_simulation(100);
        vkd3d_test_capture_snapshot state = capture_snapshot(f);
        check(state.open_capture_count == 0 &&
                state.highest_started_external_id == 101 &&
                f.snapshot(0, 0).tracking_bypassed,
                "capture-14: non-monotonic START was accepted.\n");
    }

    {
        fixture f(true, false);
        NvAPI_setLatencyMarker(f.device, 102,
                VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
        check(f.snapshot(0, 0).tracking_bypassed,
                "capture-14: future END did not fail conservatively.\n");
    }

    {
        fixture f(true, false);
        fixture::submit_pair submit = f.submit();
        NvAPI_setLatencyMarker(f.device, 101,
                VK_LATENCY_MARKER_RENDERSUBMIT_END_NV);
        f.present();
        if (submit.command)
            f.complete_gpu(submit.command, submit.vulkan);
        f.disable_reflex();
        f.enable_reflex(101);
        vkd3d_test_capture_snapshot reset = capture_snapshot(f);
        check(reset.open_capture_count == 1 &&
                reset.highest_started_external_id == 101 &&
                !f.snapshot(0, 0).tracking_bypassed,
                "capture-14: accounting epoch transition did not permit ID reset.\n");
    }

    std::printf("capture-14: marker ordering, future END, and epoch reset validated.\n");
}

struct command_ring_barrier
{
    uint32_t point = 0;
    uint32_t skip_matches = 0;
    std::mutex mutex;
    std::condition_variable cond;
    bool armed = false;
    bool arrived = false;
    bool release = false;
};

static void command_ring_hook(uint32_t point, void *userdata)
{
    command_ring_barrier *barrier = static_cast<command_ring_barrier *>(userdata);
    std::unique_lock<std::mutex> lock(barrier->mutex);

    if (point != barrier->point || !barrier->armed)
        return;
    if (barrier->skip_matches)
    {
        --barrier->skip_matches;
        return;
    }
    barrier->armed = false;
    barrier->arrived = true;
    barrier->cond.notify_all();
    barrier->cond.wait(lock, [&]() { return barrier->release; });
}

static void arm_command_ring_barrier(fixture& f, command_ring_barrier& barrier,
        uint32_t point)
{
    {
        std::lock_guard<std::mutex> lock(barrier.mutex);
        barrier.point = point;
        barrier.skip_matches = 0;
        barrier.arrived = false;
        barrier.release = false;
        barrier.armed = true;
    }
    pacer_test_set_command_ring_hook(f.queues.command_queue,
            command_ring_hook, &barrier);
}

static void wait_command_ring_barrier(command_ring_barrier& barrier)
{
    std::unique_lock<std::mutex> lock(barrier.mutex);
    barrier.cond.wait(lock, [&]() { return barrier.arrived; });
}

static void release_command_ring_barrier(fixture& f,
        command_ring_barrier& barrier)
{
    {
        std::lock_guard<std::mutex> lock(barrier.mutex);
        barrier.release = true;
        barrier.cond.notify_all();
    }
    pacer_test_set_command_ring_hook(f.queues.command_queue, nullptr, nullptr);
}

static void arm_vulkan_ring_barrier(fixture& f, command_ring_barrier& barrier,
        uint32_t point, uint32_t skip_matches = 0)
{
    {
        std::lock_guard<std::mutex> lock(barrier.mutex);
        barrier.point = point;
        barrier.skip_matches = skip_matches;
        barrier.arrived = false;
        barrier.release = false;
        barrier.armed = true;
    }
    pacer_test_set_vulkan_ring_hook(f.queues.vulkan_queue,
            command_ring_hook, &barrier);
}

static void release_vulkan_ring_barrier(fixture& f,
        command_ring_barrier& barrier)
{
    {
        std::lock_guard<std::mutex> lock(barrier.mutex);
        barrier.release = true;
        barrier.cond.notify_all();
    }
    pacer_test_set_vulkan_ring_hook(f.queues.vulkan_queue, nullptr, nullptr);
}

static uint64_t reclaim_command_slot(fixture& f, uint64_t old_command_id)
{
    static constexpr uint64_t ring_size = 2048;
    uint64_t id = 0;

    for (uint64_t expected = old_command_id + 1;
            expected <= old_command_id + ring_size; expected++)
    {
        id = pacer_command_queue_notify_legacy_submit(f.queues.command_queue);
        check(id == expected,
                "command-ring reclaim: expected generation %" PRIu64
                ", got %" PRIu64 ".\n", expected, id);
    }
    return id;
}

static void advance_vulkan_before_reclaim(fixture& f, uint64_t old_vulkan_id)
{
    static constexpr uint64_t ring_size = 2048;

    for (uint64_t expected = old_vulkan_id + 1;
            expected < old_vulkan_id + ring_size; expected++)
    {
        uint64_t id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        check(id == expected,
                "vulkan-ring advance: expected generation %" PRIu64
                ", got %" PRIu64 ".\n", expected, id);
    }
}

static uint64_t reclaim_vulkan_slot(fixture& f, uint64_t old_vulkan_id)
{
    advance_vulkan_before_reclaim(f, old_vulkan_id);
    uint64_t id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    check(id == old_vulkan_id + 2048,
            "vulkan-ring reclaim: expected generation %" PRIu64
            ", got %" PRIu64 ".\n", old_vulkan_id + 2048, id);
    return id;
}

static vkd3d_test_vulkan_slot_snapshot vulkan_slot_snapshot(fixture& f,
        uint64_t vulkan_id)
{
    vkd3d_test_vulkan_slot_snapshot snapshot = {};
    pacer_test_get_vulkan_slot_snapshot(f.queues.vulkan_queue,
            vulkan_id, &snapshot);
    return snapshot;
}

static vkd3d_test_command_slot_snapshot command_slot_snapshot(fixture& f,
        uint64_t command_id)
{
    vkd3d_test_command_slot_snapshot snapshot = {};
    pacer_test_get_command_slot_snapshot(f.queues.command_queue,
            command_id, &snapshot);
    return snapshot;
}

static void run_command_ring_legacy_publication_cases()
{
    {
        fixture f(false, false);
        command_ring_barrier barrier;
        uint64_t command_id = pacer_command_queue_notify_legacy_submit(
                f.queues.command_queue);
        uint64_t vulkan_id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        bool published = true;

        arm_command_ring_barrier(f, barrier,
                VKD3D_TEST_COMMAND_RING_VULKAN_BEFORE_SLOT);
        std::thread worker([&]() {
            published = pacer_command_queue_notify_vulkan_submit(
                    f.queues.command_queue, command_id, vulkan_id);
        });
        wait_command_ring_barrier(barrier);
        uint64_t reclaimed = reclaim_command_slot(f, command_id);
        release_command_ring_barrier(f, barrier);
        worker.join();

        vkd3d_test_command_slot_snapshot slot = command_slot_snapshot(f, command_id);
        check(!published && slot.generation == reclaimed && !slot.vulkan_id &&
                !slot.present_pending && !slot.present_epoch &&
                !slot.ledger_completion_accounted &&
                !f.snapshot(command_id, vulkan_id).submit_simulation_id,
                "command-ring-A: stale legacy publication crossed generation reclaim.\n");
    }

    {
        fixture f(false, false);
        uint64_t command_id = pacer_command_queue_notify_legacy_submit(
                f.queues.command_queue);
        uint64_t vulkan_id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
                command_id, vulkan_id),
                "command-ring-B: worker did not win before reclaim.\n");
        f.legacy_present();
        uint64_t reclaimed = reclaim_command_slot(f, command_id);
        vkd3d_test_command_slot_snapshot slot = command_slot_snapshot(f, command_id);
        check(slot.generation == reclaimed && !slot.vulkan_id &&
                !slot.present_pending && !slot.present_epoch &&
                slot.has_submit_timestamp,
                "command-ring-B: producer did not coherently reset reclaimed metadata.\n");
    }

    std::printf("command-ring-A/B: stale legacy publication lost; winning metadata reset on reclaim.\n");
}

static void run_command_ring_captured_rollback_cases()
{
    {
        fixture f(true, false);
        pacer_command_capture_lease lease =
                pacer_command_queue_acquire_capture(f.queues.command_queue);
        uint64_t command_id = pacer_command_queue_commit_capture(
                f.queues.command_queue, &lease);
        uint64_t vulkan_id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        command_ring_barrier barrier;
        bool published = true;

        arm_command_ring_barrier(f, barrier,
                VKD3D_TEST_COMMAND_RING_VULKAN_BEFORE_SLOT);
        std::thread worker([&]() {
            published = pacer_command_queue_notify_vulkan_submit(
                    f.queues.command_queue, command_id, vulkan_id);
        });
        wait_command_ring_barrier(barrier);
        uint64_t reclaimed = reclaim_command_slot(f, command_id);
        release_command_ring_barrier(f, barrier);
        worker.join();

        vkd3d_test_command_slot_snapshot abandoned =
                command_slot_snapshot(f, command_id);
        check(!published && abandoned.generation == reclaimed &&
                !abandoned.vulkan_id && abandoned.ledger_completion_accounted &&
                !f.snapshot(command_id, vulkan_id).submit_pending,
                "command-ring-C: ring-race loss did not abandon the exact captured record.\n");
        pacer_queue_notify_submit_failed(f.queues, command_id, vulkan_id);

        for (uint64_t i = 0; i < 2047; i++)
            pacer_command_queue_notify_legacy_submit(f.queues.command_queue);
        pacer_command_capture_lease reuse_lease =
                pacer_command_queue_acquire_capture(f.queues.command_queue);
        uint64_t reused = pacer_command_queue_commit_capture(
                f.queues.command_queue, &reuse_lease);
        check(reused == command_id + 4096,
                "command-ring-C/I: abandoned SubmitRecord was not reusable (%" PRIu64 ").\n",
                reused);
        if (reused)
            pacer_queue_notify_submit_failed(f.queues, reused, 0);
    }

    {
        fixture f(true, false);
        fixture::submit_pair submit = f.submit();
        pacer_queue_notify_submit_failed(f.queues, submit.command, submit.vulkan);
        vkd3d_test_capture_snapshot once = capture_snapshot(f);
        pacer_queue_notify_submit_failed(f.queues, submit.command, submit.vulkan);
        vkd3d_test_capture_snapshot twice = capture_snapshot(f);
        check(once.accounted_submits == 1 && twice.accounted_submits == 1 &&
                command_slot_snapshot(f, submit.command).ledger_completion_accounted,
                "command-ring-D: duplicate exact cleanup changed accounting twice.\n");
    }

    {
        fixture f(true, false);
        pacer_command_capture_lease lease =
                pacer_command_queue_acquire_capture(f.queues.command_queue);
        uint64_t command_id = pacer_command_queue_commit_capture(
                f.queues.command_queue, &lease);
        uint64_t reclaimed = reclaim_command_slot(f, command_id);
        f.disable_reflex();
        uint64_t vulkan_id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        bool published = pacer_command_queue_notify_vulkan_submit(
                f.queues.command_queue, command_id, vulkan_id);
        vkd3d_test_command_slot_snapshot slot = command_slot_snapshot(f, command_id);
        check(!published && slot.generation == reclaimed && !slot.vulkan_id &&
                slot.ledger_completion_accounted,
                "command-ring-E: terminal ledger publication failure remained pending.\n");
        pacer_queue_notify_submit_failed(f.queues, command_id, vulkan_id);
    }

    std::printf("command-ring-C/D/E/I: captured loss, duplicate cleanup, epoch failure, and reuse are exact.\n");
}

static void run_post_command_reclaim_failure_case()
{
    static constexpr uint64_t newer_present_epoch = 0x7a11ce55d15c71c7ull;
    static constexpr uint64_t newer_submit_timestamp = 0x5a17c0de1234ull;
    fixture f(true, false);
    fixture::submit_pair old_submit = f.submit();
    vkd3d_test_capture_snapshot committed = capture_snapshot(f);
    uint64_t reclaimed = reclaim_command_slot(f, old_submit.command);
    uint64_t new_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);

    check(old_submit.command && old_submit.vulkan &&
            committed.published_submits == 1 && !committed.accounted_submits,
            "post-reclaim-failure: captured submit was not committed and published.\n");
    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            reclaimed, new_vulkan),
            "post-reclaim-failure: failed to populate the reclaimed command slot.\n");
    check(pacer_test_set_command_submit_timestamp(f.queues.command_queue,
            reclaimed, newer_submit_timestamp),
            "post-reclaim-failure: failed to set the known newer submit timestamp.\n");
    check(pacer_test_set_command_present_state(f.queues.command_queue,
            reclaimed, newer_present_epoch),
            "post-reclaim-failure: failed to arm the newer Present state.\n");
    vkd3d_test_command_slot_snapshot newer_before =
            command_slot_snapshot(f, old_submit.command);

    pacer_queue_notify_submit_failed(f.queues,
            old_submit.command, old_submit.vulkan);
    vkd3d_test_capture_snapshot once = capture_snapshot(f);
    vkd3d_test_command_slot_snapshot newer_once =
            command_slot_snapshot(f, old_submit.command);
    vkd3d_test_vulkan_slot_snapshot new_vulkan_once =
            vulkan_slot_snapshot(f, new_vulkan);
    pacer_queue_notify_submit_failed(f.queues,
            old_submit.command, old_submit.vulkan);
    vkd3d_test_capture_snapshot twice = capture_snapshot(f);
    vkd3d_test_command_slot_snapshot newer_twice =
            command_slot_snapshot(f, old_submit.command);

    check(newer_before.generation == reclaimed &&
            newer_before.vulkan_id == new_vulkan &&
            newer_before.present_pending &&
            newer_before.present_epoch == newer_present_epoch &&
            newer_before.has_submit_timestamp &&
            newer_before.submit_timestamp == newer_submit_timestamp &&
            newer_once.generation == newer_before.generation &&
            newer_once.vulkan_id == newer_before.vulkan_id &&
            newer_once.present_pending == newer_before.present_pending &&
            newer_once.present_epoch == newer_before.present_epoch &&
            newer_once.has_submit_timestamp == newer_before.has_submit_timestamp &&
            newer_once.submit_timestamp == newer_submit_timestamp &&
            newer_twice.generation == newer_before.generation &&
            newer_twice.vulkan_id == newer_before.vulkan_id &&
            newer_twice.present_pending == newer_before.present_pending &&
            newer_twice.present_epoch == newer_before.present_epoch &&
            newer_twice.has_submit_timestamp == newer_before.has_submit_timestamp &&
            newer_twice.submit_timestamp == newer_submit_timestamp,
            "post-reclaim-failure: stale failure mutated newer command-slot state.\n");
    check(once.accounted_submits == 1 && twice.accounted_submits == 1 &&
            !f.snapshot(old_submit.command, old_submit.vulkan).submit_pending &&
            newer_once.ledger_completion_accounted,
            "post-reclaim-failure: exact ledger abandon was not idempotent after reclaim.\n");
    check(new_vulkan_once.found && !new_vulkan_once.submit_accounted,
            "post-reclaim-failure: stale failure accounted the newer Vulkan identity.\n");
    std::printf("post-reclaim-failure: exact captured abandon survived command-slot reclaim and duplicate cleanup.\n");
}

static void run_vulkan_ring_accounting_cases()
{
    {
        fixture f(false, false);
        uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(
                f.queues.vulkan_queue);
        command_ring_barrier barrier;

        pacer_queue_notify_submit_failed(f.queues, 0, old_vulkan);
        advance_vulkan_before_reclaim(f, old_vulkan);
        arm_vulkan_ring_barrier(f, barrier,
                VKD3D_TEST_VULKAN_RING_FINISH_BEFORE_SLOT);
        std::thread stale_cleanup([&]() {
            pacer_queue_notify_submit_failed(f.queues, 0, old_vulkan);
        });
        wait_command_ring_barrier(barrier);
        uint64_t reclaimed = pacer_vulkan_queue_notify_submit(
                f.queues.vulkan_queue);
        release_vulkan_ring_barrier(f, barrier);
        stale_cleanup.join();
        vkd3d_test_vulkan_slot_snapshot slot =
                vulkan_slot_snapshot(f, reclaimed);

        check(reclaimed == old_vulkan + 2048 && slot.found &&
                slot.generation == reclaimed && slot.has_submit_timestamp &&
                !slot.gpu_execution_start && !slot.gpu_execution_end &&
                !slot.submit_accounted,
                "vulkan-ring-A: stale cleanup mutated the reclaimed generation.\n");
    }

    {
        fixture f(false, false);
        uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(
                f.queues.vulkan_queue);
        command_ring_barrier barrier;
        uint64_t reclaimed = 0;

        pacer_test_set_vulkan_gpu_execution_start(f.queues.vulkan_queue,
                old_vulkan, fixed_gpu_timestamp + 1);
        advance_vulkan_before_reclaim(f, old_vulkan);
        arm_vulkan_ring_barrier(f, barrier,
                VKD3D_TEST_VULKAN_RING_PRODUCER_BEFORE_SLOT);
        std::thread producer([&]() {
            reclaimed = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        });
        wait_command_ring_barrier(barrier);
        f.complete_gpu(0, old_vulkan);
        vkd3d_test_vulkan_slot_snapshot old_slot =
                vulkan_slot_snapshot(f, old_vulkan);
        release_vulkan_ring_barrier(f, barrier);
        producer.join();
        vkd3d_test_vulkan_slot_snapshot new_slot =
                vulkan_slot_snapshot(f, reclaimed);

        check(old_slot.found && old_slot.has_submit_timestamp &&
                old_slot.gpu_execution_start == fixed_gpu_timestamp + 1 &&
                old_slot.gpu_execution_end == fixed_gpu_timestamp &&
                old_slot.submit_accounted,
                "vulkan-ring-B: cleanup did not win coherently before reclaim.\n");
        check(reclaimed == old_vulkan + 2048 && new_slot.found &&
                new_slot.generation == reclaimed && new_slot.has_submit_timestamp &&
                !new_slot.gpu_execution_start && !new_slot.gpu_execution_end &&
                !new_slot.submit_accounted,
                "vulkan-ring-B: reclaim did not reset all generation-bound state.\n");
    }

    std::printf("vulkan-ring-A/B: stale cleanup lost to reclaim; prior cleanup was coherently reset.\n");
}

static constexpr uint64_t historical_start = fixed_gpu_timestamp + 0x101;
static constexpr uint64_t historical_end = fixed_gpu_timestamp + 0x202;
static constexpr uint64_t reclaimed_start = fixed_gpu_timestamp + 0x10001;
static constexpr uint64_t reclaimed_end = fixed_gpu_timestamp + 0x20002;
static constexpr uint64_t historical_submit = fixed_gpu_timestamp + 0x303;
static constexpr uint64_t reclaimed_submit = fixed_gpu_timestamp + 0x30003;

static void set_vulkan_timestamps(fixture& f, uint64_t vulkan_id,
        uint64_t submit, uint64_t start, uint64_t end)
{
    pacer_test_set_vulkan_submit_timestamp(f.queues.vulkan_queue,
            vulkan_id, submit);
    pacer_test_set_vulkan_gpu_execution_start(f.queues.vulkan_queue,
            vulkan_id, start);
    pacer_test_set_vulkan_gpu_execution_end(f.queues.vulkan_queue,
            vulkan_id, end);
}

static void run_vulkan_gpu_start_getter_case()
{
    fixture f(false, false);
    uint64_t command_id = pacer_command_queue_notify_legacy_submit(
            f.queues.command_queue);
    uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    command_ring_barrier barrier;
    vkd3d_test_submit_iterator_snapshot result = {};

    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            command_id, old_vulkan),
            "vulkan-get-start: failed to publish the historical identity.\n");
    set_vulkan_timestamps(f, old_vulkan, historical_submit,
            historical_start, historical_end);
    pacer_queue_notify_submit_failed(f.queues, 0, old_vulkan);
    advance_vulkan_before_reclaim(f, old_vulkan);
    arm_vulkan_ring_barrier(f, barrier,
            VKD3D_TEST_VULKAN_RING_GETTER_BEFORE_SLOT);
    std::thread reader([&]() {
        pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
                command_id, &result);
    });
    wait_command_ring_barrier(barrier);
    uint64_t reclaimed = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    set_vulkan_timestamps(f, reclaimed, reclaimed_submit,
            reclaimed_start, reclaimed_end);
    release_vulkan_ring_barrier(f, barrier);
    reader.join();
    vkd3d_test_vulkan_slot_snapshot current = vulkan_slot_snapshot(f, reclaimed);

    check(reclaimed == old_vulkan + 2048 &&
            current.found && current.gpu_execution_start == reclaimed_start &&
            current.gpu_execution_end == reclaimed_end &&
            result.vulkan_id == old_vulkan && result.has_app_submit &&
            !result.has_vulkan_submit &&
            (result.vulkan_gpu_execution_start == 0 ||
                    result.vulkan_gpu_execution_start == historical_start) &&
            result.vulkan_gpu_execution_start != reclaimed_start &&
            (result.vulkan_gpu_execution_end == 0 ||
                    result.vulkan_gpu_execution_end == historical_end) &&
            result.vulkan_gpu_execution_end != reclaimed_end,
            "vulkan-get-start: GPU start getter returned reclaimed-generation data.\n");
    std::printf("vulkan-get-start: historical GPU start resolved to N or no data, never M.\n");
}

static void run_vulkan_gpu_end_getter_case()
{
    fixture f(false, false);
    uint64_t command_id = pacer_command_queue_notify_legacy_submit(
            f.queues.command_queue);
    uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    command_ring_barrier barrier;
    vkd3d_test_submit_iterator_snapshot result = {};

    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            command_id, old_vulkan),
            "vulkan-get-end: failed to publish the historical identity.\n");
    set_vulkan_timestamps(f, old_vulkan, historical_submit,
            historical_start, historical_end);
    pacer_queue_notify_submit_failed(f.queues, 0, old_vulkan);
    advance_vulkan_before_reclaim(f, old_vulkan);
    arm_vulkan_ring_barrier(f, barrier,
            VKD3D_TEST_VULKAN_RING_GETTER_BEFORE_SLOT, 1);
    std::thread reader([&]() {
        pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
                command_id, &result);
    });
    wait_command_ring_barrier(barrier);
    uint64_t reclaimed = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    set_vulkan_timestamps(f, reclaimed, reclaimed_submit,
            reclaimed_start, reclaimed_end);
    release_vulkan_ring_barrier(f, barrier);
    reader.join();
    vkd3d_test_vulkan_slot_snapshot current = vulkan_slot_snapshot(f, reclaimed);

    check(reclaimed == old_vulkan + 2048 &&
            current.found && current.gpu_execution_start == reclaimed_start &&
            current.gpu_execution_end == reclaimed_end &&
            result.vulkan_id == old_vulkan && result.has_app_submit &&
            !result.has_vulkan_submit &&
            result.vulkan_gpu_execution_start == historical_start &&
            (result.vulkan_gpu_execution_end == 0 ||
                    result.vulkan_gpu_execution_end == historical_end) &&
            result.vulkan_gpu_execution_end != reclaimed_end,
            "vulkan-get-end: GPU end getter returned reclaimed-generation data.\n");
    std::printf("vulkan-get-end: historical GPU end resolved to N or no data, never M.\n");
}

static void run_vulkan_repeated_getter_case()
{
    fixture f(false, false);
    uint64_t command_id = pacer_command_queue_notify_legacy_submit(
            f.queues.command_queue);
    uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    command_ring_barrier barrier;
    vkd3d_test_submit_iterator_snapshot before = {}, during = {};

    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            command_id, old_vulkan),
            "vulkan-get-repeat: failed to publish the historical identity.\n");
    set_vulkan_timestamps(f, old_vulkan, historical_submit,
            historical_start, historical_end);
    pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
            command_id, &before);
    pacer_queue_notify_submit_failed(f.queues, 0, old_vulkan);
    advance_vulkan_before_reclaim(f, old_vulkan);
    arm_vulkan_ring_barrier(f, barrier,
            VKD3D_TEST_VULKAN_RING_GETTER_BEFORE_SLOT);
    std::thread reader([&]() {
        pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
                command_id, &during);
    });
    wait_command_ring_barrier(barrier);
    uint64_t reclaimed = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    set_vulkan_timestamps(f, reclaimed, reclaimed_submit,
            reclaimed_start, reclaimed_end);
    release_vulkan_ring_barrier(f, barrier);
    reader.join();
    vkd3d_test_vulkan_slot_snapshot current = vulkan_slot_snapshot(f, reclaimed);

    check(before.vulkan_id == old_vulkan && before.has_app_submit &&
            before.has_vulkan_submit &&
            before.vulkan_gpu_execution_start == historical_start &&
            before.vulkan_gpu_execution_end == historical_end,
            "vulkan-get-repeat: pre-reclaim getter sequence was not coherent N data.\n");
    check(reclaimed == old_vulkan + 2048 &&
            current.found && current.gpu_execution_start == reclaimed_start &&
            current.gpu_execution_end == reclaimed_end &&
            during.vulkan_id == old_vulkan &&
            during.has_app_submit && !during.has_vulkan_submit &&
            !during.vulkan_gpu_execution_start &&
            !during.vulkan_gpu_execution_end,
            "vulkan-get-repeat: in-flight getter sequence synthesized mixed-generation data.\n");
    for (unsigned int i = 0; i < 3; i++)
    {
        vkd3d_test_submit_iterator_snapshot after = {};
        pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
                command_id, &after);
        check(after.vulkan_id == old_vulkan && after.has_app_submit &&
                !after.has_vulkan_submit &&
                !after.vulkan_gpu_execution_start &&
                !after.vulkan_gpu_execution_end,
                "vulkan-get-repeat: post-reclaim getter %u returned M under N.\n", i);
    }
    std::printf("vulkan-get-repeat: low-latency-style reads transitioned coherently from N to no data.\n");
}

static void run_vulkan_submit_getter_case()
{
    fixture f(false, false);
    uint64_t command_id = pacer_command_queue_notify_legacy_submit(
            f.queues.command_queue);
    uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    command_ring_barrier barrier;
    vkd3d_test_submit_iterator_snapshot result = {};

    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            command_id, old_vulkan),
            "vulkan-get-submit: failed to publish the historical identity.\n");
    set_vulkan_timestamps(f, old_vulkan, historical_submit,
            historical_start, historical_end);
    pacer_queue_notify_submit_failed(f.queues, 0, old_vulkan);
    advance_vulkan_before_reclaim(f, old_vulkan);
    /* The iterator reads start, end, then submit. Pause exactly the submit
     * getter so reclaim is forced between identity capture and its slot lock. */
    arm_vulkan_ring_barrier(f, barrier,
            VKD3D_TEST_VULKAN_RING_GETTER_BEFORE_SLOT, 2);
    std::thread reader([&]() {
        pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
                command_id, &result);
    });
    wait_command_ring_barrier(barrier);
    uint64_t reclaimed = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    set_vulkan_timestamps(f, reclaimed, reclaimed_submit,
            reclaimed_start, reclaimed_end);
    release_vulkan_ring_barrier(f, barrier);
    reader.join();
    vkd3d_test_vulkan_slot_snapshot current = vulkan_slot_snapshot(f, reclaimed);

    check(reclaimed == old_vulkan + 2048 && current.found &&
            current.submit_timestamp == reclaimed_submit &&
            result.vulkan_id == old_vulkan && result.has_app_submit &&
            ((result.vulkan_submit_timestamp == 0 && !result.has_vulkan_submit) ||
                    (result.vulkan_submit_timestamp == historical_submit &&
                            result.has_vulkan_submit)) &&
            result.vulkan_submit_timestamp != reclaimed_submit,
            "vulkan-get-submit: submit getter returned M timestamp under N identity.\n");
    std::printf("vulkan-get-submit: historical submit timestamp resolved to N or no data, never M.\n");
}

struct hook_lifetime_observer
{
    std::atomic<uint32_t> callbacks = {0};
    std::atomic<uint32_t> exits = {0};
};

struct hook_lifetime_state
{
    static constexpr uint64_t expected_magic = 0x91f00d5afeull;
    uint64_t magic = expected_magic;
    hook_lifetime_observer *observer;
    std::mutex mutex;
    std::condition_variable cond;
    bool entered = false;
    bool release = false;
    bool exited = false;
};

static void hook_lifetime_callback(uint32_t, void *userdata)
{
    hook_lifetime_state *state = static_cast<hook_lifetime_state *>(userdata);
    std::unique_lock<std::mutex> lock(state->mutex);

    check(state->magic == hook_lifetime_state::expected_magic,
            "hook-disarm: callback entered with invalid userdata.\n");
    state->observer->callbacks.fetch_add(1, std::memory_order_release);
    state->entered = true;
    state->cond.notify_all();
    state->cond.wait(lock, [&]() { return state->release; });
    check(state->magic == hook_lifetime_state::expected_magic,
            "hook-disarm: userdata lifetime ended before callback exit.\n");
    state->observer->exits.fetch_add(1, std::memory_order_release);
    state->exited = true;
    state->cond.notify_all();
}

static void run_hook_disarm_lifetime_case()
{
    fixture f(false, false);
    hook_lifetime_observer observer;
    auto state = std::make_unique<hook_lifetime_state>();
    std::mutex disarm_mutex;
    std::condition_variable disarm_cond;
    std::atomic<bool> disarm_returned = {false};
    bool disarm_done = false;

    state->observer = &observer;

    pacer_test_set_command_ring_hook(f.queues.command_queue,
            hook_lifetime_callback, state.get());
    std::thread producer([&]() {
        pacer_command_queue_notify_legacy_submit(f.queues.command_queue);
    });
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cond.wait(lock, [&]() { return state->entered; });
    }
    std::thread disarm([&]() {
        pacer_test_set_command_ring_hook(f.queues.command_queue,
                nullptr, nullptr);
        disarm_returned.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(disarm_mutex);
            disarm_done = true;
        }
        disarm_cond.notify_all();
    });
    pacer_test_wait_command_ring_hook_drain(f.queues.command_queue);
    check(!disarm_returned.load(std::memory_order_acquire),
            "hook-disarm: disarm returned while the callback was paused.\n");
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        check(!state->exited,
                "hook-disarm: callback exited before its explicit release.\n");
        state->release = true;
        state->cond.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cond.wait(lock, [&]() { return state->exited; });
    }
    {
        std::unique_lock<std::mutex> lock(disarm_mutex);
        disarm_cond.wait(lock, [&]() { return disarm_done; });
    }

    check(disarm_returned.load(std::memory_order_acquire) &&
            observer.callbacks.load(std::memory_order_acquire) == 1 &&
            observer.exits.load(std::memory_order_acquire) == 1,
            "hook-disarm: disarm did not return strictly after callback exit.\n");
    state.reset();
    pacer_command_queue_notify_legacy_submit(f.queues.command_queue);
    producer.join();
    disarm.join();
    check(observer.callbacks.load(std::memory_order_acquire) == 1 &&
            observer.exits.load(std::memory_order_acquire) == 1,
            "hook-disarm: a callback touched userdata after disarm returned.\n");
    std::printf("hook-disarm: disarm drained a forced in-flight callback before userdata teardown.\n");
}

static void run_vulkan_hook_disarm_lifetime_case()
{
    fixture f(false, false);
    hook_lifetime_observer observer;
    auto state = std::make_unique<hook_lifetime_state>();
    std::mutex disarm_mutex;
    std::condition_variable disarm_cond;
    std::atomic<bool> disarm_returned = {false};
    bool disarm_done = false;

    state->observer = &observer;
    pacer_test_set_vulkan_ring_hook(f.queues.vulkan_queue,
            hook_lifetime_callback, state.get());
    std::thread producer([&]() {
        pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    });
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cond.wait(lock, [&]() { return state->entered; });
    }
    std::thread disarm([&]() {
        pacer_test_set_vulkan_ring_hook(f.queues.vulkan_queue,
                nullptr, nullptr);
        disarm_returned.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(disarm_mutex);
            disarm_done = true;
        }
        disarm_cond.notify_all();
    });
    pacer_test_wait_vulkan_ring_hook_drain(f.queues.vulkan_queue);
    check(!disarm_returned.load(std::memory_order_acquire),
            "vulkan-hook-disarm: disarm returned while the callback was paused.\n");
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        check(!state->exited,
                "vulkan-hook-disarm: callback exited before its explicit release.\n");
        state->release = true;
        state->cond.notify_all();
    }
    {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cond.wait(lock, [&]() { return state->exited; });
    }
    {
        std::unique_lock<std::mutex> lock(disarm_mutex);
        disarm_cond.wait(lock, [&]() { return disarm_done; });
    }

    check(disarm_returned.load(std::memory_order_acquire) &&
            observer.callbacks.load(std::memory_order_acquire) == 1 &&
            observer.exits.load(std::memory_order_acquire) == 1,
            "vulkan-hook-disarm: disarm did not return strictly after callback exit.\n");
    state.reset();
    pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    producer.join();
    disarm.join();
    check(observer.callbacks.load(std::memory_order_acquire) == 1 &&
            observer.exits.load(std::memory_order_acquire) == 1,
            "vulkan-hook-disarm: a callback touched userdata after disarm returned.\n");
    std::printf("vulkan-hook-disarm: disarm drained a forced in-flight callback before userdata teardown.\n");
}

static void run_command_ring_present_snapshot_case()
{
    fixture f(false, false);
    uint64_t command_id = pacer_command_queue_notify_legacy_submit(
            f.queues.command_queue);
    uint64_t vulkan_id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    command_ring_barrier barrier;

    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            command_id, vulkan_id),
            "command-ring-F: failed to publish legacy submit.\n");
    arm_command_ring_barrier(f, barrier,
            VKD3D_TEST_COMMAND_RING_PRESENT_AFTER_SNAPSHOT);
    std::thread worker([&]() { f.legacy_present(); });
    wait_command_ring_barrier(barrier);
    uint64_t reclaimed = reclaim_command_slot(f, command_id);
    uint64_t new_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            reclaimed, new_vulkan),
            "command-ring-F: failed to publish reclaimed generation.\n");
    f.legacy_present();
    vkd3d_test_framepacer_snapshot before = f.snapshot(reclaimed, new_vulkan);
    release_command_ring_barrier(f, barrier);
    worker.join();
    vkd3d_test_command_slot_snapshot slot = command_slot_snapshot(f, command_id);
    vkd3d_test_framepacer_snapshot after = f.snapshot(reclaimed, new_vulkan);
    check(slot.generation == reclaimed && slot.vulkan_id == new_vulkan &&
            slot.present_pending &&
            after.gpu_finished == before.gpu_finished,
            "command-ring-F: historical Present mutated or progressed the new slot.\n");
    std::printf("command-ring-F: Present used a protected historical snapshot after reclaim.\n");
}

static void run_command_ring_completion_cases()
{
    {
        fixture f(true, false);
        fixture::submit_pair submit = f.submit();
        f.present();
        uint64_t reclaimed = reclaim_command_slot(f, submit.command);
        f.complete_gpu(submit.command, submit.vulkan);
        vkd3d_test_framepacer_snapshot state =
                f.snapshot(submit.command, submit.vulkan);
        check(command_slot_snapshot(f, submit.command).generation == reclaimed &&
                !state.submit_pending && state.gpu_finished == 2,
                "command-ring-G-captured: completion depended on physical ring retention.\n");
    }

    {
        fixture f(false, false);
        uint64_t old_command = pacer_command_queue_notify_legacy_submit(
                f.queues.command_queue);
        uint64_t old_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
                old_command, old_vulkan),
                "command-ring-G-legacy: failed to publish old generation.\n");
        f.legacy_present();
        uint64_t reclaimed = reclaim_command_slot(f, old_command);
        uint64_t new_vulkan = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
        check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
                reclaimed, new_vulkan),
                "command-ring-G-legacy: failed to publish new generation.\n");
        f.legacy_present();
        uint64_t gpu_before = f.snapshot(reclaimed, new_vulkan).gpu_finished;
        f.complete_gpu(old_command, old_vulkan);
        vkd3d_test_command_slot_snapshot slot = command_slot_snapshot(f, old_command);
        check(slot.generation == reclaimed && slot.vulkan_id == new_vulkan &&
                slot.present_pending &&
                f.snapshot(reclaimed, new_vulkan).gpu_finished == gpu_before,
                "command-ring-G-legacy: old completion consumed newer metadata.\n");
    }

    std::printf("command-ring-G: captured completion survived reclaim; legacy completion stayed generation-local.\n");
}

static void run_command_ring_iterator_case()
{
    fixture f(false, false);
    uint64_t command_id = pacer_command_queue_notify_legacy_submit(
            f.queues.command_queue);
    uint64_t vulkan_id = pacer_vulkan_queue_notify_submit(f.queues.vulkan_queue);
    command_ring_barrier barrier;
    vkd3d_test_submit_iterator_snapshot iterator = {};

    check(pacer_command_queue_notify_vulkan_submit(f.queues.command_queue,
            command_id, vulkan_id),
            "command-ring-H: failed to publish iterator generation.\n");
    set_vulkan_timestamps(f, vulkan_id, historical_submit,
            historical_start, historical_end);
    arm_command_ring_barrier(f, barrier,
            VKD3D_TEST_COMMAND_RING_ITERATOR_AFTER_SNAPSHOT);
    std::thread worker([&]() {
        pacer_test_get_submit_iterator_snapshot(f.queues.command_queue,
                command_id, &iterator);
    });
    wait_command_ring_barrier(barrier);
    uint64_t reclaimed_command = reclaim_command_slot(f, command_id);
    pacer_queue_notify_submit_failed(f.queues, command_id, vulkan_id);
    uint64_t reclaimed_vulkan = reclaim_vulkan_slot(f, vulkan_id);
    set_vulkan_timestamps(f, reclaimed_vulkan, reclaimed_submit,
            reclaimed_start, reclaimed_end);
    release_command_ring_barrier(f, barrier);
    worker.join();

    vkd3d_test_vulkan_slot_snapshot current =
            vulkan_slot_snapshot(f, reclaimed_vulkan);
    check(command_slot_snapshot(f, command_id).generation == reclaimed_command &&
            current.found && current.gpu_execution_start == reclaimed_start &&
            current.gpu_execution_end == reclaimed_end &&
            iterator.vulkan_id == vulkan_id && iterator.has_app_submit &&
            !iterator.has_vulkan_submit &&
            !iterator.vulkan_gpu_execution_start &&
            !iterator.vulkan_gpu_execution_end,
            "command-ring-H: SubmitIterator returned mixed-generation state.\n");
    std::printf("command-ring-H: all SubmitIterator getters rejected M after both rings reclaimed.\n");
}

int main()
{
    static const event_type case_a[] = {
        event_type::gpu_completion, event_type::present, event_type::present,
    };
    static const event_type case_b[] = {
        event_type::present, event_type::gpu_completion, event_type::present,
    };
    static const event_type case_c[] = {
        event_type::present, event_type::present, event_type::gpu_completion,
    };

    run_fg_case("A", case_a);
    run_fg_case("B", case_b);
    run_fg_case("C", case_c);
    run_control_case();
    run_multiple_submit_case();
    run_out_of_order_simulation_case();
    run_submit_generation_cases();
    run_mode_transition_case();
    run_delayed_epoch_completion_case();
    run_cancelled_present_case("aborted-present");
    run_cancelled_present_case("test-present");
    run_cancelled_present_case("occluded-present");
    run_inactive_present_case();
    run_present_end_without_notify_case();
    run_partial_tracking_failure_case();
    run_reflex_gap_to_legacy_case();
    run_stale_present_after_disable_case();
    run_stale_present_after_reenable_case(false);
    run_stale_present_after_reenable_case(true);
    run_unidentified_terminal_case();
    run_sealed_unknown_owner_case();
    run_multiple_open_unknown_owner_case();
    run_generation_aware_sleep_case();
    run_stale_nvapi_sleep_publication_case();
    run_post_sleep_commit_toctou_case();
    run_prediction_generation_isolation_case();
    run_mode_publication_race_case();
    run_generation_tagged_frame_state_case();
    run_stale_waitable_task_case();
    run_device_construction_raii_case();
    run_waitable_teardown_case();
    run_capture_start_a_execute_b_end_a_case();
    run_capture_close_then_publish_case();
    run_capture_execute_after_end_case();
    run_capture_commit_and_abort_case();
    run_capture_ambiguous_case();
    run_capture_abort_exactly_once_case();
    run_capture_present_final_lease_case();
    run_capture_external_reuse_case();
    run_capture_invalid_markers_case();
    run_capture_historical_thread_case();
    run_capture_external_metadata_identity_case();
    run_capture_end_after_present_case();
    run_capture_long_session_case();
    run_capture_marker_order_cases();
    run_command_ring_legacy_publication_cases();
    run_command_ring_captured_rollback_cases();
    run_post_command_reclaim_failure_case();
    run_vulkan_ring_accounting_cases();
    run_vulkan_gpu_start_getter_case();
    run_vulkan_gpu_end_getter_case();
    run_vulkan_submit_getter_case();
    run_vulkan_repeated_getter_case();
    run_hook_disarm_lifetime_case();
    run_vulkan_hook_disarm_lifetime_case();
    run_command_ring_present_snapshot_case();
    run_command_ring_completion_cases();
    run_command_ring_iterator_case();

    if (failures)
    {
        std::fprintf(stderr, "framepacer_accounting: %u failure(s).\n", failures);
        return 1;
    }

    std::printf("framepacer_accounting: all accounting orderings reproduced.\n");
    return 0;
}
