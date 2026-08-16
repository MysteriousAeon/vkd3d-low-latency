#ifndef VKD3D_TEST_HOOKS_H
#define VKD3D_TEST_HOOKS_H

#include "vkd3d_windows.h"

static const GUID VKD3D_TEST_QUEUE_TRANSITION_HOOK_GUID =
        {0xed2e30da, 0xbbd6, 0x4d9f, {0x9d, 0xc8, 0x0e, 0x21, 0xb8, 0x65, 0x1d, 0x39}};
static const GUID VKD3D_TEST_QUEUE_TRANSITION_HOOK_DRAIN_GUID =
        {0xd8512ad8, 0xfd27, 0x46f4, {0xb4, 0xca, 0xd8, 0x44, 0xd1, 0xaf, 0x8a, 0x4f}};
static const GUID VKD3D_TEST_CAPTURE_CONTROL_GUID =
        {0x4df90f0b, 0x17f8, 0x4f47, {0x82, 0xec, 0xef, 0x5a, 0x65, 0xea, 0xb5, 0xa2}};

enum vkd3d_test_queue_transition_hook_point
{
    VKD3D_TEST_QUEUE_TRANSITION_HOOK_EXECUTE_CAPTURE_ACQUIRED,
    VKD3D_TEST_QUEUE_TRANSITION_HOOK_EXECUTE_PREPARATION_BEGIN,
    VKD3D_TEST_QUEUE_TRANSITION_HOOK_PACER_GPU_COMPLETE,
};

typedef void (*vkd3d_test_queue_transition_hook_callback)(
        enum vkd3d_test_queue_transition_hook_point point, void *userdata);

struct vkd3d_test_queue_transition_hook
{
    /* Mutating this private test hook from any transition-hook callback is
     * unsupported and is rejected with E_UNEXPECTED. */
    vkd3d_test_queue_transition_hook_callback callback;
    void *userdata;
};

#ifdef VKD3D_ENABLE_TEST_HOOKS

struct pacer_device_properties;
struct pacer_device_vk_procs;

struct vkd3d_test_framepacer_snapshot
{
    uint64_t cpu_finished;
    uint64_t gpu_finished;
    uint64_t accounting_state;
    uint64_t command_submit_count;
    uint64_t command_vulkan_submit_id;
    uint64_t vulkan_gpu_timestamp;
    uint64_t first_internal_frame_id;
    uint64_t first_external_frame_id;
    uint64_t second_internal_frame_id;
    uint64_t second_external_frame_id;
    uint64_t submit_simulation_id;
    uint32_t presentation_count;
    bool submit_pending;
    bool tracking_bypassed;
    bool reflex_accounting;
};

enum vkd3d_test_command_ring_hook_point
{
    VKD3D_TEST_COMMAND_RING_PRODUCER_BEFORE_SLOT,
    VKD3D_TEST_COMMAND_RING_PRODUCER_AFTER_SLOT,
    VKD3D_TEST_COMMAND_RING_VULKAN_BEFORE_SLOT,
    VKD3D_TEST_COMMAND_RING_VULKAN_AFTER_SLOT,
    VKD3D_TEST_COMMAND_RING_PRESENT_AFTER_SNAPSHOT,
    VKD3D_TEST_COMMAND_RING_PRESENT_AFTER_CLAIM,
    VKD3D_TEST_COMMAND_RING_COMPLETION_AFTER_SLOT,
    VKD3D_TEST_COMMAND_RING_COMPLETION_BEFORE_ACCOUNTING,
    VKD3D_TEST_COMMAND_RING_COMPLETION_AFTER_ACCOUNTING,
    VKD3D_TEST_COMMAND_RING_ITERATOR_AFTER_SNAPSHOT,
};

enum vkd3d_test_vulkan_ring_hook_point
{
    VKD3D_TEST_VULKAN_RING_PRODUCER_BEFORE_SLOT,
    VKD3D_TEST_VULKAN_RING_PRODUCER_AFTER_SLOT,
    VKD3D_TEST_VULKAN_RING_FINISH_BEFORE_SLOT,
    VKD3D_TEST_VULKAN_RING_FINISH_AFTER_SLOT,
    VKD3D_TEST_VULKAN_RING_GETTER_BEFORE_SLOT,
};

typedef void (*vkd3d_test_command_ring_hook_callback)(uint32_t point,
        void *userdata);
typedef void (*vkd3d_test_vulkan_ring_hook_callback)(uint32_t point,
        void *userdata);

struct vkd3d_test_command_slot_snapshot
{
    uint64_t generation;
    uint64_t vulkan_id;
    uint64_t present_epoch;
    uint64_t submit_timestamp;
    bool present_pending;
    bool has_submit_timestamp;
    bool ledger_completion_accounted;
};

struct vkd3d_test_vulkan_slot_snapshot
{
    uint64_t generation;
    uint64_t submit_timestamp;
    uint64_t gpu_execution_start;
    uint64_t gpu_execution_end;
    bool found;
    bool has_submit_timestamp;
    bool submit_accounted;
};

struct vkd3d_test_submit_iterator_snapshot
{
    uint64_t vulkan_id;
    uint64_t vulkan_submit_timestamp;
    uint64_t vulkan_gpu_execution_start;
    uint64_t vulkan_gpu_execution_end;
    bool has_app_submit;
    bool has_vulkan_submit;
};

struct vkd3d_test_capture_snapshot
{
    uint64_t accounting_epoch;
    uint64_t simulation_id;
    uint64_t capture_generation;
    uint32_t capture_state;
    uint32_t failure_reason;
    uint32_t in_flight_publications;
    uint32_t open_capture_count;
    uint32_t active_lease_count;
    uint32_t capture_record_count;
    uint32_t end_association_count;
    uint64_t highest_started_external_id;
    uint32_t published_submits;
    uint32_t accounted_submits;
    uint32_t completed_submits;
    bool submission_seal_requested;
    bool cpu_seal_requested;
    bool submissions_sealed;
    bool cpu_sealed;
    bool tracking_failed;
    uint64_t cpu_finished;
    uint64_t gpu_finished;
};

enum vkd3d_test_capture_control_action
{
    VKD3D_TEST_CAPTURE_CONTROL_OPEN,
    VKD3D_TEST_CAPTURE_CONTROL_PRESENT,
    VKD3D_TEST_CAPTURE_CONTROL_END,
};

struct vkd3d_test_capture_control
{
    uint64_t external_reflex_id;
    uint32_t action;
};

struct vkd3d_test_latency_sleep_decision
{
    uint64_t call_count;
    uint64_t entry_accounting_state;
    uint64_t exit_accounting_state;
    uint64_t frame_id;
    uint64_t wait_id;
    uint64_t cpu_finished;
    uint64_t gpu_finished;
    bool wait_satisfied;
    bool would_start_frame;
};

struct vkd3d_test_frame_state
{
    uint64_t accounting_state;
    uint64_t frame_id;
    uint64_t external_frame_id;
    int32_t marker_value;
    bool marker_present;
};

struct vkd3d_test_waitable_task_stats
{
    uint64_t marker_reads;
    uint64_t marker_writes;
    uint64_t rejected_tasks;
};

struct vkd3d_test_nvapi_adapter_state
{
    uint64_t accounting_epoch;
    uint64_t simulation_id;
    int64_t drift;
    int64_t last_end_sleep;
    bool pending_sleep;
};

struct vkd3d_test_prediction_state
{
    uint64_t accounting_state;
    uint64_t finished_frame_id;
    int32_t predicted_gpu_time;
};

#ifdef __cplusplus
extern "C" {
#endif

void pacer_test_set_wait_latency(void *device, uint32_t wait_latency);
void pacer_test_set_latency_sleep_bypass(void *device, bool bypass);
void pacer_test_reset_latency_sleep_decision(void *device);
void pacer_test_get_latency_sleep_decision(void *device,
        struct vkd3d_test_latency_sleep_decision *decision);
void pacer_test_sleep_for_frame(void *device, uint64_t frame_id);
uint64_t pacer_test_get_sleep_entry_count(void *device);
void pacer_test_pause_after_sleep(void *device, bool pause);
bool pacer_test_sleep_returned(void *device);
void pacer_test_arm_simulation_marker_after_arrival(void *device);
void pacer_test_wait_simulation_marker_after_arrival(void *device);
void pacer_test_resume_simulation_marker_after_arrival(void *device);
void pacer_test_get_nvapi_adapter_state(void *device,
        struct vkd3d_test_nvapi_adapter_state *state);
void pacer_test_set_prediction(void *device, uint64_t frame_id,
        int32_t optimized_gpu_time);
void pacer_test_get_prediction_state(void *device,
        struct vkd3d_test_prediction_state *state);
bool pacer_test_get_last_teardown_worker_stopped(void);
void pacer_test_set_frame_state(void *device, uint64_t frame_id,
        uint64_t external_frame_id, int32_t marker_value);
void pacer_test_get_frame_state(void *device, uint64_t frame_id,
        struct vkd3d_test_frame_state *state);
void pacer_test_queue_waitable_task(void *device, uint64_t accounting_state,
        uint64_t cpu_id);
void pacer_test_wait_waitable_task_dequeued(void *device);
void pacer_test_resume_waitable_task(void *device);
void pacer_test_wait_waitable_task_idle(void *device);
void pacer_test_get_waitable_task_stats(void *device,
        struct vkd3d_test_waitable_task_stats *stats);
bool pacer_test_device_construction_failure_cleanup(
        struct pacer_device_properties *properties,
        struct pacer_device_vk_procs *vk_procs);
void pacer_test_get_framepacer_snapshot(void *device, void *command_queue,
        void *vulkan_queue, uint64_t command_submit_id,
        uint64_t vulkan_submit_id, uint64_t first_internal_frame_id,
        uint64_t second_internal_frame_id,
        struct vkd3d_test_framepacer_snapshot *snapshot);
void pacer_test_get_capture_snapshot(void *device, uint64_t capture_generation,
        struct vkd3d_test_capture_snapshot *snapshot);
bool pacer_test_accept_present(void *device, uint64_t external_reflex_id);
void pacer_test_set_command_ring_hook(void *command_queue,
        vkd3d_test_command_ring_hook_callback callback, void *userdata);
void pacer_test_pause_command_ring_hook_dispatch(void *command_queue);
void pacer_test_wait_command_ring_hook_dispatch(void *command_queue);
void pacer_test_wait_command_ring_hook_cleared(void *command_queue);
void pacer_test_wait_command_ring_hook_drain(void *command_queue);
void pacer_test_resume_command_ring_hook_dispatch(void *command_queue);
void pacer_test_set_vulkan_ring_hook(void *vulkan_queue,
        vkd3d_test_vulkan_ring_hook_callback callback, void *userdata);
void pacer_test_wait_vulkan_ring_hook_drain(void *vulkan_queue);
void pacer_test_set_vulkan_submit_timestamp(void *vulkan_queue,
        uint64_t vulkan_id, uint64_t timestamp);
void pacer_test_set_vulkan_gpu_execution_start(void *vulkan_queue,
        uint64_t vulkan_id, uint64_t timestamp);
void pacer_test_set_vulkan_gpu_execution_end(void *vulkan_queue,
        uint64_t vulkan_id, uint64_t timestamp);
void pacer_test_get_vulkan_slot_snapshot(void *vulkan_queue,
        uint64_t vulkan_id, struct vkd3d_test_vulkan_slot_snapshot *snapshot);
void pacer_test_get_command_slot_snapshot(void *command_queue,
        uint64_t command_id, struct vkd3d_test_command_slot_snapshot *snapshot);
bool pacer_test_set_command_present_state(void *command_queue,
        uint64_t command_id, uint64_t present_epoch);
bool pacer_test_set_command_submit_timestamp(void *command_queue,
        uint64_t command_id, uint64_t timestamp);
void pacer_test_get_submit_iterator_snapshot(void *command_queue,
        uint64_t command_id,
        struct vkd3d_test_submit_iterator_snapshot *snapshot);

#ifdef __cplusplus
}
#endif

#endif

#endif
