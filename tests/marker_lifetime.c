/*
 * Copyright 2026 The vkd3d-proton authors
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#define COBJMACROS
#include <vkd3d.h>
#define VKD3D_DBG_CHANNEL VKD3D_DBG_CHANNEL_API
#include "vkd3d_private.h"

#include <stdio.h>

enum marker_lifetime_event
{
    MARKER_LIFETIME_ACQUIRED,
    MARKER_LIFETIME_CHAIN_INC,
    MARKER_LIFETIME_CHAIN_DEC,
    MARKER_LIFETIME_CHAIN_CLEANUP,
    MARKER_LIFETIME_QUEUE_ACQUIRED,
    MARKER_LIFETIME_PAIRED_REFS_ACQUIRED_INSIDE_SELECTION_LOCK,
    MARKER_LIFETIME_SELECTION_LOCK_RELEASED,
    MARKER_LIFETIME_CHAIN_RELEASE_BEGIN,
    MARKER_LIFETIME_CHAIN_RELEASE_COMPLETED,
    MARKER_LIFETIME_QUEUE_RELEASE_BEGIN,
    MARKER_LIFETIME_QUEUE_RELEASE_COMPLETED,
};

typedef void (*marker_lifetime_callback)(unsigned int event,
        unsigned int refcount, void *userdata);

extern bool dxgi_vk_swap_chain_test_marker_lifetime_init(
        struct d3d12_command_queue *queue, IDXGIVkSwapChain **out);
extern void dxgi_vk_swap_chain_test_marker_lifetime_set_hook(
        IDXGIVkSwapChain *chain,
        marker_lifetime_callback callback, void *userdata);
extern void dxgi_vk_swap_chain_test_marker_lifetime_set_vk_swapchain(
        IDXGIVkSwapChain *chain, VkSwapchainKHR vk_swapchain);
extern void dxgi_vk_swap_chain_test_marker_lifetime_register(struct d3d12_device *device,
        IDXGIVkSwapChain *chain);

struct marker_lifetime_state
{
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    IDXGIVkSwapChain *chain;
    struct d3d12_device *device;
    bool acquired;
    bool marker_paused;
    bool resume;
    bool complete;
    bool underflow;
    bool public_release_while_marker_paused;
    unsigned int chain_inc_count;
    unsigned int chain_dec_count;
    unsigned int chain_cleanup_count;
    unsigned int paired_refs_acquired_count;
    unsigned int queue_acquired_count;
    unsigned int selection_lock_released_count;
    unsigned int acquired_count;
    unsigned int chain_release_begin_count;
    unsigned int chain_release_completed_count;
    unsigned int queue_release_begin_count;
    unsigned int queue_release_completed_count;
    unsigned int queue_destruction_count;
    unsigned int event_sequence;
    unsigned int paired_refs_acquired_order;
    unsigned int queue_acquired_order;
    unsigned int selection_lock_released_order;
    unsigned int acquired_order;
    unsigned int chain_release_begin_order;
    unsigned int chain_release_completed_order;
    unsigned int queue_release_begin_order;
    unsigned int queue_release_completed_order;
    unsigned int marker_call_count;
    unsigned int marker_queue_refcount;
    uint64_t frame_id;
    VkLatencyMarkerNV marker;
    HRESULT marker_hr;
};

static struct marker_lifetime_state *marker_lifetime_state;

/* The standalone test links the core implementation directly. */
bool vkd3d_debug_control_is_test_suite(void)
{
    return true;
}

bool vkd3d_debug_control_explode_on_vvl_error(void)
{
    return false;
}

bool vkd3d_debug_control_has_out_of_spec_test_behavior(unsigned int behavior)
{
    (void)behavior;
    return false;
}

unsigned int vkd3d_debug_control_get_behavior_flags(void)
{
    return 0;
}

bool vkd3d_debug_control_mute_message_id(const char *vuid)
{
    (void)vuid;
    return false;
}

static void marker_lifetime_hook(unsigned int event, unsigned int refcount, void *userdata)
{
    struct marker_lifetime_state *state = userdata;

    pthread_mutex_lock(&state->mutex);

    if (refcount > 3)
        state->underflow = true;

    switch (event)
    {
        case MARKER_LIFETIME_ACQUIRED:
            state->acquired = true;
            state->marker_paused = true;
            ++state->acquired_count;
            state->acquired_order = ++state->event_sequence;
            pthread_cond_broadcast(&state->cond);

            while (!state->resume)
                pthread_cond_wait(&state->cond, &state->mutex);
            state->marker_paused = false;
            break;

        case MARKER_LIFETIME_CHAIN_INC:
            ++state->chain_inc_count;
            break;

        case MARKER_LIFETIME_CHAIN_DEC:
            ++state->chain_dec_count;
            break;

        case MARKER_LIFETIME_CHAIN_CLEANUP:
            ++state->chain_cleanup_count;
            break;

        case MARKER_LIFETIME_QUEUE_ACQUIRED:
            ++state->queue_acquired_count;
            state->queue_acquired_order = ++state->event_sequence;
            break;

        case MARKER_LIFETIME_PAIRED_REFS_ACQUIRED_INSIDE_SELECTION_LOCK:
            ++state->paired_refs_acquired_count;
            state->paired_refs_acquired_order = ++state->event_sequence;
            break;

        case MARKER_LIFETIME_SELECTION_LOCK_RELEASED:
            ++state->selection_lock_released_count;
            state->selection_lock_released_order = ++state->event_sequence;
            break;

        case MARKER_LIFETIME_CHAIN_RELEASE_BEGIN:
            ++state->chain_release_begin_count;
            state->chain_release_begin_order = ++state->event_sequence;
            break;

        case MARKER_LIFETIME_CHAIN_RELEASE_COMPLETED:
            ++state->chain_release_completed_count;
            state->chain_release_completed_order = ++state->event_sequence;
            break;

        case MARKER_LIFETIME_QUEUE_RELEASE_BEGIN:
            ++state->queue_release_begin_count;
            state->queue_release_begin_order = ++state->event_sequence;
            break;

        case MARKER_LIFETIME_QUEUE_RELEASE_COMPLETED:
            ++state->queue_release_completed_count;
            state->queue_release_completed_order = ++state->event_sequence;
            state->marker_queue_refcount = refcount;
            if (!refcount)
                ++state->queue_destruction_count;
            break;
    }

    pthread_mutex_unlock(&state->mutex);
}

static VKAPI_ATTR void VKAPI_CALL marker_lifetime_vk_set_latency_marker(
        VkDevice device, VkSwapchainKHR swapchain, const VkSetLatencyMarkerInfoNV *info)
{
    struct marker_lifetime_state *state = marker_lifetime_state;

    (void)device;
    (void)swapchain;

    pthread_mutex_lock(&state->mutex);
    ++state->marker_call_count;
    state->frame_id = info->presentID;
    state->marker = info->marker;

    /* This endpoint is the sole mocked Vulkan call. The opaque test handle is
     * cleared before the real swapchain cleanup later runs. */
    dxgi_vk_swap_chain_test_marker_lifetime_set_vk_swapchain(state->chain, VK_NULL_HANDLE);
    pthread_mutex_unlock(&state->mutex);
}

static void *marker_lifetime_thread(void *userdata)
{
    struct marker_lifetime_state *state = userdata;
    struct d3d12_device *device = state->device;

    state->marker_hr = ID3DLowLatencyDevice_SetLatencyMarker(
            &device->ID3DLowLatencyDevice_iface,
            UINT64_C(0x123456789abcdef0),
            VK_LATENCY_MARKER_RENDERSUBMIT_START_NV);

    pthread_mutex_lock(&state->mutex);
    state->complete = true;
    pthread_cond_broadcast(&state->cond);
    pthread_mutex_unlock(&state->mutex);
    return NULL;
}

static bool marker_lifetime_test(void)
{
    static const char * const device_extensions[] =
    {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };
    static const char * const instance_extensions[] =
    {
        VK_KHR_SURFACE_EXTENSION_NAME,
    };
    static const struct vkd3d_instance_create_info instance_info =
    {
        .pfn_vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .instance_extensions = instance_extensions,
        .instance_extension_count = ARRAY_SIZE(instance_extensions),
    };
    static const struct vkd3d_device_create_info device_info =
    {
        .minimum_feature_level = D3D_FEATURE_LEVEL_11_0,
        .instance_create_info = &instance_info,
        .device_extensions = device_extensions,
        .device_extension_count = ARRAY_SIZE(device_extensions),
    };
    struct marker_lifetime_state state = {0};
    struct d3d12_command_queue *queue;
    struct d3d12_device *device_impl;
    IDXGIVkSwapChain *chain;
    ID3D12CommandQueue *queue_iface;
    ID3D12Device *device;
    D3D12_COMMAND_QUEUE_DESC desc = {0};
    PFN_vkSetLatencyMarkerNV saved_marker;
    bool saved_low_latency;
    pthread_t thread;
    ULONG refcount;
    HRESULT hr;
    bool result;

    hr = vkd3d_create_device(&device_info, &IID_ID3D12Device, (void **)&device);
    if (FAILED(hr))
    {
        fprintf(stderr, "marker-lifetime: device creation failed, hr %#lx.\n", (long)hr);
        return false;
    }

    desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = ID3D12Device_CreateCommandQueue(device, &desc,
            &IID_ID3D12CommandQueue, (void **)&queue_iface);
    if (FAILED(hr))
    {
        fprintf(stderr, "marker-lifetime: queue creation failed, hr %#lx.\n", (long)hr);
        ID3D12Device_Release(device);
        return false;
    }

    device_impl = impl_from_ID3D12Device((d3d12_device_iface *)device);
    if (!device_impl->vk_procs.vkDestroySwapchainKHR ||
            !device_impl->vkd3d_instance->vk_procs.vkDestroySurfaceKHR)
    {
        fprintf(stderr, "marker-lifetime: vkDestroySwapchainKHR unavailable.\n");
        ID3D12CommandQueue_Release(queue_iface);
        ID3D12Device_Release(device);
        return false;
    }
    queue = CONTAINING_RECORD(queue_iface, struct d3d12_command_queue,
            ID3D12CommandQueue_iface);

    if (!dxgi_vk_swap_chain_test_marker_lifetime_init(queue, &chain))
    {
        fprintf(stderr, "marker-lifetime: chain initialization failed.\n");
        ID3D12CommandQueue_Release(queue_iface);
        ID3D12Device_Release(device);
        return false;
    }

    pthread_mutex_init(&state.mutex, NULL);
    pthread_cond_init(&state.cond, NULL);
    state.chain = chain;
    state.device = device_impl;
    marker_lifetime_state = &state;

    saved_low_latency = device_impl->vk_info.NV_low_latency2;
    saved_marker = device_impl->vk_procs.vkSetLatencyMarkerNV;
    device_impl->vk_info.NV_low_latency2 = true;
    device_impl->vk_procs.vkSetLatencyMarkerNV = marker_lifetime_vk_set_latency_marker;
    dxgi_vk_swap_chain_test_marker_lifetime_set_vk_swapchain(chain,
            (VkSwapchainKHR)(uintptr_t)1);

    dxgi_vk_swap_chain_test_marker_lifetime_register(device_impl, chain);
    dxgi_vk_swap_chain_test_marker_lifetime_set_hook(chain,
            marker_lifetime_hook, &state);

    if (pthread_create(&thread, NULL, marker_lifetime_thread, &state))
    {
        fprintf(stderr, "marker-lifetime: marker thread creation failed.\n");
        return false;
    }

    pthread_mutex_lock(&state.mutex);
    while (!state.acquired)
        pthread_cond_wait(&state.cond, &state.mutex);
    if (state.marker_paused && !state.resume && !state.complete)
        state.public_release_while_marker_paused = true;
    pthread_mutex_unlock(&state.mutex);

    /* Real public Release removes selection and normal chain/queue ownership. */
    refcount = IDXGIVkSwapChain_Release(chain);
    if (refcount)
        state.underflow = true;

    /* Only the real marker queue reference must remain now. */
    refcount = ID3D12CommandQueue_Release(queue_iface);
    if (refcount != 1)
        state.underflow = true;

    pthread_mutex_lock(&state.mutex);
    state.resume = true;
    pthread_cond_broadcast(&state.cond);
    while (!state.complete)
        pthread_cond_wait(&state.cond, &state.mutex);
    pthread_mutex_unlock(&state.mutex);
    pthread_join(thread, NULL);

    result = state.marker_hr == S_OK &&
            state.marker_call_count == 1 &&
            state.frame_id == UINT64_C(0x123456789abcdef0) &&
            state.marker == VK_LATENCY_MARKER_RENDERSUBMIT_START_NV &&
            state.chain_inc_count == 1 &&
            state.chain_dec_count == 3 &&
            state.chain_cleanup_count == 1 &&
            state.paired_refs_acquired_count == 1 &&
            state.queue_acquired_count == 1 &&
            state.selection_lock_released_count == 1 &&
            state.acquired_count == 1 &&
            state.queue_acquired_order < state.paired_refs_acquired_order &&
            state.paired_refs_acquired_order < state.selection_lock_released_order &&
            state.selection_lock_released_order < state.acquired_order &&
            state.public_release_while_marker_paused &&
            state.chain_release_begin_count == 1 &&
            state.chain_release_completed_count == 1 &&
            state.queue_release_begin_count == 1 &&
            state.queue_release_completed_count == 1 &&
            state.chain_release_begin_order < state.chain_release_completed_order &&
            state.chain_release_completed_order < state.queue_release_begin_order &&
            state.queue_release_begin_order < state.queue_release_completed_order &&
            state.queue_destruction_count == 1 &&
            state.queue_acquired_order &&
            state.marker_queue_refcount == 0 &&
            !state.underflow &&
            !device_impl->swapchain_info.low_latency_swapchain;

    if (!result)
        fprintf(stderr, "marker-lifetime: hr %#lx, calls %u, id %#llx, marker %u, inc %u, dec %u, cleanup %u, pair %u/%u, queue acquire %u/%u, unlock %u/%u, pause %u/%u, public while paused %u, chain release %u/%u then %u/%u, queue release %u/%u then %u/%u, queue destroys %u, queue ref %u, underflow %u, selected %p.\n",
                (long)state.marker_hr, state.marker_call_count, (unsigned long long)state.frame_id,
                state.marker, state.chain_inc_count, state.chain_dec_count,
                state.chain_cleanup_count,
                state.paired_refs_acquired_count, state.paired_refs_acquired_order,
                state.queue_acquired_count, state.queue_acquired_order,
                state.selection_lock_released_count, state.selection_lock_released_order,
                state.acquired_count, state.acquired_order,
                state.public_release_while_marker_paused,
                state.chain_release_begin_count, state.chain_release_begin_order,
                state.chain_release_completed_count, state.chain_release_completed_order,
                state.queue_release_begin_count, state.queue_release_begin_order,
                state.queue_release_completed_count, state.queue_release_completed_order,
                state.queue_destruction_count, state.marker_queue_refcount, state.underflow,
                device_impl->swapchain_info.low_latency_swapchain);

    device_impl->vk_procs.vkSetLatencyMarkerNV = saved_marker;
    device_impl->vk_info.NV_low_latency2 = saved_low_latency;
    dxgi_vk_swap_chain_test_marker_lifetime_set_hook(NULL, NULL, NULL);
    marker_lifetime_state = NULL;
    pthread_cond_destroy(&state.cond);
    pthread_mutex_destroy(&state.mutex);

    if (ID3D12Device_Release(device))
        result = false;

    return result;
}

int main(void)
{
    if (!marker_lifetime_test())
    {
        fprintf(stderr, "marker-lifetime: regression failed.\n");
        return 1;
    }

    return 0;
}
