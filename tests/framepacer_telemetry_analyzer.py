#!/usr/bin/env python3
"""Deterministic refusal tests for vkd3d-fg-latency-analyze.py."""

import json
import os
import subprocess
import sys
import tempfile
import unittest


ANALYZER = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "programs", "vkd3d-fg-latency-analyze.py")
del sys.argv[1:]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(ANALYZER)))


def run_capture(records=None, raw=None):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as stream:
        path = stream.name
        if raw is not None:
            stream.write(raw)
        else:
            for record in records:
                stream.write(json.dumps(record) + "\n")
    try:
        return subprocess.run([sys.executable, ANALYZER, path], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False)
    finally:
        os.unlink(path)


def run_record(schema=2):
    return {
        "record_type": "RUN", "schema_version": schema, "run_id": "run",
        "build_id": "test", "process_id": 1, "start_cpu_timestamp_ns": 100,
        "cpu_clock": "dxvk_high_resolution_clock_ns", "wait_latency": 3,
        "ring_capacity": 8,
    }


def summary(published, dropped=0, schema=2):
    return {
        "record_type": "SUMMARY", "schema_version": schema, "run_id": "run",
        "published_records": published, "dropped_records": dropped,
        "end_cpu_timestamp_ns": 1000,
    }


def event(record_type, sequence, **values):
    result = {
        "record_type": record_type, "schema_version": 2,
        "event_sequence": sequence, "phase": "COMPLETE",
        "cpu_timestamp_ns": 500, "device_id": 1,
        "epoch_id": 1, "simulation_id": 1,
        "external_reflex_id": 0, "capture_generation": 1, "flags": 2,
        "queue_role": "NORMAL", "capture_class": "RENDER_CAPTURED",
    }
    result.update(values)
    if record_type == "GPU_FRONTIER":
        result.setdefault("published_submit_count", 1)
        result.setdefault("completed_submit_count", 1)
    return result


def first_failure(sequence, **values):
    result = event("FIRST_FAILURE", sequence, schema_version=3,
                   phase="COMPLETE", flags=1,
                   queue_role="UNKNOWN", capture_class="AMBIGUOUS")
    result.update({
        "failure_reason": "INVALID_PRESENT_ATTEMPT_ZERO_TOKEN",
        "cpu_finished_watermark": 1, "gpu_finished_watermark": 1,
        "context_id": 0, "context_value0": 5, "context_value1": 7,
        "context_value2": 9, "context_count0": 0, "context_count1": 42,
        "reflex_accounting_active": True, "present_token_provided": False,
        "caller_token_thread_match": False,
    })
    result.update(values)
    return result


def marker(sequence, kind, arrival, serialization, disposition, **values):
    result = event("MARKER", sequence, schema_version=4, epoch_id=5,
                   external_reflex_id=1, phase="COMPLETE",
                   queue_role="UNKNOWN", capture_class="AMBIGUOUS",
                   marker_kind=kind, marker_arrival_sequence=arrival,
                   observed_accounting_state=5, observed_reflex_epoch=5,
                   serialization_sequence=serialization,
                   disposition=disposition, thread_id=17)
    result.update(values)
    return result


class AnalyzerValidationTests(unittest.TestCase):
    def test_unsupported_schema(self):
        result = run_capture([run_record(1), summary(0, schema=1)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported schema_version", result.stderr)

    def test_missing_summary_is_incomplete(self):
        result = run_capture([run_record()])
        self.assertEqual(result.returncode, 2)
        self.assertIn("incomplete capture", result.stderr)

    def test_malformed_truncated_json(self):
        result = run_capture(raw=json.dumps(run_record()) + "\n{")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid JSON on line 2", result.stderr)

    def test_missing_required_event_field(self):
        record = event("GPU_FRONTIER", 1, gpu_completion_host_ns=400,
                       publication_cpu_ns=500)
        del record["cpu_timestamp_ns"]
        result = run_capture([run_record(), record, summary(1)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing required field cpu_timestamp_ns", result.stderr)

    def test_gpu_frontier_rejects_untruthful_submit_counts(self):
        record = event("GPU_FRONTIER", 1, gpu_completion_host_ns=400,
                       publication_cpu_ns=500, published_submit_count=0,
                       completed_submit_count=0)
        result = run_capture([run_record(), record, summary(1)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("GPU_FRONTIER submit counts are not truthful", result.stderr)

    def test_valid_first_failure_is_summarized_without_latency_causality(self):
        failure = first_failure(1)
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FIRST_FAILURE: device=1 epoch=1 "
                      "reason=INVALID_PRESENT_ATTEMPT_ZERO_TOKEN", result.stdout)
        self.assertNotIn("subreason=", result.stdout)

    def test_invalid_render_start_subreason_is_accepted_and_reported(self):
        failure = first_failure(1, failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT")
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("reason=INVALID_RENDER_START", result.stdout)
        self.assertIn("subreason=MAPPING_ABSENT", result.stdout)

    def test_all_known_invalid_render_start_subreasons_are_accepted(self):
        for subreason in (
                "ZERO_EXTERNAL_ID", "NON_MONOTONIC_OR_DUPLICATE",
                "MAPPING_ABSENT", "MAPPING_TARGET_MISSING",
                "MAPPED_WRONG_EPOCH", "MAPPED_SUBMISSIONS_SEALED",
                "MAPPED_TRACKING_FAILED"):
            failure = first_failure(1, failure_reason="INVALID_RENDER_START",
                                    invalid_render_start_subreason=subreason)
            result = run_capture([run_record(3), failure, summary(1, schema=3)])
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_capture_006_style_v3_first_failure_without_subreason_remains_accepted(self):
        failure = first_failure(1, failure_reason="INVALID_RENDER_START")
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("subreason=", result.stdout)

    def test_capture_007_style_v3_without_markers_refuses_marker_order_claim(self):
        failure = first_failure(1, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT")
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ORDERING_NOT_DISTINGUISHABLE (causal render identity unavailable)",
                      result.stdout)

    def test_marker_diagnostic_binds_causal_render_despite_publication_inversion(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        # R2 is published before causal R1. Selecting by event_sequence would
        # call this an arrival-first/serialization-second inversion; R1 proves late.
        later_render = marker(2, "RENDERSUBMIT_START", 13, 21, "REACHED_RENDER_START")
        causal_render = marker(3, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(4, "SIMULATION_START", 12, 22, "ACCEPTED_MAPPING")
        result = run_capture([run_record(4), failure, later_render, causal_render,
                              simulation, summary(4, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_START_LATE_AFTER_RENDER", result.stdout)
        self.assertNotIn("SIMULATION_START_ARRIVED_FIRST_BUT_SERIALIZED_AFTER_RENDER",
                         result.stdout)

    def test_marker_diagnostic_reports_arrival_first_serialization_second(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(3, "SIMULATION_START", 10, 21, "ACCEPTED_MAPPING")
        result = run_capture([run_record(4), failure, render, simulation, summary(3, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_START_ARRIVED_FIRST_BUT_SERIALIZED_AFTER_RENDER",
                      result.stdout)

    def test_marker_diagnostic_refuses_epoch_reuse(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(3, "SIMULATION_START", 10, 19, "ACCEPTED_MAPPING",
                            observed_accounting_state=7, observed_reflex_epoch=7)
        result = run_capture([run_record(4), failure, render, simulation, summary(3, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_START_NOT_SEEN_BEFORE_FAILURE", result.stdout)

    def test_marker_diagnostic_refuses_other_device(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(3, "SIMULATION_START", 10, 19, "ACCEPTED_MAPPING",
                            device_id=2)
        result = run_capture([run_record(4), failure, render, simulation, summary(3, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_START_NOT_SEEN_BEFORE_FAILURE", result.stdout)

    def test_marker_diagnostic_reports_inactive_without_epoch_claim(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(3, "SIMULATION_START", 10, 0, "IGNORED_INACTIVE",
                            observed_accounting_state=2, observed_reflex_epoch=0)
        result = run_capture([run_record(4), failure, render, simulation, summary(3, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_START_IGNORED_INACTIVE", result.stdout)
        self.assertNotIn("PRE_ACTIVATION", result.stdout)

    def test_marker_diagnostic_reports_different_external_id(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(3, "SIMULATION_START", 10, 19, "ACCEPTED_MAPPING",
                            external_reflex_id=2)
        result = run_capture([run_record(4), failure, render, simulation, summary(3, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIMULATION_START_DIFFERENT_EXTERNAL_ID", result.stdout)

    def test_marker_diagnostic_reports_not_seen_only_without_drops(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        complete = run_capture([run_record(4), failure, render, summary(2, schema=4)])
        self.assertEqual(complete.returncode, 0, complete.stderr)
        self.assertIn("SIMULATION_START_NOT_SEEN_BEFORE_FAILURE", complete.stdout)
        dropped = run_capture([run_record(4), failure, render, summary(2, dropped=1, schema=4)])
        self.assertEqual(dropped.returncode, 0, dropped.stderr)
        self.assertIn("ORDERING_NOT_DISTINGUISHABLE (marker evidence incomplete", dropped.stdout)

    def test_marker_diagnostic_same_epoch_simulation_before_render_is_conservative(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT",
                                originating_marker_serialization_sequence=20)
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        simulation = marker(3, "SIMULATION_START", 10, 19, "ACCEPTED_MAPPING")
        result = run_capture([run_record(4), failure, render, simulation, summary(3, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ORDERING_NOT_DISTINGUISHABLE", result.stdout)

    def test_schema4_without_causal_identity_is_conservative(self):
        failure = first_failure(1, schema_version=4, epoch_id=5, external_reflex_id=1,
                                failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="MAPPING_ABSENT")
        render = marker(2, "RENDERSUBMIT_START", 11, 20, "REACHED_RENDER_START")
        result = run_capture([run_record(4), failure, render, summary(2, schema=4)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ORDERING_NOT_DISTINGUISHABLE (causal render identity unavailable)",
                      result.stdout)

    def test_originating_marker_identity_requires_invalid_render_start(self):
        failure = first_failure(1, schema_version=4,
                                originating_marker_serialization_sequence=20)
        result = run_capture([run_record(4), failure, summary(1, schema=4)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires INVALID_RENDER_START", result.stderr)

    def test_originating_marker_identity_rejects_malformed_value(self):
        for causal_sequence in (0, -1, True, "20"):
            with self.subTest(causal_sequence=causal_sequence):
                failure = first_failure(1, schema_version=4,
                                        failure_reason="INVALID_RENDER_START",
                                        originating_marker_serialization_sequence=causal_sequence)
                result = run_capture([run_record(4), failure, summary(1, schema=4)])
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a positive int", result.stderr)

    def test_invalid_render_start_subreason_rejects_explicit_null(self):
        failure = first_failure(1, failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason=None)
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be str", result.stderr)

    def test_invalid_render_start_subreason_rejects_non_string_types(self):
        for subreason in (1, True, [], {}):
            with self.subTest(subreason=subreason):
                failure = first_failure(1, failure_reason="INVALID_RENDER_START",
                                        invalid_render_start_subreason=subreason)
                result = run_capture([run_record(3), failure, summary(1, schema=3)])
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be str", result.stderr)

    def test_unknown_invalid_render_start_subreason_is_rejected(self):
        failure = first_failure(1, failure_reason="INVALID_RENDER_START",
                                invalid_render_start_subreason="SPECULATIVE")
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported INVALID_RENDER_START subreason", result.stderr)

    def test_invalid_render_start_subreason_requires_matching_reason(self):
        failure = first_failure(1,
                                invalid_render_start_subreason="MAPPING_ABSENT")
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires INVALID_RENDER_START", result.stderr)

    def test_null_subreason_requires_matching_reason(self):
        failure = first_failure(1, invalid_render_start_subreason=None)
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires INVALID_RENDER_START", result.stderr)

    def test_first_failure_requires_reason(self):
        failure = first_failure(1)
        del failure["failure_reason"]
        result = run_capture([run_record(3), failure, summary(1, schema=3)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing required field failure_reason", result.stderr)

    def test_duplicate_first_failure_is_rejected(self):
        first = first_failure(1)
        second = first_failure(2, failure_reason="COMPLETION_FAILURE")
        result = run_capture([run_record(3), first, second, summary(2, schema=3)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate FIRST_FAILURE", result.stderr)

    def test_capture_without_first_failure_remains_valid(self):
        result = run_capture([run_record(3), summary(0, schema=3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FIRST_FAILURE: none", result.stdout)

    def test_disabled_legacy_completion_keeps_the_layer_one_locking_path(self):
        with open(os.path.join(ROOT, "libs", "vkd3d", "framepacer", "command_queue.h"),
                  encoding="utf-8") as stream:
            command_queue = stream.read()
        completion = command_queue[command_queue.index("void notifyVulkanGpuExecutionEnd"):
                                  command_queue.index("void notifySubmitFailed")]
        disabled = completion[completion.index("if (!telemetryEnabled)"):
                              completion.index("SubmitTelemetryMetadata capturedMetadata")]
        self.assertIn("m_simulationLedger.ownsSubmit", disabled)
        self.assertIn("else\n                    completeLegacy();", disabled)
        self.assertLess(disabled.index("completeLegacy();"), disabled.index("return;"))

    def test_drops_are_reported_and_refused(self):
        result = run_capture([run_record(), summary(0, dropped=7)])
        self.assertEqual(result.returncode, 0)
        self.assertIn("dropped records: 7 (capture incomplete)", result.stdout)
        self.assertIn("causal conclusion: REFUSED (records were dropped", result.stdout)

    def test_uncertainty_overlap_is_indeterminate(self):
        render = event("SUBMIT", 1, gpu_bottom_host_ns=500,
                       calibration_max_deviation_ns=20)
        oob = event("SUBMIT", 2, gpu_bottom_host_ns=510,
                    calibration_max_deviation_ns=20, queue_role="OOB_RENDER",
                    capture_class="UNCAPTURED")
        frontier = event("GPU_FRONTIER", 3, gpu_completion_host_ns=510,
                         publication_cpu_ns=515)
        result = run_capture([run_record(), render, oob, frontier, summary(3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OOB-tail-proxy-after-render=INDETERMINATE", result.stdout)
        self.assertIn("timestamp uncertainty overlaps claimed ordering", result.stdout)

    def test_center_selected_tail_cannot_hide_a_later_render_envelope(self):
        render_center_tail = event("SUBMIT", 1, gpu_bottom_host_ns=100,
                                   calibration_max_deviation_ns=1)
        render_uncertain = event("SUBMIT", 2, gpu_bottom_host_ns=90,
                                 calibration_max_deviation_ns=30)
        oob = event("SUBMIT", 3, gpu_bottom_host_ns=105,
                    calibration_max_deviation_ns=1, queue_role="OOB_RENDER",
                    capture_class="UNCAPTURED")
        result = run_capture([run_record(), render_center_tail, render_uncertain,
                              oob, summary(3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OOB-tail-proxy-after-render=INDETERMINATE", result.stdout)

    def test_separated_tail_envelopes_support_ordering(self):
        render = event("SUBMIT", 1, gpu_bottom_host_ns=100,
                       calibration_max_deviation_ns=5)
        oob = event("SUBMIT", 2, gpu_bottom_host_ns=130,
                    calibration_max_deviation_ns=5, queue_role="OOB_RENDER",
                    capture_class="UNCAPTURED")
        frontier = event("GPU_FRONTIER", 3, gpu_completion_host_ns=130,
                         publication_cpu_ns=140)
        result = run_capture([run_record(), render, oob, frontier, summary(3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OOB-tail-proxy-after-render=YES", result.stdout)
        self.assertIn("gpuFinished-publication-after-OOB-tail-proxy=YES", result.stdout)

    def test_full_width_calibration_deviation_is_not_narrowed(self):
        deviation = (1 << 63) + 17
        render = event("SUBMIT", 1, gpu_bottom_host_ns=100,
                       calibration_max_deviation_ns=deviation)
        oob = event("SUBMIT", 2, gpu_bottom_host_ns=200,
                    calibration_max_deviation_ns=deviation,
                    queue_role="OOB_RENDER", capture_class="UNCAPTURED")
        result = run_capture([run_record(), render, oob, summary(2)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OOB-tail-proxy-after-render=INDETERMINATE", result.stdout)

    def test_valid_complete_capture(self):
        wait = event("PACING", 1, phase="WAIT", gpu_wait_begin_ns=200,
                     gpu_wait_end_ns=300, logical_depth_at_entry=3)
        result = run_capture([run_record(), wait, summary(1)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("dropped records: 0", result.stdout)
        self.assertIn("gpu_wait: median=0.000 ms", result.stdout)
        self.assertIn("present classification: UNKNOWN/NONE", result.stdout)
        self.assertIn("display time and input-to-photon latency: not measured", result.stdout)

    def test_same_local_identity_on_different_devices_does_not_correlate(self):
        render = event("SUBMIT", 1, device_id=1, gpu_bottom_host_ns=500,
                       calibration_max_deviation_ns=1)
        oob = event("SUBMIT", 2, device_id=2, gpu_bottom_host_ns=510,
                    calibration_max_deviation_ns=1, queue_role="OOB_RENDER",
                    capture_class="UNCAPTURED")
        frontier = event("GPU_FRONTIER", 3, device_id=1,
                         gpu_completion_host_ns=500, publication_cpu_ns=515)
        result = run_capture([run_record(), render, oob, frontier, summary(3)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("correlated render/OOB simulations: 0", result.stdout)

    def test_sequence_gap_is_rejected(self):
        wait = event("PACING", 2, phase="WAIT", gpu_wait_begin_ns=200,
                     gpu_wait_end_ns=300, logical_depth_at_entry=3)
        result = run_capture([run_record(), wait, summary(1)])
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected event_sequence 1, got 2", result.stderr)

    def test_calibration_snapshot_uses_existing_serialization(self):
        with open(os.path.join(ROOT, "libs", "vkd3d", "framepacer", "command_queue.h"),
                  encoding="utf-8") as stream:
            command_queue = stream.read()
        with open(os.path.join(ROOT, "libs", "vkd3d", "framepacer", "framepacer.h"),
                  encoding="utf-8") as stream:
            framepacer = stream.read()
        self.assertNotIn("m_calibratedDeviceTimestamps.getCalibration()", command_queue)
        snapshot = framepacer[framepacer.index("getTelemetryCalibrationSnapshot"):
                              framepacer.index("static bool isReflexAccountingState")]
        self.assertLess(snapshot.index("m_progressMutex"), snapshot.index("getCalibration()"))

    def test_present_publication_follows_oob_end(self):
        with open(os.path.join(ROOT, "libs", "vkd3d", "swapchain.c"),
                  encoding="utf-8") as stream:
            source = stream.read()
        start = source.index("static void dxgi_vk_swap_chain_present_iteration")
        end = source.index("static void dxgi_vk_swap_chain_signal_waitable_handle", start)
        iteration = source[start:end]
        release = iteration.index("vkd3d_queue_release")
        marker = iteration.index("VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_END_NV")
        timeline = iteration.index("vkd3d_queue_timeline_trace_register_instantaneous")
        publication = iteration.index("pacer_notify_vk_present")
        self.assertLess(release, marker)
        self.assertLess(marker, publication)
        self.assertLess(timeline, publication)
        oob_interval = iteration[iteration.index(
            "VK_LATENCY_MARKER_OUT_OF_BAND_PRESENT_START_NV"):marker]
        self.assertNotIn("pacer_telemetry_now_ns", oob_interval)
        self.assertNotIn("pacer_notify_vk_present", oob_interval)


if __name__ == "__main__":
    unittest.main()
