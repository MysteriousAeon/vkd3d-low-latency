#!/usr/bin/env python3
"""Validate and summarize VKD3D_FG_LATENCY_TELEMETRY JSONL captures."""

import argparse
import collections
import json
import math
import statistics
import sys


SUPPORTED_SCHEMAS = {2, 3}
EVENT_TYPES = {"PACING", "SUBMIT", "PRESENT", "GPU_FRONTIER", "PREDICTION",
               "FIRST_FAILURE"}
FIRST_FAILURE_REASONS = {
    "INVALID_RENDER_START", "INVALID_RENDER_END", "CAPTURE_ACQUIRE_FAILURE",
    "CAPTURE_LEASE_IDENTITY_FAILURE", "SUBMIT_OR_CAPTURE_SEAL_FAILURE",
    "PRESENT_START_FAILURE", "PRESENT_CANCEL_FAILURE", "PRESENT_RECORD_FAILURE",
    "VULKAN_PUBLICATION_FAILURE", "COMPLETION_FAILURE", "ABANDONED_CAPTURED_SUBMIT",
    "BRIDGE_CAPTURE_EXCEPTION", "BRIDGE_MARKER_EXCEPTION",
    "INVALID_PRESENT_ATTEMPT_ZERO_TOKEN",
    "INVALID_PRESENT_ATTEMPT_THREAD_MISMATCH", "INVALID_ABORTED_PRESENT_TOKEN",
    "EXPLICIT_FORCE_OR_OTHER",
}
INVALID_RENDER_START_SUBREASONS = {
    "ZERO_EXTERNAL_ID", "NON_MONOTONIC_OR_DUPLICATE", "MAPPING_ABSENT",
    "MAPPING_TARGET_MISSING", "MAPPED_WRONG_EPOCH",
    "MAPPED_SUBMISSIONS_SEALED", "MAPPED_TRACKING_FAILED",
}


def is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def require(record, line, fields):
    for name, kind in fields.items():
        if name not in record:
            raise ValueError(f"line {line}: missing required field {name}")
        value = record[name]
        valid = ((kind is int and is_integer(value)) or
                 (kind is bool and isinstance(value, bool)) or
                 (kind is str and isinstance(value, str)))
        if not valid:
            raise ValueError(f"line {line}: field {name} must be {kind.__name__}")


def validate(records):
    if not records:
        raise ValueError("empty capture")
    if records[0].get("record_type") != "RUN":
        raise ValueError("capture must begin with RUN")
    if records[-1].get("record_type") != "SUMMARY":
        raise ValueError("incomplete capture: terminal SUMMARY is missing")
    if sum(r.get("record_type") == "RUN" for r in records) != 1:
        raise ValueError("capture must contain exactly one RUN")
    if sum(r.get("record_type") == "SUMMARY" for r in records) != 1:
        raise ValueError("capture must contain exactly one terminal SUMMARY")

    run = records[0]
    summary = records[-1]
    require(run, 1, {
        "record_type": str, "schema_version": int, "run_id": str,
        "build_id": str, "process_id": int, "start_cpu_timestamp_ns": int,
        "cpu_clock": str, "wait_latency": int, "ring_capacity": int,
    })
    schema = run["schema_version"]
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported schema_version {schema}")
    if run["cpu_clock"] != "dxvk_high_resolution_clock_ns":
        raise ValueError(f"unsupported cpu_clock {run['cpu_clock']!r}")

    require(summary, len(records), {
        "record_type": str, "schema_version": int, "run_id": str,
        "published_records": int, "dropped_records": int,
        "end_cpu_timestamp_ns": int,
    })
    if summary["schema_version"] != schema:
        raise ValueError("SUMMARY schema_version does not match RUN")
    if summary["run_id"] != run["run_id"]:
        raise ValueError("SUMMARY run_id does not match RUN")
    if summary["end_cpu_timestamp_ns"] < run["start_cpu_timestamp_ns"]:
        raise ValueError("SUMMARY timestamp precedes RUN timestamp")
    if summary["published_records"] < 0 or summary["dropped_records"] < 0:
        raise ValueError("SUMMARY counters must be non-negative")

    common = {
        "record_type": str, "schema_version": int, "event_sequence": int,
        "phase": str, "cpu_timestamp_ns": int, "device_id": int, "epoch_id": int,
        "simulation_id": int, "external_reflex_id": int,
        "capture_generation": int, "flags": int, "queue_role": str,
        "capture_class": str,
    }
    pacing_fields = {
        "WAIT": {"gpu_wait_begin_ns": int, "gpu_wait_end_ns": int,
                 "logical_depth_at_entry": int},
        "DECISION": {"cpu_delay_us": int, "gpu_delay_us": int,
                     "limiter_delay_us": int, "selected_delay_us": int,
                     "prediction_available": bool, "optimized_gpu_time_us": int},
        "SLEEP": {"requested_sleep_us": int, "sleep_begin_ns": int,
                  "sleep_end_ns": int},
    }
    submit_fields = {
        "gpu_bottom_host_ns": int, "calibration_max_deviation_ns": int,
    }
    frontier_fields = {"gpu_completion_host_ns": int, "publication_cpu_ns": int,
                       "published_submit_count": int, "completed_submit_count": int}
    first_failure_fields = {
        "failure_reason": str, "cpu_finished_watermark": int,
        "gpu_finished_watermark": int, "context_id": int,
        "context_value0": int, "context_value1": int, "context_value2": int,
        "context_count0": int, "context_count1": int,
        "reflex_accounting_active": bool, "present_token_provided": bool,
        "caller_token_thread_match": bool,
    }

    expected_sequence = 1
    first_failures = set()
    for line, record in enumerate(records[1:-1], 2):
        record_type = record.get("record_type")
        if record_type not in EVENT_TYPES:
            raise ValueError(f"line {line}: unsupported record_type {record_type!r}")
        require(record, line, common)
        if record["device_id"] <= 0:
            raise ValueError(f"line {line}: device_id must be positive")
        if record["schema_version"] != schema:
            raise ValueError(f"line {line}: schema_version does not match RUN")
        if record["event_sequence"] != expected_sequence:
            raise ValueError(f"line {line}: expected event_sequence "
                             f"{expected_sequence}, got {record['event_sequence']}")
        expected_sequence += 1
        if record_type == "PACING" and record["phase"] in pacing_fields:
            require(record, line, pacing_fields[record["phase"]])
        elif record_type == "SUBMIT":
            require(record, line, submit_fields)
            if record["calibration_max_deviation_ns"] < 0:
                raise ValueError(f"line {line}: calibration deviation is negative")
        elif record_type == "GPU_FRONTIER":
            require(record, line, frontier_fields)
            if (record["published_submit_count"] <= 0 or
                    record["published_submit_count"] != record["completed_submit_count"]):
                raise ValueError(f"line {line}: GPU_FRONTIER submit counts are not truthful")
        elif record_type == "FIRST_FAILURE":
            if schema < 3:
                raise ValueError(f"line {line}: FIRST_FAILURE requires schema_version 3")
            require(record, line, first_failure_fields)
            if record["failure_reason"] not in FIRST_FAILURE_REASONS:
                raise ValueError(f"line {line}: unsupported FIRST_FAILURE reason "
                                 f"{record['failure_reason']!r}")
            if "invalid_render_start_subreason" in record:
                if record["failure_reason"] != "INVALID_RENDER_START":
                    raise ValueError(f"line {line}: invalid_render_start_subreason requires "
                                     "INVALID_RENDER_START")
                subreason = record["invalid_render_start_subreason"]
                if not isinstance(subreason, str):
                    raise ValueError(f"line {line}: field invalid_render_start_subreason "
                                     "must be str")
                if subreason not in INVALID_RENDER_START_SUBREASONS:
                    raise ValueError(f"line {line}: unsupported INVALID_RENDER_START subreason "
                                     f"{subreason!r}")
            if not record["reflex_accounting_active"]:
                raise ValueError(f"line {line}: FIRST_FAILURE was not emitted for active Reflex accounting")
            key = (record["device_id"], record["epoch_id"])
            if key in first_failures:
                raise ValueError(f"line {line}: duplicate FIRST_FAILURE for active epoch")
            first_failures.add(key)

    event_count = len(records) - 2
    if summary["published_records"] != event_count:
        raise ValueError("SUMMARY published_records does not match event count")
    return run, summary


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def describe_ms(label, values_us):
    if not values_us:
        print(f"{label}: unavailable")
        return
    print(f"{label}: median={statistics.median(values_us) / 1000:.3f} ms "
          f"p95={percentile(values_us, .95) / 1000:.3f} ms n={len(values_us)}")


def load(path):
    records = []
    with open(path, "r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"line {number} is not a JSON object")
            records.append(value)
    return records


def tail_envelope(records, time_field, uncertainty_field=None):
    """Return the conservative bounds of a group's latest possible event."""
    def interval(record):
        uncertainty = record[uncertainty_field] if uncertainty_field else 0
        return record[time_field] - uncertainty, record[time_field] + uncertainty

    intervals = [interval(record) for record in records]
    return max(lower for lower, _ in intervals), max(upper for _, upper in intervals)


def compare_envelopes(left, right):
    """Compare aggregate bounds without inventing order inside uncertainty overlap."""
    if left[0] > right[1]:
        return "YES"
    if left[1] < right[0]:
        return "NO"
    return "INDETERMINATE"


def analyze(records, summary):
    counts = collections.Counter(r["record_type"] for r in records)
    print("record counts:")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    dropped = summary["dropped_records"]
    print(f"dropped records: {dropped}" + (" (capture incomplete)" if dropped else ""))

    first_failures = [r for r in records if r.get("record_type") == "FIRST_FAILURE"]
    if first_failures:
        for failure in first_failures:
            print("FIRST_FAILURE: "
                  f"device={failure['device_id']} epoch={failure['epoch_id']} "
                  f"reason={failure['failure_reason']} simulation={failure['simulation_id']} "
                  f"capture={failure['capture_generation']} "
                  f"cpuFinished={failure['cpu_finished_watermark']} "
                  f"gpuFinished={failure['gpu_finished_watermark']}"
                  + (f" subreason={failure['invalid_render_start_subreason']}"
                     if 'invalid_render_start_subreason' in failure else ""))
    else:
        print("FIRST_FAILURE: none")

    waits = [r for r in records if r.get("record_type") == "PACING" and r["phase"] == "WAIT"]
    decisions = [r for r in records if r.get("record_type") == "PACING" and r["phase"] == "DECISION"]
    sleeps = [r for r in records if r.get("record_type") == "PACING" and r["phase"] == "SLEEP"]
    describe_ms("gpu_wait", [(r["gpu_wait_end_ns"] - r["gpu_wait_begin_ns"]) / 1000
                             for r in waits if r["gpu_wait_end_ns"] >= r["gpu_wait_begin_ns"]])
    describe_ms("cpu_delay", [r["cpu_delay_us"] for r in decisions])
    describe_ms("gpu_delay", [r["gpu_delay_us"] for r in decisions])
    describe_ms("limiter_delay", [r["limiter_delay_us"] for r in decisions])
    describe_ms("requested_sleep", [r["requested_sleep_us"] for r in sleeps])
    describe_ms("actual_sleep", [(r["sleep_end_ns"] - r["sleep_begin_ns"]) / 1000
                                 for r in sleeps if r["sleep_end_ns"] >= r["sleep_begin_ns"]])

    depths = collections.Counter(r["logical_depth_at_entry"] for r in waits)
    print("logical frameId - gpuFinished depth distribution: " +
          (", ".join(f"{depth}:{count}" for depth, count in sorted(depths.items())) or "unavailable"))

    completed = [r for r in records if r.get("record_type") == "SUBMIT" and r["flags"] & 2]
    roles = collections.Counter(r["queue_role"] for r in completed)
    print(f"OOB_RENDER submits: {roles['OOB_RENDER']}")
    print(f"OOB_PRESENT submits: {roles['OOB_PRESENT']}")
    describe_ms("optimizedGpuTime", [r["optimized_gpu_time_us"] for r in decisions
                                     if r["prediction_available"]])

    by_simulation = collections.defaultdict(list)
    for record in completed:
        if record["simulation_id"] and record["gpu_bottom_host_ns"]:
            by_simulation[(record["device_id"], record["epoch_id"],
                           record["simulation_id"])].append(record)
    frontiers = collections.defaultdict(list)
    for record in records:
        if record.get("record_type") == "GPU_FRONTIER":
            frontiers[(record["device_id"], record["epoch_id"],
                       record["simulation_id"])].append(record)

    correlated = 0
    indeterminate = 0
    for key, submits in sorted(by_simulation.items()):
        render = [r for r in submits if r["capture_class"] == "RENDER_CAPTURED"]
        oob = [r for r in submits if r["queue_role"] in ("OOB_RENDER", "OOB_PRESENT")]
        if not render or not oob:
            continue
        correlated += 1
        render_tail = tail_envelope(render, "gpu_bottom_host_ns",
                                    "calibration_max_deviation_ns")
        oob_tail = tail_envelope(oob, "gpu_bottom_host_ns",
                                 "calibration_max_deviation_ns")
        oob_after_render = compare_envelopes(oob_tail, render_tail)
        frontier = frontiers.get(key)
        frontier_after_oob = "UNAVAILABLE"
        if frontier:
            frontier_after_oob = compare_envelopes(
                tail_envelope(frontier, "publication_cpu_ns"), oob_tail)
        indeterminate += oob_after_render == "INDETERMINATE" or frontier_after_oob == "INDETERMINATE"
        print(f"simulation device={key[0]} epoch={key[1]} id={key[2]}: "
              f"OOB-tail-proxy-after-render={oob_after_render} "
              f"gpuFinished-publication-after-OOB-tail-proxy={frontier_after_oob}")

    print(f"correlated render/OOB simulations: {correlated}; uncertainty-indeterminate: {indeterminate}")
    print("interpretation: event sequence is record order, not causality; OOB queue role is evidence only; "
          "OOB-tail is a proxy, not proven FG auxiliary work.")

    ambiguous = any(r["queue_role"] == "UNKNOWN" or r["capture_class"] == "AMBIGUOUS"
                    for r in completed)
    reasons = []
    if dropped:
        reasons.append("records were dropped")
    if ambiguous:
        reasons.append("submit classification is missing/ambiguous")
    if indeterminate:
        reasons.append("timestamp uncertainty overlaps claimed ordering")
    if not correlated:
        reasons.append("no render/OOB correlation exists")
    if reasons:
        print("causal conclusion: REFUSED (" + "; ".join(reasons) + ").")
    else:
        print("causal conclusion: REFUSED (ordering does not prove gating or FG auxiliary identity).")
    print("present classification: UNKNOWN/NONE; this capture cannot distinguish real/generated Present.")
    print("display time and input-to-photon latency: not measured.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl")
    args = parser.parse_args()
    try:
        records = load(args.jsonl)
        _, summary = validate(records)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    analyze(records, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
