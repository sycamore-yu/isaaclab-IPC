#!/usr/bin/env python3
"""Analyze a completed MARSIM run's saved references on the CPU.

Required run artifacts are metrics.json, run.json, policy/metadata.json,
policy_stats.jsonl and policy/reference-plans.pt. The evaluator's received
solve events define the exact window. Failure events remain failed attempts.
Only load trusted project archives: torch.load unpickles the saved data classes.
It does not construct a frontend, regenerate yaw, or load the native planner.

Exit 0 means analysis completed, including a negative quality result. Inspect
summary.json for reference quality, failed attempts and recorded task outcomes.
Incomplete/corrupt evidence or undefined curve analysis raises an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePath

import isaaclab_ipc.pathcover_validation as validation
from isaaclab_ipc.pathcover_validation import (
    ArchiveIntegrityError,
    PlanCache,
    analyze_archive,
    planning_attempts,
    select_records,
    sha,
    summarize_quality,
)

ARTIFACTS = ("metrics.json", "run.json", "policy/metadata.json", "policy_stats.jsonl", "policy/reference-plans.pt")


def _require(condition, message):
    if not condition:
        raise ArchiveIntegrityError(message)


def _same_recorded_path(first, second):
    """Compare archived paths without resolving today's filesystem/symlinks."""
    first, second = PurePath(first), PurePath(second)
    if first == second:
        return True
    if first.is_absolute() == second.is_absolute():
        return False
    relative, absolute = (second, first) if first.is_absolute() else (first, second)
    return absolute.parts[-len(relative.parts):] == relative.parts


def _recorded_digest(mapping, path):
    matches = [digest for name, digest in mapping.items() if _same_recorded_path(name, path)]
    _require(len(matches) == 1, f"Expected one recorded identity for {path}")
    return matches[0]


def validate_saved_identity(identity, identity_sha, start, end, recorded_run, metadata, artifact_sha256):
    """Check saved identity evidence, never the current installed binaries.

    Campaign-specific policy/domain/budget checks remain the responsibility of
    the captured start/end checks. Their end artifact hashes bind this analysis
    to the files that were checked. A validation-only executable is not a ROS
    process and is not required in run.json's runtime executable list.
    """
    schema_version = identity.get("schema_version")
    _require(schema_version in (2, 3) and identity.get("status") == "identity_captured",
             "Expected a completed schema-2/schema-3 identity capture")
    for phase, boundary in (("start", start), ("end", end)):
        _require(boundary.get("phase") == phase and boundary.get("identity_matches") is True
                 and boundary.get("identity_sha256") == identity_sha,
                 f"Saved {phase} identity check is missing or inconsistent")
    _require(start.get("run") == end.get("run"), "Start/end identity checks refer to different runs")
    evidence = end["run_evidence"]
    for key, name in (("run_json", "run.json"), ("metadata", "policy/metadata.json"), ("metrics", "metrics.json")):
        _require(evidence[key]["sha256"] == artifact_sha256[name], f"{name} changed after the end identity check")
    controller, frontend = identity["controller"], identity["frontend"]
    _require(recorded_run.get("git_revision") == controller["git_head"] == evidence["git_revision"],
             "Recorded controller Git differs from captured identity")
    libraries = recorded_run.get("native_library_sha256", {})
    for origin in (frontend["library"], controller["native"]["library"]):
        _require(_recorded_digest(libraries, origin["path"]) == origin["sha256"],
                 "Recorded native library differs from captured identity")
    _require(_same_recorded_path(metadata["solver"]["native_extension_path"],
                                 controller["native"]["library"]["path"]),
             "Actor's recorded native extension differs from captured identity")
    runtime = identity["executor"]["runtime_binary_sha256"]
    executables = recorded_run.get("executable_sha256", {})
    _require(runtime and len(executables) == len(runtime), "Recorded ROS executable scope differs")
    for name, digest in runtime.items():
        _require(_recorded_digest(executables, name) == digest, f"Recorded ROS executable differs: {name}")
    runtime_libraries = identity["executor"].get("runtime_library_sha256")
    if "runtime_library_sha256" in identity["executor"]:
        recorded_runtime_libraries = recorded_run.get("runtime_library_sha256")
        _require(isinstance(runtime_libraries, dict) and isinstance(recorded_runtime_libraries, dict)
                 and len(recorded_runtime_libraries) == len(runtime_libraries),
                 "Recorded runtime library scope differs")
        for name, digest in runtime_libraries.items():
            _require(_recorded_digest(recorded_runtime_libraries, name) == digest,
                     f"Recorded runtime library differs: {name}")
    checked_sources = 0
    for name, digest in controller["source_file_sha256"].items():
        if name.startswith(("src/", "ros2/", "scripts/")) and PurePath(name).suffix in {
            ".py", ".sh", ".cpp", ".hpp", ".xml", ".yaml", ".yml", ".json"
        }:
            _require(_recorded_digest(recorded_run.get("source_sha256", {}), name) == digest,
                     f"Recorded runtime source differs: {name}")
            checked_sources += 1
    checkpoint = metadata.get("checkpoint_sha256")
    training = {name: origin for name, origin in identity.get("training", {}).items()
                if origin.get("checkpoint", {}).get("sha256") == checkpoint} if checkpoint else {}
    _require(checkpoint is None or training, "Recorded checkpoint is absent from captured training origins")
    return {
        "identity_evidence_matches": True, "recorded_run_name": end.get("run"),
        "identity_schema_version": schema_version,
        "controller_git_head": controller["git_head"],
        "controller_source_sha256": controller["source_sha256"],
        "checked_runtime_source_count": checked_sources,
        "admm_linear_solver": controller.get("admm_linear_solver"),
        "frontend_origin": frontend, "controller_native_origin": controller["native"],
        "training_origins_for_checkpoint": training,
        "configuration": identity.get("configuration"),
        "configuration_manifest": identity.get("configuration_manifest"),
        "p0_reuse": identity.get("p0_reuse"),
        "runtime_binary_sha256": runtime,
        "runtime_library_sha256": runtime_libraries,
        "validation_only_binary_sha256": identity["executor"].get("validation_binary_sha256", {}),
        "scope": "Saved capture/start/end evidence and recorded run artifacts agree. "
                 "No current binary, source tree, checkpoint or device is revalidated. "
                 "Validation-only binaries retain their captured hashes and are not runtime processes. "
                 "Configuration and P0 reuse fields retain captured provenance; "
                 "the configuration manifest is not independently revalidated here.",
    }


def report_markdown(report):
    quality, coverage = report["quality"], report["coverage"]
    attempts = report["planning_attempts"]
    lines = ["# Saved-reference offline validation", "",
             f"Run: `{report['run_directory']}`.", "",
             f"Complete-reference quality: **{quality['reference_quality_passed']}**. "
             "This result is separate from navigation success and measured/MPC constraints.", "",
             f"Matched {coverage['matched_record_count']}/{coverage['evaluator_solve_count']} evaluator solve records; "
             f"excluded {coverage['excluded_count']} other archive records by exact stamp and sequence membership.",
             f"Planning attempts: {attempts['attempt_count']}; returned plans: {attempts['returned_plan_count']}; "
             f"failed attempts without a returned plan: {attempts['failed_attempt_count']}.",
             f"Complete-reference failures: {quality['failure_count']}; "
             f"categories: {quality['failure_category_counts']}.", "",
             "| Raw complete-curve extremum | Value | Sequence |", "|---|---:|---:|"]
    for name, item in {**quality["raw_polynomial_extrema"], **quality["actual_saved_yaw_extrema"]}.items():
        if item is not None:
            lines.append(f"| {name} | {item['value']:.12g} | {item['plan_sequence_id']} |")
    lines.extend(["", "## Scope", "",
                  "All raw signed SFC/axis excesses, fixed/free initial-boundary differences, "
                  "terminal left limits and deployed terminal sampling are retained in references.json.",
                  "Nominal thrust shortfalls and zero-yaw quadrature flags remain separate from physical quality.",
                  "Failed attempts retain their original errors and latency. No trajectory is invented for them.",
                  "Recorded task metrics are included without recomputing collision or measured execution.", ""])
    lines.extend(f"- {item}" for item in report["limits"])
    lines.extend(["", "## Provenance", "",
                  f"Recorded Git: `{report['provenance']['recorded_git_revision']}`.",
                  f"Saved identity evidence matches: {report['provenance']['saved_identity']['identity_evidence_matches']}.",
                  "Input and analysis source SHA256 values are in both JSON reports. "
                  "Without optional identity evidence, only supplied-file provenance is recorded.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed run artifact directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="New report directory; final reports are not overwritten")
    parser.add_argument("--cache-directory", type=Path, help="Optional resumable per-plan cache (default: OUTPUT/plan-cache)")
    parser.add_argument("--expected-git-prefix", help="Optional recorded-run Git expectation, not the current checkout")
    parser.add_argument("--frontend-manifest", type=Path,
                        help="Optional native-binary release manifest with so_sha256, not a configuration manifest")
    parser.add_argument("--identity", type=Path,
                        help="Optional schema-2/schema-3 captured identity; requires both boundary checks")
    parser.add_argument("--identity-start", type=Path)
    parser.add_argument("--identity-end", type=Path)
    args = parser.parse_args()
    identity_paths = (args.identity, args.identity_start, args.identity_end)
    if any(identity_paths) and not all(identity_paths):
        parser.error("--identity, --identity-start and --identity-end are required together")
    folder, output = args.run_dir.resolve(), args.output_dir.resolve()
    missing = [str(folder / name) for name in ARTIFACTS if not (folder / name).is_file()]
    _require(not missing, "Required completed-run evidence is missing: " + ", ".join(missing))
    for name in ("references.json", "summary.json", "summary.md"):
        _require(not (output / name).exists(), f"Preserve existing report: {output / name}")
    started = datetime.now(UTC).isoformat()
    metadata = json.loads((folder / "policy/metadata.json").read_text())
    recorded_run = json.loads((folder / "run.json").read_text())
    task_metrics = json.loads((folder / "metrics.json").read_text())
    rows = [json.loads(line)["data"] for line in (folder / "policy_stats.jsonl").read_text().splitlines() if line.strip()]
    artifacts = {name: sha(folder / name) for name in ARTIFACTS}
    sources = {"pathcover_validation.py": sha(validation.__file__), "analyze_marsim_references.py": sha(__file__)}
    supplied = {}
    for key, path in (("frontend_manifest", args.frontend_manifest), ("identity", args.identity),
                      ("identity_start", args.identity_start), ("identity_end", args.identity_end)):
        if path is not None:
            supplied[key] = {"path": str(path.resolve()), "sha256": sha(path)}
    git_matches = None
    if args.expected_git_prefix:
        git_matches = str(recorded_run.get("git_revision", "")).startswith(args.expected_git_prefix)
        _require(git_matches, "Recorded Git does not match --expected-git-prefix")
    frontend_hashes = [digest for path, digest in recorded_run.get("native_library_sha256", {}).items()
                       if PurePath(path).name.startswith("_pathcover_native")]
    published_frontend, frontend_matches = None, None
    if args.frontend_manifest is not None:
        release = json.loads(args.frontend_manifest.read_text())
        published_frontend = release.get("so_sha256")
        _require(isinstance(published_frontend, str) and published_frontend,
                 "--frontend-manifest requires a native-binary release with so_sha256; "
                 "a configuration manifest is not a binary release")
        frontend_matches = frontend_hashes == [published_frontend]
        _require(frontend_matches, "Recorded frontend differs from supplied release manifest")
    saved_identity = {"identity_evidence_matches": None,
                      "scope": "No capture/start/end evidence supplied; only supplied-file provenance is recorded."}
    if args.identity is not None:
        identity = json.loads(args.identity.read_text())
        saved_identity = validate_saved_identity(
            identity, supplied["identity"]["sha256"], json.loads(args.identity_start.read_text()),
            json.loads(args.identity_end.read_text()), recorded_run, metadata, artifacts)
        if args.frontend_manifest is not None:
            _require(identity["frontend"]["manifest"]["sha256"] == supplied["frontend_manifest"]["sha256"],
                     "Supplied release manifest differs from captured identity")

    # Loading restores saved data-class attributes; no constructor or planner call.
    # Keep Torch out of the numerical module and map every archive tensor to CPU.
    import torch

    archive = torch.load(folder / "policy/reference-plans.pt", map_location="cpu", weights_only=False)
    select_records(archive, rows, metadata)  # Coverage gate precedes cache creation/reuse.
    cache = PlanCache(args.cache_directory or output / "plan-cache", {
        "artifact_sha256": artifacts, "analysis_source_sha256": sources,
        "supplied_identity_artifacts": supplied,
    })
    report = analyze_archive(archive, rows, metadata, cache=cache)
    report.update({
        "schema_version": 1, "run_directory": str(folder),
        "analysis_started_utc": started, "analysis_completed_utc": datetime.now(UTC).isoformat(),
        "planning_attempts": planning_attempts(rows),
        "recorded_task_metrics": task_metrics,
        "provenance": {
            "artifact_sha256": artifacts, "analysis_source_sha256": sources,
            "supplied_identity_artifacts": supplied, "saved_identity": saved_identity,
            "recorded_git_revision": recorded_run.get("git_revision"),
            "expected_git_prefix": args.expected_git_prefix, "git_prefix_matches": git_matches,
            "recorded_frontend_sha256": frontend_hashes,
            "published_frontend_sha256": published_frontend, "frontend_sha256_matches": frontend_matches,
            "recorded_native_library_sha256": recorded_run.get("native_library_sha256"),
            "recorded_executable_sha256": recorded_run.get("executable_sha256"),
            "recorded_runtime_library_sha256": recorded_run.get("runtime_library_sha256"),
            "recorded_source_sha256": recorded_run.get("source_sha256"),
            "recorded_parameters": metadata["parameters"], "recorded_solver": metadata["solver"],
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "command": [sys.executable, *sys.argv], "python_version": sys.version,
            "per_plan_cache": str(cache.directory),
            "computed_this_invocation": cache.computed, "reused_this_invocation": cache.reused,
        },
    })
    report["quality"] = summarize_quality(report)
    output.mkdir(parents=True, exist_ok=True)
    cache.write(output / "references.json", report)
    summary = {key: value for key, value in report.items() if key != "plans"}
    summary["per_plan_report"] = {"path": str(output / "references.json"), "sha256": sha(output / "references.json")}
    cache.write(output / "summary.json", summary)
    (output / "summary.md").write_text(report_markdown(report))
    cache.write(cache.directory / "progress.json", {
        "completed_plans": len(report["plans"]), "total_selected_plans": len(report["plans"]),
        "computed_this_invocation": cache.computed, "reused_this_invocation": cache.reused,
        "all_selected_plans_analyzed": True, "aggregate_available": True,
        "aggregate": str(output / "references.json"), "aggregate_sha256": sha(output / "references.json"),
    })
    print(json.dumps({"output_directory": str(output), "coverage": report["coverage"],
                      "reference_quality_passed": report["quality"]["reference_quality_passed"],
                      "reference_failure_count": report["quality"]["failure_count"],
                      "failed_planning_attempt_count": report["planning_attempts"]["failed_attempt_count"]}, indent=2))


if __name__ == "__main__":
    main()
