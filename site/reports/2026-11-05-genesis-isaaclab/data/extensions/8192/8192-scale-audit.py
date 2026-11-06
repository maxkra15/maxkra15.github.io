# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Audit reset/control indexing at environment 8191 using one frozen-adapter scene.

Run in an existing engine environment with the frozen harness on PYTHONPATH:
    python 8192-scale-audit.py --config /path/config.json --output /path/audit.json

This is untimed validation, not a benchmark or a replacement for the small physical
validators. It never accesses native/private model arrays or changes engine code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

NUM_ENVS = 8192
LAST_ENV = NUM_ENVS - 1
STEPS = 16
EXCLUDED_RUNTIME = {"report", "export_figures", "assemble_report", "validate_adapter"}
EXPECTED_HARNESSES = {
    "anymal": "2579e4ca836e956784261e09b0e193d63e40d96742ed5b305a76daf0e33970af",
    "cartpole": "2579e4ca836e956784261e09b0e193d63e40d96742ed5b305a76daf0e33970af",
    "franka": "634c1fcc57ac70c130f95b36899d8f2fa7e1484d1db21ca8bdbab7c3b3669cc4",
}
UPSTREAM = {
    "isaaclab": "efa240ec39794f5d80430cbe38620626954c2c4d",
    "genesis": "969c3bd1449d0f0c79effd7b6a72614ed46ba6e7",
    "newton": "42f7def118c2c4299ce493457bafd4e8a22a696e",
}


def record(result: dict, name: str, passed: bool, **evidence) -> None:
    """Retain an assertion's numerical evidence and stop on a broken contract."""
    result["checks"][name] = {"passed": passed, **evidence}
    if not passed:
        raise AssertionError(name)


def snapshot(adapter, config: dict, result: dict) -> dict:
    """Copy public state and enforce the 8192-row prerequisite for index assertions."""
    import torch

    state = adapter.state()
    shapes = {
        "joint_pos": (NUM_ENVS, len(config["joint_names"])),
        "joint_vel": (NUM_ENVS, len(config["joint_names"])),
        "root_pos": (NUM_ENVS, 3),
        "root_quat": (NUM_ENVS, 4),
        "root_lin_vel": (NUM_ENVS, 3),
        "root_ang_vel": (NUM_ENVS, 3),
    }
    if config["task"] == "franka":
        shapes["ee_pos"] = (NUM_ENVS, 3)
    if set(state) != set(shapes):
        raise AssertionError(f"Unexpected public state fields: {sorted(state)}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[name] or not value.is_cuda:
            raise AssertionError(f"Invalid {name} shape/device: {getattr(value, 'shape', None)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"Non-finite {name}")
    result["state_shapes"] = {name: list(value.shape) for name, value in state.items()}
    return {name: value.clone() for name, value in state.items()}


def byte_changes(before, after) -> dict:
    """Report exact per-row changes, including a changed sign bit on zero."""
    import torch

    changed = (before.contiguous().view(torch.uint8) != after.contiguous().view(torch.uint8)).any(-1)
    ids = changed.nonzero().flatten()
    return {
        "changed_environments": len(ids),
        "first_changed_ids": ids[:16].tolist(),
        "max_abs_delta": (after - before).abs().max().item(),
    }


def replay(adapter, config: dict, result: dict, nominal, action) -> dict:
    """Replay 16 steps after a full public reset, without hidden solver-state writes."""
    adapter.reset(joint_pos=nominal)
    adapter.apply_action(action)
    for _ in range(STEPS):
        adapter.physics_step()
    adapter.synchronize()
    return snapshot(adapter, config, result)


def audit(config: dict, config_sha256: str, output: Path) -> dict:
    """Measure only the final-environment reset/control contract in one scene."""
    if output.exists():
        raise FileExistsError("Use a new output path for each audit attempt; prior evidence is retained")
    result = {
        "schema_version": 1,
        "kind": "batch_scale_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": config_sha256,
        "status": "failed",
        "config": config,
        "task": config.get("task"),
        "backend": config.get("backend"),
        "num_envs": config.get("num_envs"),
        "checks": {},
        "scope": "Untimed final-environment indexing; no repeated FK/mass/dynamics benchmark",
        "audit_protocol": {
            "selected_env_id": LAST_ENV,
            "graph_warmup_steps": STEPS,
            "reset_probe_steps": 1,
            "control_replays": ["nominal", "last-row pulse", "nominal"],
            "steps_per_replay": STEPS,
        },
    }
    adapter = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if config["num_envs"] != NUM_ENVS or config["vision"] != "none" or config["mode"] != "rollout":
            raise ValueError("Expected an 8192-environment, state-only rollout config")
        if config["backend"] not in {"genesis", "isaaclab"} or config["task"] not in EXPECTED_HARNESSES:
            raise ValueError("Expected a main-backend ANYmal/cartpole/Franka config")
        os.environ.setdefault("PYGLET_HEADLESS", "1")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("ISAACSIM_ACCEPT_EULA", "Y")
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        module = importlib.import_module(config["adapter_module"])
        import tasks
        import torch

        harness = Path(module.__file__).resolve().parent
        if Path(tasks.__file__).resolve().parent != harness:
            raise ValueError("Task and adapter imports must come from the same frozen harness")
        actual_hash = hashlib.sha256(
            b"".join(
                path.name.encode() + path.read_bytes()
                for path in sorted(harness.glob("*.py"))
                if path.stem not in EXCLUDED_RUNTIME
            )
        ).hexdigest()
        expected_hash = EXPECTED_HARNESSES[config["task"]]
        provenance = config["provenance"]
        if actual_hash != expected_hash or provenance["harness_sha256"] != expected_hash:
            raise ValueError(f"Frozen runtime fingerprint mismatch: {actual_hash}")
        if provenance["source_revisions"] != UPSTREAM:
            raise ValueError("Declared upstream pins differ from the measured sources")
        result["provenance"] = {
            "harness_sha256": actual_hash,
            "source_revisions": provenance["source_revisions"],
            "adapter_path": module.__file__,
            "task_path": tasks.__file__,
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        adapter = module.Adapter(config)
        result["backend_metadata"] = adapter.metadata
        properties = torch.cuda.get_device_properties(config["device"])
        result["provenance"].update(cuda_device_uuid=str(properties.uuid), cuda_device_name=properties.name)
        packages = {}
        for name in ("genesis", "newton", "isaaclab"):
            package = sys.modules.get(name)
            if package is not None:
                path = Path(package.__file__).resolve()
                revision = subprocess.run(
                    ["git", "-C", str(path.parent), "rev-parse", "HEAD"], capture_output=True, text=True
                )
                packages[name] = {
                    "path": str(path),
                    "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
                }
        result["provenance"]["loaded_packages"] = packages
        nominal = torch.tensor(config["initial_joint_pos"], device=config["device"], dtype=torch.float32).expand(
            NUM_ENVS, -1
        )
        cartpole = config["task"] == "cartpole"
        actions = torch.zeros((NUM_ENVS, 1), device=config["device"]) if cartpole else nominal.clone()
        # Establish actual captured stepping before exercising high-index writes.
        replay(adapter, config, result, nominal, actions)
        diagnostics = getattr(adapter, "diagnostics", None)
        if diagnostics is not None:
            graph_active = diagnostics()["graph_active"]
            result["graph_active_before_index_audit"] = graph_active
            if graph_active is not True:
                raise AssertionError("The high-index audit must exercise an active captured graph")
        # Keep untouched rows observably away from the default pose. Otherwise
        # a zero-force upright cartpole could hide an accidental global reset.
        reset_probe = nominal.clone()
        reset_probe[:, -1] += torch.linspace(0.03, 0.06, NUM_ENVS, device=config["device"])
        adapter.reset(joint_pos=reset_probe)
        adapter.apply_action(actions)
        adapter.physics_step()
        adapter.synchronize()
        before = snapshot(adapter, config, result)
        probe_separation = (before["joint_pos"][:-1, -1] - nominal[:-1, -1]).abs().min().item()
        if probe_separation < 1e-4:
            raise AssertionError("Untouched reset-probe rows must differ observably from nominal")
        selected = torch.tensor([LAST_ENV], device=config["device"], dtype=torch.long)
        marked = nominal[LAST_ENV : LAST_ENV + 1].clone()
        marked[:, -1] += 0.08
        adapter.reset(selected, joint_pos=marked)
        adapter.synchronize()
        after = snapshot(adapter, config, result)
        changes = {name: byte_changes(before[name][:-1], after[name][:-1]) for name in before}
        record(
            result,
            "last_environment_subset_reset_preserves_all_other_state_bits",
            all(value["changed_environments"] == 0 for value in changes.values()),
            reset_env_id=LAST_ENV,
            untouched_environments=LAST_ENV,
            minimum_untouched_last_joint_offset_from_nominal=probe_separation,
            per_field=changes,
        )
        expected_root = torch.tensor([0.0, 0.0, config["initial_height"]], device=config["device"], dtype=torch.float32)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=config["device"])
        joint_error = (after["joint_pos"][LAST_ENV] - marked[0]).abs().max().item()
        root_error = (after["root_pos"][LAST_ENV] - expected_root).abs().max().item()
        quat_error = min(
            (after["root_quat"][LAST_ENV] - identity).abs().max().item(),
            (after["root_quat"][LAST_ENV] + identity).abs().max().item(),
        )
        velocity_error = max(
            after[name][LAST_ENV].abs().max().item() for name in ("joint_vel", "root_lin_vel", "root_ang_vel")
        )
        record(
            result,
            "last_environment_receives_requested_reset",
            joint_error <= 1e-6 and root_error <= 1e-4 and quat_error <= 1e-6 and velocity_error <= 1e-6,
            joint_position_error=joint_error,
            root_position_error_m=root_error,
            quaternion_component_error=quat_error,
            maximum_velocity_error=velocity_error,
            requested_joints=marked[0].tolist(),
            observed_joints=after["joint_pos"][LAST_ENV].tolist(),
            joint_tolerance=1e-6,
            root_tolerance_m=1e-4,
            quaternion_tolerance=1e-6,
            velocity_tolerance=1e-6,
        )
        # A/B/A bounds public-reset replay drift independently of the pulse.
        baseline_before = replay(adapter, config, result, nominal, actions)
        pulsed = actions.clone()
        control_column = 0 if cartpole else len(config["joint_names"]) - 1
        pulse = 10.0 if cartpole else 0.1
        pulsed[LAST_ENV, control_column] += pulse
        response = replay(adapter, config, result, nominal, pulsed)
        baseline_after = replay(adapter, config, result, nominal, actions)
        if diagnostics is not None:
            result["backend_diagnostics_after_index_audit"] = diagnostics()
            if result["backend_diagnostics_after_index_audit"]["graph_active"] is not True:
                raise AssertionError("The captured graph was lost during the reset/control index audit")
        bounds = {}
        for name in baseline_before:
            tolerance = 1e-3 if name in {"joint_vel", "root_lin_vel", "root_ang_vel"} else 1e-4
            low, high = baseline_before[name], baseline_after[name]
            reference = (low + high) * 0.5
            bounds[name] = {
                "baseline_replay_max_abs_drift": (high - low).abs().max().item(),
                "unselected_pulse_max_abs_delta": (response[name][:-1] - reference[:-1]).abs().max().item(),
                "tolerance": tolerance,
            }
        record(
            result,
            "last_environment_control_preserves_other_worlds",
            all(
                item["baseline_replay_max_abs_drift"] <= item["tolerance"]
                and item["unselected_pulse_max_abs_delta"] <= item["tolerance"]
                for item in bounds.values()
            ),
            replay="Nominal / last-environment pulse / nominal, each after full public reset",
            steps_per_replay=STEPS,
            duration_s=STEPS * config["dt"],
            per_field=bounds,
        )
        coordinate = 0 if cartpole else control_column
        reference = (
            baseline_before["joint_pos"][LAST_ENV, coordinate] + baseline_after["joint_pos"][LAST_ENV, coordinate]
        ) * 0.5
        delta = (response["joint_pos"][LAST_ENV, coordinate] - reference).item()
        record(
            result,
            "last_environment_control_reaches_last_actuated_coordinate",
            delta > 1e-3,
            env_id=LAST_ENV,
            joint_name=config["joint_names"][coordinate],
            control_delta=pulse,
            control_units="N" if cartpole else "rad",
            observed_positive_position_response=delta,
            response_units="m" if cartpole else "rad",
            minimum_response=1e-3,
            pulse_joint_position=response["joint_pos"][LAST_ENV, coordinate].item(),
            baseline_joint_position=reference.item(),
        )
        result["status"] = "ok"
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if adapter is not None:
            try:
                adapter.close()
            except Exception as error:
                result.update(status="failed", teardown_error=f"{type(error).__name__}: {error}")
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    result = audit(json.loads(raw), hashlib.sha256(raw).hexdigest(), args.output)
    raise SystemExit(0 if result["status"] == "ok" else 1)
