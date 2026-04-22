#!/usr/bin/env python3
# Copyright (c) 2025 — milestone-1 bridge only: observe sim → POST /act → print response.
# Does not apply VLA actions to the simulator.

"""
Read one RGB frame and one proprio vector from unitree_sim_isaaclab, build the payload
expected by unifolm-vla/deployment/model_server/run_real_eval_server.py (POST /act),
send to the VLA server, print action shape and first row.

Run from an Isaac Lab / Unitree sim environment (same as sim_main.py), typically:
  ./isaaclab.sh -p ../sim_to_vla_client.py -- --enable_cameras --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint

Adjust paths if your launcher lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Project paths (sibling layout: vla-sim-integration/{sim_to_vla_client.py,unitree_sim_isaaclab/})
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_UNITREE_ROOT = os.path.join(_SCRIPT_DIR, "unitree_sim_isaaclab")
os.environ["PROJECT_ROOT"] = _UNITREE_ROOT
if _UNITREE_ROOT not in sys.path:
    sys.path.insert(0, _UNITREE_ROOT)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _tensor_to_uint8_hw3(rgb_tensor, log: logging.Logger) -> "object":
    """
    run_real_eval_server.check_image_format requires np.ndarray, shape (H, W, 3), dtype uint8.

    TODO(runtime): Confirm Isaac Lab Camera output dtype/range for your sensor config (float32 in
    [0,1] vs [0,255] vs uint8). This conversion is best-effort only.
    """
    import numpy as np

    arr = rgb_tensor.detach().cpu().numpy()
    if arr.ndim != 3:
        log.error(
            "TODO: Expected camera rgb with ndim==3 (H,W,C); got shape %s. "
            "If layout is CHW, transpose before sending.",
            arr.shape,
        )
        raise ValueError(f"full_image shape not HWC: {arr.shape}")
    if arr.shape[-1] != 3:
        log.error("TODO: full_image must have 3 channels last; got %s", arr.shape)
        raise ValueError(f"full_image channels: {arr.shape}")

    if arr.dtype == np.uint8:
        log.debug("Camera array already uint8, shape=%s", arr.shape)
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        mx = float(arr.max()) if arr.size else 0.0
        if mx <= 1.0 + 1e-5:
            out = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            log.debug("Converted float [0,1] image to uint8, shape=%s", out.shape)
            return out
        out = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
        log.debug("Converted float image with max=%s to uint8, shape=%s", mx, out.shape)
        return out

    log.warning(
        "TODO: Unexpected camera dtype %s; casting with astype(uint8) may be wrong.",
        arr.dtype,
    )
    return arr.astype(np.uint8)


def _build_state_vector(env, log: logging.Logger) -> "object":
    """
    run_real_eval_server stacks observation[\"state\"] from each time step and normalizes with
    checkpoint norm_stats.

    TODO(integration): The training checkpoint defines PROPRIO_DIM and ordering. This client
    concatenates the MDP helper outputs (body 87 + gripper positions 2) = 89 floats only as a
    concrete placeholder so the HTTP round-trip can be tested. Do not assume this matches
    unifolm-vla training data without verifying norm_stats / dataset schema.

    TODO(server): run_real_eval_server inherits ACTION_DIM/PROPRIO_DIM from constants.py based on
    sys.argv (e.g. \"joint\" in argv → G1). Align server launch command with your checkpoint.
    """
    import numpy as np
    import torch
    from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
    from tasks.common_observations.gripper_state import get_robot_gipper_joint_states

    boy = get_robot_boy_joint_states(env, enable_dds=False)
    grip = get_robot_gipper_joint_states(env, enable_dds=False)
    if not isinstance(boy, torch.Tensor) or not isinstance(grip, torch.Tensor):
        raise TypeError("Expected torch.Tensor joint state tensors from observation helpers.")

    boy_np = boy[0].detach().cpu().numpy().astype(np.float32)
    grip_np = grip[0].detach().cpu().numpy().astype(np.float32)
    state = np.concatenate([boy_np, grip_np], axis=0)
    log.info(
        "Built placeholder state vector: body helper shape %s + gripper pos %s → state shape %s",
        boy_np.shape,
        grip_np.shape,
        state.shape,
    )
    log.debug("state min/max: %s / %s", float(state.min()), float(state.max()))
    return state


def _maybe_wrist_images(env, log: logging.Logger) -> dict:
    """Optional keys: any name containing substring 'wrist' (run_real_eval_server.py)."""
    out = {}
    for key in ("left_wrist_camera", "right_wrist_camera"):
        if key not in env.scene.keys():
            log.debug("Scene has no sensor key %r; skipping wrist image.", key)
            continue
        try:
            rgb = env.scene[key].data.output["rgb"][0]
            out[key] = _tensor_to_uint8_hw3(rgb, log)
            log.info("Wrist image %r shape=%s dtype=%s", key, out[key].shape, out[key].dtype)
        except Exception as e:
            log.warning("TODO: Failed to read wrist camera %r: %s", key, e)
    return out


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Milestone 1: sim observation → POST /act → print response.")
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-PickPlace-RedBlock-G129-Dex1-Joint",
        help="Registered gym task id (default matches unitree_sim_isaaclab/tasks/g1_tasks/.../__init__.py).",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the red block",
        help="Text stored in observations[0][\"instruction\"] (server reads this field only from index 0).",
    )
    parser.add_argument(
        "--act-url",
        type=str,
        default="http://127.0.0.1:8777/act",
        help="Full URL for POST /act (run_real_eval_server default port 8777).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging.")
    AppLauncher.add_app_launcher_args(parser)

    args_cli = parser.parse_args()
    _setup_logging(args_cli.verbose)
    log = logging.getLogger("sim_to_vla_client")

    if getattr(args_cli, "no_render", False):
        os.environ["LIVESTREAM"] = str(getattr(args_cli, "livestream_type", 2))
        os.environ["PUBLIC_IP"] = getattr(args_cli, "public_ip", "127.0.0.1")
    else:
        os.environ["LIVESTREAM"] = "0"

    # Isaac / sim imports (match sim_main.py ordering)
    import pinocchio  # noqa: F401

    log.info("Starting Isaac Lab AppLauncher (task=%s).", args_cli.task)
    simulation_app = None
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import tasks  # noqa: F401 — registers gym environments
    import requests
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env = None
    try:
        log.info("Loading env cfg via parse_env_cfg(%r).", args_cli.task)
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)
        env.sim.reset()
        env.reset()
        log.info("Environment reset complete.")

        # Refresh sensors so rgb buffers may be populated (same idea as camera_state.py).
        dt = float(getattr(env, "physics_dt", getattr(env, "step_dt", 0.02)))
        sensors = getattr(env.scene, "sensors", None) or {}
        for sensor in sensors.values():
            try:
                sensor.update(dt, force_recompute=False)
            except Exception as ex:
                log.debug("sensor.update skipped: %s", ex)

        if "front_camera" not in env.scene.keys():
            log.error(
                "TODO: Scene has no 'front_camera'. Enable cameras (e.g. --enable_cameras) or pick a task with CameraPresets."
            )
            return 2

        rgb_t = env.scene["front_camera"].data.output["rgb"][0]
        full_image = _tensor_to_uint8_hw3(rgb_t, log)
        log.info("full_image shape=%s dtype=%s", full_image.shape, full_image.dtype)

        state = _build_state_vector(env, log)

        observation = {
            "full_image": full_image,
            "instruction": args_cli.instruction,
            "state": state,
        }
        observation.update(_maybe_wrist_images(env, log))

        payload = {"observations": [observation]}
        log.debug("Payload keys: observations[0] = %s", list(observation.keys()))

        try:
            import json_numpy

            json_numpy.patch()
            log.debug("json_numpy.patch() applied for numpy serialization.")
        except ImportError:
            log.error(
                "Install json_numpy on the client (same as run_real_eval_server) so numpy arrays serialize in JSON."
            )
            return 3

        body = json.dumps(payload)
        log.info("POST %s (body bytes=%d)", args_cli.act_url, len(body.encode("utf-8")))
        log.debug("instruction=%r", args_cli.instruction)

        resp = requests.post(
            args_cli.act_url,
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        log.info("Response HTTP %s", resp.status_code)
        if resp.status_code != 200:
            log.error("Response text (truncated): %s", resp.text[:2000])
            return 4

        text = resp.text.strip()
        if text == "error" or text.startswith('"error"'):
            log.error("Server returned string error (see run_real_eval_server get_server_action except path).")
            return 5

        try:
            data = resp.json()
        except json.JSONDecodeError:
            log.error("TODO: Non-JSON success body; raw (truncated): %s", text[:2000])
            return 6

        import numpy as np

        arr = np.asarray(data, dtype=np.float64)
        log.info("Returned action array shape=%s dtype=%s", arr.shape, arr.dtype)
        if arr.ndim == 0:
            log.error("TODO: Unexpected scalar response: %s", data)
            return 7
        row0 = arr[0] if arr.ndim >= 2 else arr
        log.info("First action row (shape=%s): %s", getattr(row0, "shape", ()), row0)
        print("action_shape:", arr.shape)
        print("first_action_row:", row0)
        return 0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as ex:
                log.debug("env.close(): %s", ex)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception as ex:
                log.debug("simulation_app.close(): %s", ex)


if __name__ == "__main__":
    raise SystemExit(main())
