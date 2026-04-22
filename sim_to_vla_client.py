#!/usr/bin/env python3
"""
Minimal G1 bridge: read Isaac Lab shared-memory camera + robot state, POST /act, print action.

Does not start the simulator or VLA server. Does not apply actions back to the sim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

_REPO = Path(__file__).resolve().parent
_ISAAC_ROOT = _REPO / "unitree_sim_isaaclab"
if str(_ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAAC_ROOT))

from dds.sharedmemorymanager import SharedMemoryManager  # noqa: E402
from tools.shared_memory_utils import MultiImageReader  # noqa: E402

try:
    import json_numpy

    json_numpy.patch()
except ImportError as e:
    raise SystemExit(
        "json_numpy is required to serialize numpy images/state in the same format "
        "as run_real_eval_server.py expects. Install it in this environment.\n"
        f"Original error: {e}"
    ) from e

VLA_ACT_URL_DEFAULT = "http://127.0.0.1:8777/act"
ROBOT_STATE_SHM_NAME = "isaac_robot_state"


def _bgr_to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.asarray(img, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 image, got shape={img.shape}, dtype={img.dtype}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_observation(
    *,
    instruction: str,
    head_bgr: np.ndarray,
    left_bgr: np.ndarray | None,
    right_bgr: np.ndarray | None,
    state_vector: np.ndarray,
) -> dict:
    """
    One element of payload['observations'] with exact keys expected by get_server_action.
    """
    obs: dict = {
        "full_image": _bgr_to_rgb_u8(head_bgr),
        "instruction": instruction,
        "state": state_vector,
    }
    if left_bgr is not None:
        obs["left_wrist"] = _bgr_to_rgb_u8(left_bgr)
    if right_bgr is not None:
        obs["right_wrist"] = _bgr_to_rgb_u8(right_bgr)
    return obs


def build_g1_state_vector(robot: dict) -> tuple[np.ndarray, list[str]]:
    """
    Map isaac_robot_state JSON into a single proprio vector for /act.

    Returns:
        (vector, todo_notes)

    TODO(proprio): Checkpoint norm_stats['proprio'] must match this vector's length and
    semantic ordering (which joints, pos vs vel, IMU inclusion). This client only forwards
    G1 body joint positions as published by g1_29dof_state → G1RobotDDS.write_robot_state
    (29 floats). If your unnorm_key expects a different PROPRIO_DIM (e.g. 16 or 23), replace
    this function accordingly — do not assume without matching the training config.
    """
    todos: list[str] = []
    pos = robot.get("joint_positions")
    if pos is None:
        raise KeyError("isaac_robot_state missing 'joint_positions'")
    vec = np.asarray(pos, dtype=np.float32).reshape(-1)
    todos.append(
        "Verify len(state) matches vla.norm_stats[<unnorm_key>]['proprio'] for your checkpoint; "
        "currently using joint_positions only (29-D from sim G1 publish path)."
    )
    return vec, todos


def read_one_frame(
    reader: MultiImageReader,
    state_shm: SharedMemoryManager,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    head = reader.read_single_image("head")
    if head is None:
        raise RuntimeError(
            "No head image from shared memory (isaac_head_image_shm). "
            "Is the simulator running and publishing cameras?"
        )
    left = reader.read_single_image("left")
    right = reader.read_single_image("right")
    robot = state_shm.read_data()
    if robot is None:
        raise RuntimeError(
            f"No data in shared memory {ROBOT_STATE_SHM_NAME}. "
            "Is the G1 sim writing robot state?"
        )
    return head, left, right, robot


def post_act(url: str, observations: list[dict], timeout: float) -> np.ndarray:
    body = json.dumps({"observations": observations})
    r = requests.post(url, data=body.encode("utf-8"), headers={"Content-Type": "application/json"}, timeout=timeout)
    r.raise_for_status()
    text = r.text.strip()
    if text == "error":
        raise RuntimeError("Server returned error string; check VLA server logs for traceback.")
    out = json.loads(text)
    if isinstance(out, np.ndarray):
        return out
    return np.asarray(out)


def main() -> None:
    p = argparse.ArgumentParser(description="G1-only: SHM observation → POST /act → print action.")
    p.add_argument("--url", default=VLA_ACT_URL_DEFAULT, help="VLA server /act URL")
    p.add_argument(
        "--instruction",
        default="complete the task.",
        help="Language instruction sent as observations[0]['instruction']",
    )
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument(
        "--loop",
        action="store_true",
        help="Repeat read → POST → print until Ctrl+C (same rate as you invoke; no extra timing)",
    )
    args = p.parse_args()

    reader = MultiImageReader()
    state_shm = SharedMemoryManager(name=ROBOT_STATE_SHM_NAME, size=3072)

    print("Using interfaces: MultiImageReader.read_single_image; SharedMemoryManager.read_data", flush=True)

    while True:
        head, left, right, robot = read_one_frame(reader, state_shm)
        state_vec, todos = build_g1_state_vector(robot)
        for line in todos:
            print(f"TODO: {line}", flush=True)

        obs = build_observation(
            instruction=args.instruction,
            head_bgr=head,
            left_bgr=left,
            right_bgr=right,
            state_vector=state_vec,
        )
        payload = {"observations": [obs]}

        action = post_act(args.url, payload["observations"], args.timeout)
        print(f"action shape: {action.shape}", flush=True)
        if action.ndim >= 2:
            print(f"action first row: {action[0]}", flush=True)
        else:
            print(f"action first row (1D): {action}", flush=True)

        if not args.loop:
            break


if __name__ == "__main__":
    main()
