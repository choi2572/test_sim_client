#!/usr/bin/env python3
"""
Minimal G1 bridge: read Isaac Lab shared-memory camera + robot state, POST /act, print action.

Does not start the simulator or VLA server. Does not apply actions back to the sim.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import requests

_REPO = Path(__file__).resolve().parent
_ISAAC_ROOT = _REPO / "unitree_sim_isaaclab"
if str(_ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAAC_ROOT))

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


def open_robot_state_shm_readonly(name: str) -> shared_memory.SharedMemory:
    """
    Attach to an existing segment only (do not create).

    Creating a fresh segment here would not match the simulator's buffer and yields data_len==0 forever.
    """
    try:
        return shared_memory.SharedMemory(name=name)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Shared memory {name!r} does not exist yet. Start the Isaac Lab sim first so "
            "G1RobotDDS can create it (robot_type g129 or h1_2 with DDS). "
            "If the sim log shows a different segment name, pass --robot-state-shm."
        ) from e


def read_robot_state_payload(shm: shared_memory.SharedMemory) -> dict | None:
    """Same layout as dds.sharedmemorymanager.SharedMemoryManager.read_data (no lock)."""
    try:
        data_len = int.from_bytes(shm.buf[4:8], "little")
        if data_len == 0:
            return None
        json_bytes = bytes(shm.buf[8 : 8 + data_len])
        data = json.loads(json_bytes.decode("utf-8"))
        data["_timestamp"] = int.from_bytes(shm.buf[0:4], "little")
        return data
    except Exception:
        return None


def wait_for_robot_state(
    shm: shared_memory.SharedMemory,
    shm_name: str,
    *,
    timeout_sec: float,
    poll_sec: float,
) -> dict:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        payload = read_robot_state_payload(shm)
        if payload is not None:
            return payload
        time.sleep(poll_sec)
    raise RuntimeError(
        f"Timed out after {timeout_sec}s: no JSON robot state in {shm_name!r} "
        "(length field still 0). The sim must step the env so get_robot_boy_joint_states runs "
        "with DDS g129 and writes via G1RobotDDS.write_robot_state. "
        "If this persists, the writer may be on a different POSIX SHM name than "
        f"{shm_name!r} (check sim logs for the segment name) and pass --robot-state-shm."
    )


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
    state_shm: shared_memory.SharedMemory,
    robot_state_shm_name: str,
    *,
    wait_state_sec: float,
    state_poll_sec: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    head = reader.read_single_image("head")
    if head is None:
        raise RuntimeError(
            "No head image from shared memory (isaac_head_image_shm). "
            "Is the simulator running and publishing cameras?"
        )
    left = reader.read_single_image("left")
    right = reader.read_single_image("right")
    robot = wait_for_robot_state(
        state_shm,
        robot_state_shm_name,
        timeout_sec=wait_state_sec,
        poll_sec=state_poll_sec,
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
        "--robot-state-shm",
        default=ROBOT_STATE_SHM_NAME,
        help="POSIX shared memory name for G1RobotDDS input (default: isaac_robot_state)",
    )
    p.add_argument(
        "--wait-state-sec",
        type=float,
        default=120.0,
        help="Max seconds to wait for first non-empty robot state JSON (sim must step env / write SHM)",
    )
    p.add_argument(
        "--state-poll-sec",
        type=float,
        default=0.05,
        help="Sleep between polls while waiting for robot state",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Repeat read → POST → print until Ctrl+C (same rate as you invoke; no extra timing)",
    )
    args = p.parse_args()

    reader = MultiImageReader()
    state_shm = open_robot_state_shm_readonly(args.robot_state_shm)

    print(
        "Using: MultiImageReader.read_single_image; shared_memory.SharedMemory(attach-only) + SHM read layout",
        flush=True,
    )

    try:
        while True:
            head, left, right, robot = read_one_frame(
                reader,
                state_shm,
                args.robot_state_shm,
                wait_state_sec=args.wait_state_sec,
                state_poll_sec=args.state_poll_sec,
            )
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
    finally:
        state_shm.close()


if __name__ == "__main__":
    main()
