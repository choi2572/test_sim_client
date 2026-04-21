#!/usr/bin/env python3
# Copyright: integration shim — minimal read (Isaac Lab SHM) → inference POST → print action.
"""
Minimal bridge shim: one RGB frame (shared memory), one robot state (shared memory),
one HTTP POST using the exact JSON keys expected by unitree_deploy's LongConnectionClient.

TODO: deployment/model_server/run_real_eval_server.py exposes POST /act with a different
      JSON schema ({ "observations": [ { "full_image", "instruction", "state", ... } ] }).
      This file does NOT implement /act unless you add a separate adapter process.
TODO: LongConnectionClient expects response {"result": "ok", "action": ...}. The stock
      FastAPI VLA server returns a raw action array (or json_numpy). If your server is
      run_real_eval_server.py, responses will not match this client — add a gateway or
      extend parsing below.
TODO: Proprio "observation.state" here is isaac_robot_state["joint_positions"] only.
      robot_client.py uses real env obs.observation["qpos"]; verify dims/order match your ckpt.
TODO: Whole-body / dds_wholebody command paths are intentionally not implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, MutableMapping

import cv2
import numpy as np
import requests
import torch

# -----------------------------------------------------------------------------
# Repo paths (workspace layout: vla-sim-integration/{unitree_sim_isaaclab, unifolm-vla, ...})
# -----------------------------------------------------------------------------
_SHIM_DIR = Path(__file__).resolve().parent
_ISAACLAB_ROOT = _SHIM_DIR / "unitree_sim_isaaclab"
if not _ISAACLAB_ROOT.is_dir():
    raise RuntimeError(
        f"Expected unitree_sim_isaaclab at {_ISAACLAB_ROOT}. "
        "Run this script from the vla-sim-integration workspace root."
    )
sys.path.insert(0, str(_ISAACLAB_ROOT))

from dds.sharedmemorymanager import SharedMemoryManager  # noqa: E402
from tools.shared_memory_utils import MultiImageReader  # noqa: E402

# Exact copies of robot_client.py (unitree_deploy/scripts/robot_client.py) — do not rename.
ZERO_ACTION: Dict[str, torch.Tensor] = {
    "g1_dex1": torch.zeros(16, dtype=torch.float32),
    "z1_dual_dex1_realsense": torch.zeros(14, dtype=torch.float32),
    "z1_realsense": torch.zeros(7, dtype=torch.float32),
}

CAM_KEY = {
    "g1_dex1": "cam_right_high",
    "z1_dual_dex1_realsense": "cam_high",
    "z1_realsense": "cam_high",
}

# LongConnectionClient (unitree_deploy/unitree_deploy/utils/eval_utils.py) — endpoint path.
PREDICT_ACTION_PATH = "/predict_action"


def populate_queues(
    queues: MutableMapping[str, Deque[torch.Tensor]],
    batch: Dict[str, torch.Tensor],
) -> MutableMapping[str, Deque[torch.Tensor]]:
    """Mirrors unitree_deploy.utils.eval_utils.populate_queues (verbatim behavior)."""
    for key in batch:
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


def read_one_rgb(
    reader: MultiImageReader,
    image_slot: str,
    log: logging.Logger,
) -> np.ndarray:
    """
    Read one frame from Isaac Lab camera SHM (isaac_{slot}_image_shm).

    MultiImageWriter converts RGB tensor → BGR before encoding; decoded image is BGR uint8.
    robot_client.prepare_observation uses cv2.cvtColor(..., cv2.COLOR_BGR2RGB) before torch.
    """
    img = reader.read_single_image(image_slot)
    if img is None:
        raise RuntimeError(
            f"No image from SHM for slot {image_slot!r} (expected name isaac_{image_slot}_image_shm). "
            "Is sim_main running with cameras enabled?"
        )
    log.info("RGB source: MultiImageReader.read_single_image(%r) -> shape=%s dtype=%s", image_slot, img.shape, img.dtype)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb


def read_one_robot_state(
    shm: SharedMemoryManager,
    log: logging.Logger,
) -> np.ndarray:
    """
    Read one dict from isaac_robot_state (JSON via SharedMemoryManager).

    TODO: G1RobotDDS.write_robot_state writes joint_positions from a reordered 29-DOF slice;
          confirm this ordering matches training / VLA norm_stats for your unnorm_key.
    """
    data = shm.read_data()
    if not data:
        raise RuntimeError(
            "isaac_robot_state read_data() returned empty/None. "
            "Is the sim publishing state (g1_29dof_state / DDS bridge active)?"
        )
    log.debug("isaac_robot_state raw keys: %s", sorted(data.keys()))
    if "joint_positions" not in data:
        raise KeyError(
            "isaac_robot_state has no 'joint_positions' key — not guessing other fields. "
            f"Keys present: {list(data.keys())}"
        )
    qpos = np.asarray(data["joint_positions"], dtype=np.float32)
    log.info(
        "State source: isaac_robot_state['joint_positions'] -> shape=%s dtype=%s",
        qpos.shape,
        qpos.dtype,
    )
    return qpos


def build_batch_from_sim(
    rgb_hwc_uint8: np.ndarray,
    qpos: np.ndarray,
    robot_type: str,
    log: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """
    Aligns with robot_client.prepare_observation tensor shapes (CHW float tensor for image).
    CAM_KEY is kept for documentation / parity with robot_client; sim uses explicit image_slot instead.
    """
    if robot_type not in ZERO_ACTION:
        raise KeyError(f"robot_type {robot_type!r} not in ZERO_ACTION keys: {list(ZERO_ACTION.keys())}")
    _ = CAM_KEY[robot_type]  # noqa: F841 — explicit reference so field names stay tied to robot_client.py
    observation: Dict[str, torch.Tensor] = {
        "observation.images.top": torch.from_numpy(rgb_hwc_uint8).permute(2, 0, 1),
        "observation.state": torch.from_numpy(qpos),
        "action": ZERO_ACTION[robot_type],
    }
    log.info(
        "Batch tensors: observation.images.top shape=%s dtype=%s; observation.state shape=%s; action shape=%s",
        observation["observation.images.top"].shape,
        observation["observation.images.top"].dtype,
        observation["observation.state"].shape,
        observation["action"].shape,
    )
    return observation


def build_long_connection_payload(
    language_instruction: str,
    cond_obs_queues: MutableMapping[str, Deque[torch.Tensor]],
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Exact payload keys as unitree_deploy.utils.eval_utils.LongConnectionClient.predict_action.
    """
    payload = {
        "language_instruction": language_instruction,
        "observation.state": torch.stack(list(cond_obs_queues["observation.state"])).tolist(),
        "observation.images.top": torch.stack(list(cond_obs_queues["observation.images.top"])).tolist(),
        "action": torch.stack(list(cond_obs_queues["action"])).tolist(),
    }
    log.info(
        "POST body keys (exact): %s",
        list(payload.keys()),
    )
    log.debug("observation.state stacked shape (list len): %d", len(payload["observation.state"]))
    return payload


def parse_response(resp: requests.Response, log: logging.Logger) -> None:
    log.info("HTTP %s", resp.status_code)
    text_preview = (resp.text or "")[:800]
    log.debug("Response body preview: %s", text_preview)
    try:
        data = resp.json()
    except json.JSONDecodeError:
        log.error("Response is not JSON. Raw (truncated): %s", text_preview)
        raise

    if isinstance(data, dict) and data.get("result") == "ok" and "action" in data:
        action = torch.tensor(data["action"])
        log.info("Parsed gateway-style response: tensor shape=%s dtype=%s", action.shape, action.dtype)
        print(action)
        return

    # TODO: run_real_eval_server.py returns JSONResponse(action ndarray) with no "result" wrapper.
    if isinstance(data, list):
        log.warning(
            "TODO: Response is a bare JSON list — possible /act or raw action array. "
            "Verify server type and schema."
        )
        arr = np.array(data)
        log.info("Coerced to ndarray shape=%s dtype=%s", arr.shape, arr.dtype)
        print(arr)
        return

    log.warning(
        "TODO: Unrecognized response shape (expected gateway dict or list). Keys: %s",
        list(data.keys()) if isinstance(data, dict) else type(data),
    )
    print(json.dumps(data if isinstance(data, (dict, list)) else str(data), indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal sim SHM → VLA HTTP shim (predict_action schema).")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for inference (robot_client.py uses port 8000; run_real_eval_server default is 8777).",
    )
    parser.add_argument(
        "--robot-type",
        default="g1_dex1",
        choices=list(ZERO_ACTION.keys()),
        help="Must match ZERO_ACTION dict in robot_client.py.",
    )
    parser.add_argument(
        "--image-slot",
        default="head",
        choices=["head", "left", "right"],
        help="Which isaac_{slot}_image_shm to read (see tools/shared_memory_utils.get_shm_name).",
    )
    parser.add_argument(
        "--language-instruction",
        default="validation shim instruction",
        help="Passed as language_instruction (same key as eval_utils.LongConnectionClient).",
    )
    parser.add_argument(
        "--observation-horizon",
        type=int,
        default=2,
        help="Deque length for observation.state / observation.images.top (matches robot_client default).",
    )
    parser.add_argument(
        "--action-queue-len",
        type=int,
        default=16,
        help="Deque maxlen for action (matches robot_client cond_obs_queues['action']).",
    )
    parser.add_argument(
        "--robot-state-shm-name",
        default="isaac_robot_state",
        help="Shared memory name written by G1RobotDDS / g1_29dof_state (do not guess).",
    )
    parser.add_argument(
        "--robot-state-shm-size",
        type=int,
        default=3072,
        help="Size passed to SharedMemoryManager (must be >= g1_robot_dds input_size).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="-v INFO (default), -vv DEBUG",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose >= 2 else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s [sim_robot_client] %(message)s",
    )
    log = logging.getLogger("sim_robot_client")

    log.info("Workspace shim dir: %s", _SHIM_DIR)
    log.info("Isaac Lab root on sys.path: %s", _ISAACLAB_ROOT)

    reader = MultiImageReader()
    rgb = read_one_rgb(reader, args.image_slot, log)
    robot_shm = SharedMemoryManager(name=args.robot_state_shm_name, size=args.robot_state_shm_size)
    log.info("Attached robot state SHM name=%s", robot_shm.get_name())
    qpos = read_one_robot_state(robot_shm, log)

    batch = build_batch_from_sim(rgb, qpos, args.robot_type, log)

    cond_obs_queues: Dict[str, Deque[torch.Tensor]] = {
        "observation.images.top": deque(maxlen=args.observation_horizon),
        "observation.state": deque(maxlen=args.observation_horizon),
        "action": deque(maxlen=args.action_queue_len),
    }
    populate_queues(cond_obs_queues, batch)

    payload = build_long_connection_payload(args.language_instruction, cond_obs_queues, log)

    url = f"{args.base_url.rstrip('/')}{PREDICT_ACTION_PATH}"
    log.info("POST %s", url)

    # Single-shot request (LongConnectionClient retries forever; we fail loud for validation).
    resp = requests.post(url, json=payload, timeout=120)
    parse_response(resp, log)


if __name__ == "__main__":
    main()
