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

# `joint_positions` in SHM follow get_robot_boy_joint_states: pos_buf[k] == full_joint_pos[boy_joint_indices[k]]
# (unitree_sim_isaaclab/tasks/common_observations/g1_29dof_state.py)
G1_BOY_JOINT_INDICES: tuple[int, ...] = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
# G1_29_JointIndex (0..28 body) -> index into 29-vector from SHM
_GLOBAL_TO_SHM29: dict[int, int] = {g: k for k, g in enumerate(G1_BOY_JOINT_INDICES)}

# ---------------------------------------------------------------------------
# EE 6D 체크포인트의 23차원 proprio는 ee_state_6d (손 위치/자세+허리 등)이지 «관절각 23개»가 아님.
# 아래 프리셋은 «차원만 23으로 맞춰 서버가 돌아가게» 하려는 실험용이며 학습 분포와 의미가 다름.
# 대응(프리셋 한 줄 = SHM에서 가져올 G1 글로벌 관절 인덱스, unifolm arm_indexs.G1_29_JointIndex 와 동일 번호):
#   15..21 왼팔 7, 22..28 오른팔 7, 12..14 허리 3, 0,1,2,6,7,8 양쪽 고관절 pitch/roll/yaw 6 → 합 23
# ---------------------------------------------------------------------------
EXPERIMENTAL_MANIP23_GLOBAL_INDICES: tuple[int, ...] = (
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    12,
    13,
    14,
    0,
    1,
    2,
    6,
    7,
    8,
)


def global_indices_to_shm_slots(global_indices: list[int]) -> list[int]:
    out: list[int] = []
    for g in global_indices:
        gi = int(g)
        if gi not in _GLOBAL_TO_SHM29:
            raise ValueError(f"Invalid G1 global joint index {gi} (expected 0..28 in body set)")
        out.append(_GLOBAL_TO_SHM29[gi])
    return out


def gather_shm_joint_positions_by_global_indices(
    joint_positions_29: np.ndarray, global_indices: list[int]
) -> np.ndarray:
    """Pick a subset of body joints: indices are G1_29_JointIndex-style 0..28, not SHM order."""
    v = np.asarray(joint_positions_29, dtype=np.float32).reshape(-1)
    if v.size != 29:
        raise ValueError(f"Expected 29 SHM joint positions, got {v.size}")
    slots = global_indices_to_shm_slots(global_indices)
    return v[slots].copy()


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


def pose17_to_pose23(pose17: np.ndarray) -> np.ndarray:
    """
    Same layout as unifolm-vla/prepare_data/hdf5_to_rlds/rlds_dataset.py::batch_pose17_to_pose23
    for a single timestep.

    pose17: L_xyz(3), L_rpy(3), R_xyz(3), R_rpy(3), waist5(5)  -> total 17
    returns: L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6), waist5(5) -> total 23
    """
    v = np.asarray(pose17, dtype=np.float64).reshape(17)
    L_xyz = v[0:3]
    L_rpy = v[3:6]
    R_xyz = v[6:9]
    R_rpy = v[9:12]
    waist5 = v[12:17]

    roll, pitch, yaw = L_rpy[0], L_rpy[1], L_rpy[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    col1 = np.array([cy * cp, sy * cp, -sp], dtype=np.float64)
    col2 = np.array(
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
        dtype=np.float64,
    )
    L_6d = np.concatenate([col1, col2])

    roll, pitch, yaw = R_rpy[0], R_rpy[1], R_rpy[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    col1 = np.array([cy * cp, sy * cp, -sp], dtype=np.float64)
    col2 = np.array(
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr],
        dtype=np.float64,
    )
    R_6d = np.concatenate([col1, col2])

    return np.concatenate([L_xyz, L_6d, R_xyz, R_6d, waist5]).astype(np.float32)


def _load_json_vector(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "state" in raw:
            raw = raw["state"]
        elif "ee_state_6d" in raw:
            raw = raw["ee_state_6d"]
        elif "ee_qpos" in raw:
            raw = raw["ee_qpos"]
        else:
            raise ValueError(
                f"{path}: expected a list or dict with 'state', 'ee_state_6d', or 'ee_qpos'"
            )
    return np.asarray(raw, dtype=np.float32).reshape(-1)


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


def build_g1_state_vector(
    robot: dict,
    *,
    state_json: Path | None,
    ee_pose17_json: Path | None,
    gather_global_indices_json: Path | None,
    experimental_manip23: bool,
) -> tuple[np.ndarray, list[str]]:
    """
    Build observation['state'] for /act.

    정석: Unitree G1 EE 체크포인트는 23-D **ee_state_6d** (카테시안+6D회전+허리5 등) — 관절 23개가 아님.
    SHM의 29개는 boy reorder된 **관절각**이라, EE 표현과 1:1 대응표로 바꾸는 것은 불가(역기구학/FK 필요).
    """
    todos: list[str] = []

    if state_json is not None:
        vec = _load_json_vector(state_json)
        todos.append(
            f"Using --state-json ({vec.size} floats); must match checkpoint norm_stats proprio length."
        )
        return vec, todos

    if ee_pose17_json is not None:
        p17 = _load_json_vector(ee_pose17_json)
        if p17.size != 17:
            raise ValueError(f"--ee-pose17-json must have 17 floats (got {p17.size})")
        vec = pose17_to_pose23(p17)
        return vec, todos

    pos = robot.get("joint_positions")
    if pos is None:
        raise KeyError("isaac_robot_state missing 'joint_positions'")
    vec = np.asarray(pos, dtype=np.float32).reshape(-1)

    if vec.size == 29:
        if experimental_manip23 and gather_global_indices_json is not None:
            raise ValueError("Use only one of --experimental-manip23-from-shm29 or --gather-global-indices-json")
        if experimental_manip23:
            vec = gather_shm_joint_positions_by_global_indices(vec, list(EXPERIMENTAL_MANIP23_GLOBAL_INDICES))
            todos.append(
                "실험용: 29-D SHM에서 글로벌 관절 인덱스 "
                f"{EXPERIMENTAL_MANIP23_GLOBAL_INDICES} 를 순서대로 23개 뽑음. "
                "이 벡터는 ee_state_6d가 아님 — EE_R6_G1 학습 체크포인트와 의미·분포 불일치, 추론 품질 보장 없음."
            )
            return vec, todos
        if gather_global_indices_json is not None:
            raw = json.loads(gather_global_indices_json.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("--gather-global-indices-json must be a JSON list of integers 0..28")
            gix = [int(x) for x in raw]
            vec = gather_shm_joint_positions_by_global_indices(vec, gix)
            todos.append(
                f"Using --gather-global-indices-json: {len(gix)} joints (G1 global indices {gix}). "
                "차원은 맞출 수 있어도 EE 6D proprio와 같다고 가정하면 안 됨."
            )
            return vec, todos

        raise RuntimeError(
            "SHM joint_positions is 29-D, but typical Unitree G1 *EE 6D* checkpoints expect "
            "23-D ee_state_6d proprio (norm_stats shape (23,)), not raw body joints. "
            "That mismatch causes: ValueError operands (1,29) and (23,). "
            "옵션:\n"
            "  • 정석: --ee-pose17-json 또는 --state-json (23-D ee_state_6d).\n"
            "  • «일단 돌려보기»만: --experimental-manip23-from-shm29 (관절 부분집합 23개, 의미 불일치).\n"
            "  • 직접 고르기: --gather-global-indices-json FILE (JSON 리스트, G1 글로벌 관절 번호 0..28, "
            "순서가 출력 벡터 순서). SHM 슬롯 변환은 boy_joint_indices 기준으로 이 파일에 내장됨.\n"
            "  • 또는 joint-space로 학습된 다른 체크포인트 사용."
        )

    todos.append(
        "Verify len(state) matches vla.norm_stats[<unnorm_key>]['proprio'] for your checkpoint."
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
        raise RuntimeError(
            "VLA /act failed: server returned plain text 'error'. Check run_real_eval_server.py logs for the traceback."
        )
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"VLA /act returned non-JSON body (first 500 chars): {text[:500]!r}"
        ) from e
    # FastAPI often JSON-encodes a Python str return as the JSON string "error".
    if isinstance(out, str) and out == "error":
        raise RuntimeError(
            "VLA /act failed: server returned JSON \"error\" (exception inside get_server_action). "
            "Check the terminal where run_real_eval_server.py is running for the traceback."
        )
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
    p.add_argument(
        "--state-json",
        type=Path,
        default=None,
        help="JSON file: list of floats or {\"state\": [...]} sent as observation['state'] (e.g. 23-D ee_state_6d)",
    )
    p.add_argument(
        "--ee-pose17-json",
        type=Path,
        default=None,
        help="JSON list of 17 floats (ee_qpos layout) converted to 23-D ee_state_6d via pose17_to_pose23",
    )
    p.add_argument(
        "--experimental-manip23-from-shm29",
        action="store_true",
        help=(
            "If SHM has 29 body joints, take a fixed list of 23 global joint angles (arms+waist+hips). "
            "Does NOT equal ee_state_6d; only for smoke-testing the server."
        ),
    )
    p.add_argument(
        "--gather-global-indices-json",
        type=Path,
        default=None,
        help=(
            "JSON list of G1 body joint indices 0..28 (G1_29_JointIndex). "
            "Length should match checkpoint proprio dim (e.g. 23). Order = output state vector order."
        ),
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
            state_vec, todos = build_g1_state_vector(
                robot,
                state_json=args.state_json,
                ee_pose17_json=args.ee_pose17_json,
                gather_global_indices_json=args.gather_global_indices_json,
                experimental_manip23=args.experimental_manip23_from_shm29,
            )
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
