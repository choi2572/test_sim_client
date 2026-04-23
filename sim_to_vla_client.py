#!/usr/bin/env python3
"""
Minimal G1 bridge: read Isaac Lab shared-memory camera + robot state, POST /act, print action.

Does not start the simulator or VLA server. Optional: --execute_action sends LowCmd; for UnifoLM
ee_action_6d (23-D) use --execute-ee-action-ik (see vla_ee_ik_bridge.py).
"""
from __future__ import annotations

import argparse
import ctypes
import inspect
import json
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

_REPO = Path(__file__).resolve().parent
_ISAAC_ROOT = _REPO / "unitree_sim_isaaclab"
if str(_ISAAC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ISAAC_ROOT))

from tools.shared_memory_utils import SimpleImageHeader, get_shm_name  # noqa: E402

_SHM_ATTACH_WARNED = False


def attach_shared_memory_readonly(name: str) -> shared_memory.SharedMemory:
    """
    Attach to an existing segment created by the simulator.

    Python 3.12+ defaults to track=True; on close/process exit the resource tracker may
    shm_unlink the segment and break the still-running sim. Use track=False when supported.
    """
    global _SHM_ATTACH_WARNED
    sig = inspect.signature(shared_memory.SharedMemory)
    if "track" in sig.parameters:
        return shared_memory.SharedMemory(name=name, track=False)
    if not _SHM_ATTACH_WARNED:
        print(
            "Warning: This Python has no SharedMemory(track=). If /dev/shm segments disappear "
            "after the client exits, use Python 3.12+ or keep the client alive (--loop).",
            flush=True,
        )
        _SHM_ATTACH_WARNED = True
    return shared_memory.SharedMemory(name=name)


class AttachOnlyImageReader:
    """Same decode logic as MultiImageReader.read_single_image, but opens SHM with track=False (3.12+)."""

    def __init__(self) -> None:
        self.last_timestamps: dict[str, int] = {}
        self.buffer: dict[str, np.ndarray] = {}
        self.shms: dict[str, shared_memory.SharedMemory] = {}

    def read_single_image(self, image_name: str) -> Optional[np.ndarray]:
        try:
            shm_name = get_shm_name(image_name)
            if shm_name not in self.shms:
                try:
                    self.shms[shm_name] = attach_shared_memory_readonly(shm_name)
                except FileNotFoundError:
                    return None

            shm = self.shms[shm_name]
            header_size = ctypes.sizeof(SimpleImageHeader)
            header_data = bytes(shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            last_ts = self.last_timestamps.get(image_name, 0)
            if header.timestamp <= last_ts:
                return self.buffer.get(image_name)

            data_start = header_size
            data_end = data_start + header.data_size
            payload = bytes(shm.buf[data_start:data_end])

            if header.encoding == 1:
                encoded = np.frombuffer(payload, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    return None
            else:
                image = np.frombuffer(payload, dtype=np.uint8)
                expected_size = header.height * header.width * header.channels
                if image.size != expected_size:
                    return None
                image = image.reshape(header.height, header.width, header.channels)

            self.buffer[image_name] = image
            self.last_timestamps[image_name] = header.timestamp
            return image
        except Exception:
            return None

    def close(self) -> None:
        for shm in self.shms.values():
            try:
                shm.close()
            except Exception:
                pass
        self.shms.clear()
        self.buffer.clear()
        self.last_timestamps.clear()

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

# G1 arm command path (same as unitree_sim_isaaclab/dds/g1_robot_dds.py subscriber).
# Not used: send_commands_keyboard.py / send_commands_8bit.py → rt/run_command/cmd (wholebody).
LOWCMD_TOPIC = "rt/lowcmd"
LOWCMD_MSG = "unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_"

# Wrist motors in G1_29_JointIndex numbering (see unifolm-vla unitree_deploy arm_indexs.py).
_WRIST_MOTOR_IDX: frozenset[int] = frozenset({19, 20, 21, 26, 27, 28})
_LEG_WEAK_KP: frozenset[int] = frozenset({3, 9})  # ankle pitch — lower kp in g1_arm

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

# Policy /act joint action: 23 floats in ascending motor index order, excluding joints held from sim.
# Held indices (ankles PR+RR, waist roll+pitch): filled from current SHM→motor q, not from server.
G1_ACTION23_HOLD_MOTOR_INDICES: frozenset[int] = frozenset({4, 5, 10, 11, 13, 14})
G1_ACTION23_MOTOR_INDICES_ASC: tuple[int, ...] = tuple(
    i for i in range(29) if i not in G1_ACTION23_HOLD_MOTOR_INDICES
)
assert len(G1_ACTION23_MOTOR_INDICES_ASC) == 23

# Per-motor soft limits (rad); same order as G1_29_JointIndex 0..28 (deploy joint table).
_G1_LIMIT_LO = (
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -2.9671,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52,
    -3.0892,
    -1.5882,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.614429558,
    -1.614429558,
    -3.0892,
    -2.2515,
    -2.618,
    -1.0472,
    -1.972222054,
    -1.614429558,
    -1.614429558,
)
_G1_LIMIT_HI = (
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    0.5236,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52,
    2.6704,
    2.2515,
    2.618,
    2.0944,
    1.972222054,
    1.614429558,
    1.614429558,
    2.6704,
    1.5882,
    2.618,
    2.0944,
    1.972222054,
    1.614429558,
    1.614429558,
)
G1_MOTOR_Q_MIN = np.array(_G1_LIMIT_LO, dtype=np.float64)
G1_MOTOR_Q_MAX = np.array(_G1_LIMIT_HI, dtype=np.float64)

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
        return attach_shared_memory_readonly(name)
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
    task_name: Optional[str] = None,
) -> dict:
    """
    One element of payload['observations'] with exact keys expected by get_server_action.
    """
    obs: dict = {
        "full_image": _bgr_to_rgb_u8(head_bgr),
        "instruction": instruction,
        "state": state_vector,
    }
    if task_name is not None:
        obs["task_name"] = task_name
    if left_bgr is not None:
        obs["left_wrist"] = _bgr_to_rgb_u8(left_bgr)
    if right_bgr is not None:
        obs["right_wrist"] = _bgr_to_rgb_u8(right_bgr)
    return obs


def validate_task_name_in_dataset_statistics(task_name: str, stats_path: Path) -> None:
    """Ensure task_name is a top-level key like run_real_eval_server's vla.norm_stats[task_name]."""
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"dataset_statistics.json not found at {stats_path!s}. "
            "Pass --dataset-statistics-json pointing to the JSON next to your checkpoint run, "
            "or place dataset_statistics.json in the repo root."
        )
    raw = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{stats_path} must be a JSON object with task keys at the top level.")
    if task_name not in raw:
        keys = sorted(raw.keys())
        raise ValueError(
            f"task_name {task_name!r} is not a top-level key in {stats_path}. "
            f"Available keys: {keys}"
        )
    block = raw[task_name]
    if not isinstance(block, dict):
        raise ValueError(f"{stats_path}[{task_name!r}] must be an object with 'action' and 'proprio'.")
    for need in ("action", "proprio"):
        if need not in block:
            raise ValueError(
                f"{stats_path}[{task_name!r}] missing {need!r} (run_real_eval_server expects both)."
            )


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

    ee6_shm = robot.get("ee_state_6d")
    if ee6_shm is not None:
        vec = np.asarray(ee6_shm, dtype=np.float32).reshape(-1)
        if vec.size != 23:
            raise ValueError(
                f"robot['ee_state_6d'] from SHM must be length 23 (got {vec.size}). "
                "Update unitree_sim_isaaclab g1_29dof_state / Isaac USD link names if needed."
            )
        todos.append(
            "Using robot['ee_state_6d'] from Isaac SHM (sim-computed; matches RLDS ee_state_6d layout when FK matches training)."
        )
        return vec, todos

    eeq_shm = robot.get("ee_qpos")
    if eeq_shm is not None:
        p17 = np.asarray(eeq_shm, dtype=np.float64).reshape(-1)
        if p17.size != 17:
            raise ValueError(f"robot['ee_qpos'] from SHM must be length 17 (got {p17.size})")
        vec = pose17_to_pose23(p17)
        todos.append("Using robot['ee_qpos'] from Isaac SHM → pose17_to_pose23.")
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
            "  • Isaac가 SHM에 ee_state_6d/ee_qpos를 쓰도록 하면: 최신 unitree_sim_isaaclab의 g1_29dof_state + "
            "g1_robot_dds.write_robot_state 확장 사용(자동으로 sim_to_vla_client가 읽음).\n"
            "  • 정석(파일): --ee-pose17-json 또는 --state-json (23-D ee_state_6d).\n"
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
    reader: AttachOnlyImageReader,
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


def shm_boy29_to_motor29_q(joint_positions_boy: np.ndarray) -> np.ndarray:
    """SHM joint_positions are boy-reordered; LowCmd uses motor index 0..28 (G1_29_JointIndex)."""
    b = np.asarray(joint_positions_boy, dtype=np.float64).reshape(29)
    m = np.empty(29)
    for motor_i in range(29):
        m[motor_i] = b[_GLOBAL_TO_SHM29[motor_i]]
    return m


def merge_server23_to_motor29(server23: np.ndarray, motor_q_current: np.ndarray) -> np.ndarray:
    """
    Expand 23-D policy output (ascending motor index order, skipping hold indices) to full motor q.

    Indices 4,5,10,11,13,14 keep motor_q_current; others are overwritten from server23 in order
    G1_ACTION23_MOTOR_INDICES_ASC.
    """
    s = np.asarray(server23, dtype=np.float64).reshape(-1)
    if s.size != 23:
        raise ValueError(f"merge_server23_to_motor29: expected 23 targets, got {s.size}")
    m = np.asarray(motor_q_current, dtype=np.float64).reshape(29)
    out = m.copy()
    for k, motor_i in enumerate(G1_ACTION23_MOTOR_INDICES_ASC):
        out[motor_i] = s[k]
    return out


def clip_motor29_to_limits(motor_q: np.ndarray) -> np.ndarray:
    q = np.asarray(motor_q, dtype=np.float64).reshape(29)
    return np.clip(q, G1_MOTOR_Q_MIN, G1_MOTOR_Q_MAX)


def _kp_kd_for_motor(motor_i: int) -> tuple[float, float]:
    if motor_i in _WRIST_MOTOR_IDX:
        return 40.0, 1.5
    if 15 <= motor_i <= 28:
        return 80.0, 3.0
    if motor_i in _LEG_WEAK_KP:
        return 80.0, 3.0
    return 300.0, 3.0


def action_vector_first_timestep(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(-1)
    if a.ndim == 2:
        return a[0].reshape(-1)
    if a.ndim == 3:
        return a[0, 0].reshape(-1)
    raise ValueError(f"Unexpected action shape {a.shape} ndim={a.ndim}")


def extract_arm14_joint_targets(step1d: np.ndarray) -> np.ndarray:
    """
    Arm-only: return 14 target q for motors 15..28 (left arm then right arm, G1 motor order).

    Does NOT support 23-D EE actions (needs IK). See unitree_sim_isaaclab/action_provider/action_provider_dds.py
    (_arm_source_indices: command positions[15:29] → sim arm joints).
    """
    d = int(step1d.size)
    arm_clip = 2.5
    if d == 23:
        raise RuntimeError(
            "execute_action: 23-D joint actions use merge_server23 / build_motor_cmd_q_from_server23 in main; "
            "not arm14 extraction."
        )
    if d == 16:
        raise RuntimeError(
            "execute_action: length-16 joint action layout not mapped to sim arm14; TODO verify training order."
        )
    if d == 14:
        return np.clip(step1d, -arm_clip, arm_clip)
    if d == 29:
        return np.clip(step1d[15:29], -arm_clip, arm_clip)
    raise RuntimeError(
        f"execute_action: unsupported action length {d}. Supported: 14 (arm q), 23 (motor q ascending, merge), "
        "29 (motor q, uses slice [15:29])."
    )


def _limit_arm_delta(arm_tgt: np.ndarray, arm_cur: np.ndarray, max_delta: float) -> np.ndarray:
    d = np.clip(arm_tgt - arm_cur, -max_delta, max_delta)
    return np.clip(arm_cur + d, -2.5, 2.5)


def build_motor_cmd_q_from_server23(
    server23: np.ndarray,
    motor_q_current: np.ndarray,
    max_arm_delta: float,
) -> np.ndarray:
    """Merge 23-D joint action with sim hold axes, clip limits, rate-limit arm slice 15:29."""
    merged = merge_server23_to_motor29(server23, motor_q_current)
    merged = clip_motor29_to_limits(merged)
    arm_cur = motor_q_current[15:29].reshape(14)
    arm_tgt = merged[15:29].reshape(14)
    arm_send = _limit_arm_delta(arm_tgt, arm_cur, max_arm_delta)
    motor_cmd_q = merged.copy()
    motor_cmd_q[15:29] = arm_send
    return motor_cmd_q


def publish_g129_lowcmd(
    *,
    motor_cmd_q: np.ndarray,
    lowcmd_pub: object,
    crc: object,
    log_detail: str,
) -> None:
    """Publish one LowCmd: full motor q[0:29] (length-29 vector, motor index order)."""
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    cmd = np.asarray(motor_cmd_q, dtype=np.float64).reshape(29)
    print(
        f"[execute_action] {log_detail} → ChannelPublisher({LOWCMD_TOPIC!r}, LowCmd_) "
        f"→ sim G1RobotDDS / dds_robot_cmd",
        flush=True,
    )
    print(f"[execute_action] motor_cmd q[0:29]: {np.array2string(cmd, precision=4)}", flush=True)

    msg = unitree_hg_msg_dds__LowCmd_()
    msg.mode_pr = 0
    msg.mode_machine = 0
    n_motors = len(msg.motor_cmd)
    for i in range(n_motors):
        msg.motor_cmd[i].mode = 1
        if i < 29:
            kp, kd = _kp_kd_for_motor(i)
            msg.motor_cmd[i].kp = kp
            msg.motor_cmd[i].kd = kd
            msg.motor_cmd[i].q = float(cmd[i])
        else:
            msg.motor_cmd[i].kp = 20.0
            msg.motor_cmd[i].kd = 1.0
            msg.motor_cmd[i].q = 0.0
        msg.motor_cmd[i].dq = 0.0
        msg.motor_cmd[i].tau = 0.0
    msg.crc = crc.Crc(msg)
    try:
        lowcmd_pub.Write(msg)
        print("[execute_action] publisher.Write(LowCmd_) completed without exception (ok=True)", flush=True)
    except Exception as e:
        print(f"[execute_action] publisher.Write(LowCmd_) failed: {e!r} (ok=False)", flush=True)
        raise


def publish_g129_arm_lowcmd(
    *,
    joint_positions_shm_boy29: np.ndarray,
    arm14_target: np.ndarray,
    max_arm_delta: float,
    lowcmd_pub: object,
    crc: object,
) -> None:
    """
    Publish one LowCmd on rt/lowcmd. Simulator G1RobotDDS.dds_subscriber writes dds_robot_cmd SHM;
    DDSActionProvider.get_action applies arm slots only (legs/waist commanded q[0:15] are not mapped to sim
    joints today — only 15:28 feed the Isaac arm).
    """
    motor_q = shm_boy29_to_motor29_q(joint_positions_shm_boy29)
    arm_cur = motor_q[15:29].copy()
    arm14 = np.asarray(arm14_target, dtype=np.float64).reshape(14)
    arm_send = _limit_arm_delta(arm14, arm_cur, max_arm_delta)
    motor_cmd_q = motor_q.copy()
    motor_cmd_q[15:29] = arm_send

    print(
        f"[execute_action] arm-only: parsed arm14 (motors 15..28, after delta limit): "
        f"{np.array2string(arm_send, precision=4)}",
        flush=True,
    )
    print(
        f"[execute_action] motor q[0:15] hold from current sim q: {np.array2string(motor_cmd_q[:15], precision=4)}",
        flush=True,
    )
    publish_g129_lowcmd(
        motor_cmd_q=motor_cmd_q,
        lowcmd_pub=lowcmd_pub,
        crc=crc,
        log_detail="arm-only LowCmd (legs/waist from current state)",
    )


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
    p.add_argument(
        "--dataset-statistics-json",
        type=Path,
        default=_REPO / "dataset_statistics.json",
        help=(
            "dataset_statistics.json path (checkpoint run dir). Used only to validate --task-name "
            f"against top-level keys (default: {_REPO / 'dataset_statistics.json'})"
        ),
    )
    p.add_argument(
        "--task-name",
        type=str,
        default=None,
        help=(
            "If set, add observations[0]['task_name'] so run_real_eval_server selects "
            "norm_stats[task_name]['action'/'proprio']. Must match a top-level key in "
            "--dataset-statistics-json."
        ),
    )
    p.add_argument(
        "--execute_action",
        action="store_true",
        help=(
            "After /act, publish via DDS rt/lowcmd (LowCmd_). Joint-space: 14 (arm only), "
            "23 (ascending motor order, hold ankles+waist roll/pitch from sim), or 29 (arm slice from full motor q). "
            "For UnifoLM ee_action_6d (23), use --execute-ee-action-ik instead of treating 23 as joint targets."
        ),
    )
    p.add_argument(
        "--execute-ee-action-ik",
        action="store_true",
        help=(
            "With --execute_action and 23-D /act output: decode ee_action_6d → SE(3)×2, run unitree_lerobot "
            "G1_29_ArmIK (Pinocchio/CasADi), then LowCmd. Requires pinocchio, casadi, meshcat. "
            "Waist+gripper components in the 23-vector are not driven by IK (model locks those joints)."
        ),
    )
    p.add_argument(
        "--dds-domain",
        type=int,
        default=1,
        help="ChannelFactoryInitialize(domain_id); match unitree_sim_isaaclab dds_create (default 1)",
    )
    p.add_argument(
        "--execute-max-arm-delta",
        type=float,
        default=0.15,
        help="Max |Δq| rad per arm joint per command (safety)",
    )
    args = p.parse_args()

    if args.task_name is not None:
        try:
            validate_task_name_in_dataset_statistics(args.task_name, args.dataset_statistics_json)
        except (OSError, ValueError) as e:
            raise SystemExit(str(e)) from e

    if args.execute_ee_action_ik and not args.execute_action:
        raise SystemExit("--execute-ee-action-ik requires --execute_action")

    lowcmd_pub = None
    lowcmd_crc = None
    ee_ik_solver = None
    if args.execute_action:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        print(f"[execute_action] ChannelFactoryInitialize({args.dds_domain})", flush=True)
        ChannelFactoryInitialize(args.dds_domain)
        lowcmd_pub = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
        lowcmd_pub.Init()
        lowcmd_crc = CRC()
        print(
            f"[execute_action] publisher ready: topic={LOWCMD_TOPIC!r} msg=LowCmd_ (persistent for this process)",
            flush=True,
        )
        if args.execute_ee_action_ik:
            from vla_ee_ik_bridge import create_g129_ee_ik_solver

            print("[execute_action] Initializing G1_29_ArmIK for ee_action_6d (may take a few seconds)...", flush=True)
            ee_ik_solver = create_g129_ee_ik_solver()

    reader = AttachOnlyImageReader()
    state_shm = open_robot_state_shm_readonly(args.robot_state_shm)

    print(
        "Using: AttachOnlyImageReader (track=False on Python 3.12+); robot state SHM same attach mode",
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
                task_name=args.task_name,
            )
            payload = {"observations": [obs]}

            action = post_act(args.url, payload["observations"], args.timeout)
            print(f"action shape: {action.shape}", flush=True)
            if action.ndim >= 2:
                print(f"action first row: {action[0]}", flush=True)
            else:
                print(f"action first row (1D): {action}", flush=True)

            if args.execute_action:
                assert lowcmd_pub is not None and lowcmd_crc is not None
                step_vec = action_vector_first_timestep(action)
                print(f"[execute_action] first-timestep vector len={step_vec.size}", flush=True)
                motor_q = shm_boy29_to_motor29_q(
                    np.asarray(robot["joint_positions"], dtype=np.float64)
                )
                if step_vec.size == 23:
                    if args.execute_ee_action_ik:
                        from vla_ee_ik_bridge import solve_arm_ik_from_ee_action23

                        assert ee_ik_solver is not None
                        arm_cur = motor_q[15:29].astype(np.float64).reshape(14)
                        sol_q, _tau = solve_arm_ik_from_ee_action23(ee_ik_solver, step_vec, arm_cur)
                        merged = np.asarray(motor_q, dtype=np.float64).reshape(29).copy()
                        arm_send = _limit_arm_delta(
                            sol_q.reshape(14), arm_cur.reshape(14), args.execute_max_arm_delta
                        )
                        merged[15:29] = arm_send
                        merged = clip_motor29_to_limits(merged)
                        publish_g129_lowcmd(
                            motor_cmd_q=merged,
                            lowcmd_pub=lowcmd_pub,
                            crc=lowcmd_crc,
                            log_detail=(
                                "23-D ee_action_6d → G1_29_ArmIK → arm q[15:29]; "
                                "legs/waist from current sim; IK ignores tail5 (gripper+waist in action vec)"
                            ),
                        )
                    else:
                        motor_cmd_q = build_motor_cmd_q_from_server23(
                            step_vec, motor_q, args.execute_max_arm_delta
                        )
                        publish_g129_lowcmd(
                            motor_cmd_q=motor_cmd_q,
                            lowcmd_pub=lowcmd_pub,
                            crc=lowcmd_crc,
                            log_detail=(
                                "23-D joint /act: ascending motors "
                                f"{G1_ACTION23_MOTOR_INDICES_ASC}, hold {sorted(G1_ACTION23_HOLD_MOTOR_INDICES)} from sim"
                            ),
                        )
                elif step_vec.size in (14, 29):
                    arm14 = extract_arm14_joint_targets(step_vec)
                    publish_g129_arm_lowcmd(
                        joint_positions_shm_boy29=np.asarray(robot["joint_positions"], dtype=np.float64),
                        arm14_target=arm14,
                        max_arm_delta=args.execute_max_arm_delta,
                        lowcmd_pub=lowcmd_pub,
                        crc=lowcmd_crc,
                    )
                else:
                    raise RuntimeError(
                        f"--execute_action: action length {step_vec.size} not supported. "
                        "Use 14 (arm), 23 (joint ascending + sim hold), or 29 (full motor, arm-only sim mapping)."
                    )

            if not args.loop:
                break
    finally:
        try:
            state_shm.close()
        except Exception:
            pass
        reader.close()


if __name__ == "__main__":
    main()
