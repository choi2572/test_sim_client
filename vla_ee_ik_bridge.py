"""
Decode UnifoLM-VLA ee_action_6d (23) into SE(3) targets and run LeRobot G1_29 Arm IK (Pinocchio/CasADi).

Requires: pinocchio, casadi, meshcat (see unitree_lerobot). Run from repo root; URDF loads with cwd
set to unitree_lerobot/ during solver construction (handled here).

The last 5 components of the 23-vector (gripper means + waist yaw/roll/pitch) are not applied by this
IK: the reduced model locks waist and hands. Only left/right wrist Cartesian + 6D rotation drive IK.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass


def sixd_to_rotation_matrix(sixd: np.ndarray) -> np.ndarray:
    """
    Recover R from the first two columns (each 3-vector), Zhou-style Gram–Schmidt.
    Layout matches unifolm-vla batch_pose17_to_pose23 (orthonormal columns from RPY).
    """
    v = np.asarray(sixd, dtype=np.float64).reshape(6)
    a1 = v[0:3]
    a2 = v[3:6]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-9:
        return np.eye(3)
    b1 = a1 / n1
    c2 = a2 - float(np.dot(b1, a2)) * b1
    n2 = np.linalg.norm(c2)
    if n2 < 1e-9:
        b2 = np.array([0.0, 1.0, 0.0]) if abs(b1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        b2 = b2 - float(np.dot(b1, b2)) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-9)
    else:
        b2 = c2 / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def ee_action23_to_left_right_homogeneous(ee23: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ee_action_6d layout: L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6), tail5(5) — tail ignored for IK here.
    Returns 4x4 homogeneous transforms in the same convention expected by G1_29_ArmIK.solve_ik.
    """
    v = np.asarray(ee23, dtype=np.float64).reshape(23)
    L_xyz = v[0:3]
    L_6d = v[3:9]
    R_xyz = v[9:12]
    R_6d = v[12:18]
    R_L = sixd_to_rotation_matrix(L_6d)
    R_R = sixd_to_rotation_matrix(R_6d)
    T_L = np.eye(4, dtype=np.float64)
    T_L[:3, :3] = R_L
    T_L[:3, 3] = L_xyz
    T_R = np.eye(4, dtype=np.float64)
    T_R[:3, :3] = R_R
    T_R[:3, 3] = R_xyz
    return T_L, T_R


def _lerobot_project_root() -> Path:
    return Path(__file__).resolve().parent / "unitree_lerobot"


def create_g129_ee_ik_solver(Unit_Test: bool = False, Visualization: bool = False) -> Any:
    """
    Build G1_29_ArmIK with working directory set so relative URDF paths resolve.
    """
    lr = _lerobot_project_root()
    if not lr.is_dir():
        raise FileNotFoundError(
            f"unitree_lerobot not found at {lr}. Clone or place the LeRobot package next to this file."
        )
    if str(lr) not in sys.path:
        sys.path.insert(0, str(lr))

    try:
        from unitree_lerobot.eval_robot.robot_control.robot_arm_ik import G1_29_ArmIK
    except ImportError as e:
        raise ImportError(
            "Failed to import G1_29_ArmIK. Install: pip install pinocchio casadi meshcat "
            "(and LeRobot deps). Original error: "
            f"{e}"
        ) from e

    old = os.getcwd()
    try:
        os.chdir(lr)
        return G1_29_ArmIK(Unit_Test=Unit_Test, Visualization=Visualization)
    finally:
        os.chdir(old)


def solve_arm_ik_from_ee_action23(
    solver: Any,
    ee23: np.ndarray,
    arm14_current_motor_order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    arm14_current_motor_order: same layout as motor_q[15:29] (left arm 7 then right arm 7, G1 motor index order).
    Returns (q14_solution, tau_ff) from G1_29_ArmIK.solve_ik.
    """
    T_L, T_R = ee_action23_to_left_right_homogeneous(ee23)
    q_cur = np.asarray(arm14_current_motor_order, dtype=np.float64).reshape(14)
    return solver.solve_ik(T_L, T_R, q_cur, None)
