#!/usr/bin/env python3
"""
Mock joint command test (no VLA server, no /act).

- 0.1 s 주기로 rt/lowcmd LowCmd_ 발행
- 관측 state(23)는 로그용으로만 고정 벡터 출력; 실제 제어는 SHM 현재 q + 23-D policy 레이아웃 병합
- G1 모터 인덱스 25 (오른팔 elbow) 목표만 -0.5 ~ 0.5 사인으로 흔듦

실행 전: Isaac + G1 DDS가 돌아가고 isaac_robot_state SHM이 있어야 함.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sim_to_vla_client import (  # noqa: E402
    G1_ACTION23_MOTOR_INDICES_ASC,
    LOWCMD_TOPIC,
    ROBOT_STATE_SHM_NAME,
    build_motor_cmd_q_from_server23,
    open_robot_state_shm_readonly,
    publish_g129_lowcmd,
    read_robot_state_payload,
    shm_boy29_to_motor29_q,
)

# G1_29_JointIndex.kRightElbow
MOTOR_RIGHT_ELBOW = 25


def main() -> None:
    p = argparse.ArgumentParser(description="Mock 23-D LowCmd: sweep motor 25 (right elbow) only.")
    p.add_argument("--robot-state-shm", default=ROBOT_STATE_SHM_NAME)
    p.add_argument("--dds-domain", type=int, default=1)
    p.add_argument("--dt", type=float, default=0.1, help="Command period (seconds)")
    p.add_argument("--sweep-period", type=float, default=4.0, help="Sine period for elbow (seconds)")
    p.add_argument("--elbow-amp", type=float, default=0.5, help="|sin| amplitude (rad), range [-amp, amp]")
    p.add_argument(
        "--max-arm-delta",
        type=float,
        default=1.0,
        help="Per-step arm rate limit passed to build_motor_cmd_q_from_server23 (rad)",
    )
    args = p.parse_args()

    if MOTOR_RIGHT_ELBOW not in G1_ACTION23_MOTOR_INDICES_ASC:
        raise SystemExit(f"Motor {MOTOR_RIGHT_ELBOW} not in action23 layout {G1_ACTION23_MOTOR_INDICES_ASC}")
    k_elbow = G1_ACTION23_MOTOR_INDICES_ASC.index(MOTOR_RIGHT_ELBOW)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(args.dds_domain)
    lowcmd_pub = ChannelPublisher(LOWCMD_TOPIC, LowCmd_)
    lowcmd_pub.Init()
    crc = CRC()

    shm = open_robot_state_shm_readonly(args.robot_state_shm)
    print(
        f"[mock] {LOWCMD_TOPIC!r} @ dt={args.dt}s | motor {MOTOR_RIGHT_ELBOW} → slot {k_elbow}/23 | "
        f"sin amp=±{args.elbow_amp} period={args.sweep_period}s | Ctrl+C stop",
        flush=True,
    )

    t0 = time.monotonic()
    while True:
        payload = read_robot_state_payload(shm)
        if payload is None or "joint_positions" not in payload:
            time.sleep(args.dt)
            continue
        boy29 = np.asarray(payload["joint_positions"], dtype=np.float64).reshape(29)
        motor_q = shm_boy29_to_motor29_q(boy29)

        server23 = np.array([motor_q[i] for i in G1_ACTION23_MOTOR_INDICES_ASC], dtype=np.float64)
        t = time.monotonic() - t0
        server23[k_elbow] = float(args.elbow_amp * math.sin(2.0 * math.pi * t / args.sweep_period))

        motor_cmd_q = build_motor_cmd_q_from_server23(
            server23, motor_q, max_arm_delta=args.max_arm_delta
        )

        print(
            f"[mock] elbow_cmd={server23[k_elbow]:+.4f} q25_send={motor_cmd_q[25]:+.4f}",
            flush=True,
        )
        publish_g129_lowcmd(
            motor_cmd_q=motor_cmd_q,
            lowcmd_pub=lowcmd_pub,
            crc=crc,
            log_detail="mock elbow sweep",
        )
        time.sleep(args.dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[mock] stopped.", flush=True)
        sys.exit(0)
