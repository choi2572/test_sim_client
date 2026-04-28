import argparse
import time

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from openpi_client import image_tools
from openpi_client import websocket_client_policy


def make_env():
    controller_config = load_composite_controller_config(controller="BASIC")
    controller_config["body_parts"]["right"]["type"] = "OSC_POSE"
    controller_config["body_parts"]["right"]["input_max"] = 1
    controller_config["body_parts"]["right"]["input_min"] = -1
    controller_config["body_parts"]["right"]["output_max"] = [0.05, 0.05, 0.05, 0.15, 0.15, 0.15]
    controller_config["body_parts"]["right"]["output_min"] = [-0.05, -0.05, -0.05, -0.15, -0.15, -0.15]

    return suite.make(
        "Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=224,
        camera_widths=224,
        control_freq=20,
    )


def get_policy_action(client, obs, prompt):
    agent_img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]

    # OpenPI 쪽 유틸: uint8 + resize 보장
    agent_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(agent_img, 224, 224)
    )
    wrist_img = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_img, 224, 224)
    )

    request = {
        "observation/exterior_image_1_left": agent_img,
        "observation/wrist_image_left": wrist_img,
        "prompt": prompt,
    }

    out = client.infer(request)
    action = np.asarray(out["actions"][0], dtype=np.float32)

    return action


def sanitize_action(action, action_dim, gripper_sign=1.0, scale=1.0):
    """
    pi05_libero action은 보통 7D:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
    robosuite Lift + Panda + OSC_POSE도 action_dim 7 근처라 일단 그대로 매핑.
    """
    a = np.zeros(action_dim, dtype=np.float32)
    n = min(action_dim, action.shape[0])
    a[:n] = action[:n]

    a[:6] *= scale
    a[-1] *= gripper_sign

    return np.clip(a, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the cube")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--gripper-sign", type=float, default=1.0)
    parser.add_argument("--chunk-every", type=int, default=10)
    args = parser.parse_args()

    env = make_env()
    obs = env.reset()

    print("obs keys:", obs.keys())
    print("action_dim:", env.action_dim)

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    action_chunk = None
    chunk_i = 0

    for t in range(args.steps):
        # 매 step마다 inference 때리면 너무 느릴 수 있어서 chunk를 받아서 몇 step 재사용
        if action_chunk is None or chunk_i >= len(action_chunk) or t % args.chunk_every == 0:
            agent_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["agentview_image"], 224, 224)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["robot0_eye_in_hand_image"], 224, 224)
            )

            request = {
                "observation/exterior_image_1_left": agent_img,
                "observation/wrist_image_left": wrist_img,
                "prompt": args.prompt,
            }

            out = client.infer(request)
            action_chunk = np.asarray(out["actions"], dtype=np.float32)
            chunk_i = 0

            print(f"[{t}] got action_chunk:", action_chunk.shape)

        raw_action = action_chunk[chunk_i]
        chunk_i += 1

        action = sanitize_action(
            raw_action,
            env.action_dim,
            gripper_sign=args.gripper_sign,
            scale=args.scale,
        )

        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)

        if done:
            print("done at step", t)
            break

    while True:
        env.render()
        time.sleep(0.03)


if __name__ == "__main__":
    main()
