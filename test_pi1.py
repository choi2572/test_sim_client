import time
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

controller_config = load_composite_controller_config(
    controller="BASIC"
)
# arm controller를 Cartesian pose 제어로
controller_config["body_parts"]["right"]["type"] = "OSC_POSE"
controller_config["body_parts"]["right"]["input_max"] = 1
controller_config["body_parts"]["right"]["input_min"] = -1
controller_config["body_parts"]["right"]["output_max"] = [0.05, 0.05, 0.05, 0.15, 0.15, 0.15]
controller_config["body_parts"]["right"]["output_min"] = [-0.05, -0.05, -0.05, -0.15, -0.15, -0.15]

env = suite.make(
    "Lift",
    robots="Panda",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()
print("action_dim:", env.action_dim)
print("obs keys:", obs.keys())

def step(action, n=1):
    for _ in range(n):
        obs, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
    return obs

def move_towards(target_pos, steps=120, gain=8.0, grip=-1.0):
    global obs
    for _ in range(steps):
        eef = obs["robot0_eef_pos"]
        delta = target_pos - eef
        action = np.zeros(env.action_dim)
        action[:3] = np.clip(delta * gain, -1.0, 1.0)
        action[3:6] = 0.0
        action[-1] = grip
        obs = step(action)

# robosuite Lift는 cube_pos가 보통 obs에 있음
cube_pos = obs["cube_pos"]
print("cube_pos:", cube_pos)

above = cube_pos + np.array([0.0, 0.0, 0.18])
near = cube_pos + np.array([0.0, 0.0, 0.035])
lift = cube_pos + np.array([0.0, 0.0, 0.35])

# 1. 위로 이동, gripper open
move_towards(above, steps=120, grip=-1.0)

# 2. 내려가기
move_towards(near, steps=100, grip=-1.0)

# 3. gripper close
close_action = np.zeros(env.action_dim)
close_action[-1] = 1.0
obs = step(close_action, n=80)

# 4. 들어올리기
move_towards(lift, steps=180, grip=1.0)

print("done")
while True:
    env.render()
    time.sleep(0.03)
