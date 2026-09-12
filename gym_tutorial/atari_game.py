import gymnasium as gym
import time

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs, info = env.reset()

for i in range(1500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    time.sleep(0.01)
    if done:
        obs, info = env.reset()

env.close()
