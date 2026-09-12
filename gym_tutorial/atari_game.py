import gymnasium as gym
import time
import numpy as np
from collections import deque


class ConcatObjs(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, terminated, info

    def _get_ob(self):
        return np.array(self.frames)


def main():
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")

    # wrapping the env, 4 layers
    wrapped_env = ConcatObjs(env, 4)

    print(f"Oriignal Observation space: {env.observation_space}")
    print(f"Original Action space: {env.action_space}")

    print(f"New obseveration space is {wrapped_env.observation_space}")
    print(f"new action space is {wrapped_env.observation_space}")

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


if __name__ == "__main__":
    main()
