import gymnasium as gym
import time
import numpy as np
from collections import deque
import random


# we want to normalize the pixel observations by 255
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # normalize observation by 255
        return obs / 255.0


# clipping the rewards between 0 and 1
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # clip the rewards between 0 and 1
        return np.clip(reward, 0, 1)


# prevent the slider from moving to the left
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if action == 3:
            return random.choice([0, 1, 2])
        return action


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

    # chaining these environments together
    wrapped_env2 = ObservationWrapper(RewardWrapper(ActionWrapper(env)))

    print("=========== \n\n")
    print(f"Orignal Observation space: {env.observation_space}")
    print(f"Original Action space: {env.action_space}")
    print(" ---------- ")
    print(f"New obseveration space is {wrapped_env.observation_space}")
    print(f"new action space is {wrapped_env.observation_space}")
    print("============ \n\n")

    obs, info = env.reset()

    for i in range(1500):
        print("\n\n Starting the original environment\n\n")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(0.01)
        if done:
            obs, info = env.reset()
        print("\n\nOriginal Environment completed\n\n")

    # env.close()

    obs = wrapped_env2.reset()

    for step in range(500):
        print("\n\nStarting the wrapped environmentn\n\n")
        action = wrapped_env2.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env2.step(action)

        # raise a flag (kind of like exceptions here) if the values have not been vectorised properly
        if (obs > 1.0).any() or (obs < 0.0).any():
            print("max and min value of observations out of range")

        # raise a flag if reward has not been clipped
        if reward < 0.0 or reward > 1.0:
            assert False, "reward out of bounds"

        # checking the rendering if the slider moves to the left
        wrapped_env2.render()

        time.sleep(0.001)
        print("\n\nWrapped Environment finished\n\n")

    wrapped_env2.close()


if __name__ == "__main__":
    main()
