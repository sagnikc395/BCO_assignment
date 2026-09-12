import gymnasium as gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import time


class OldGymAPI(gym.Wrapper):
    """baselines' SubprocVecEnv worker expects the legacy gym API:
    reset() -> obs and step() -> (obs, reward, done, info).
    gymnasium returns (obs, info) and a 5-tuple, so translate back."""

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info


def make_env():
    return OldGymAPI(gym.make("BreakoutNoFrameskip-v4", render_mode="human"))


# on macOS multiprocessing uses "spawn", so the child re-imports this file;
# everything that starts processes has to live behind the main guard
if __name__ == "__main__":
    num_envs = 3
    envs = [make_env for _ in range(num_envs)]

    # vec envs
    envs = SubprocVecEnv(envs)

    # get initial state
    init_obs = envs.reset()

    # get a list of observations corresponding to parallel environments
    print(f"number of envs: {len(init_obs)}")

    # check out the obs
    one_obs = init_obs[0]
    print(f"shape of one env: {one_obs.shape}")

    # prepare a list of actions and apply them to env
    for i in range(1000):
        actions = [envs.action_space.sample() for _ in range(num_envs)]
        envs.step(actions)
        # render_mode="human" means each subprocess draws its own window on step;
        # baselines' VecEnv.render() is just a warning, so don't call it
        time.sleep(0.001)

    envs.close()
