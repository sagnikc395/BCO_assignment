import gymnasium as gym
import matplotlib.pyplot as plt
import time


env = gym.make("MountainCar-v0", render_mode="human")  # other options like "rgb_array"

# number of steps we run the agent for
num_steps = 1500
# plt.imshow(env)
# reset env and see the initial observation
obs = env.reset()
print(f"The initial observation is {obs}")


print(f"upper bound for env observation {env.observation_space.high}")
print(f"lower bound for env observation {env.observation_space.low}")

# while not done:
#    # sample a random action from the entire action space
#    random_action = env.action_space.sample()

# take the action and get the new observation space
# new_obs, reward, done, truncated, info = env.step(random_action)
# print(f"the new observation is {new_obs}")

for step in range(num_steps):
    # for now take random action
    action = env.action_space.sample()
    # apply the action
    obs, reward, done, truncated, info = env.step(action)

    # render
    env.render()
    # wait a bit before next frame to prevent crazy fast videos
    time.sleep(0.001)
    # if episode is up , then start another one
    if done:
        env.reset()


env.close()
# render the plot
# plt.imshow(env)

# render method
# doesnt work anymore in new versions of openai gymanasium
# env.render(mode="human")
