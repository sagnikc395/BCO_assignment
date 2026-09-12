import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="rgb_array")  # other options like "human"
obs_space = env.observation_space
action_space = env.action_space

print(f"The observation space: {obs_space}")
print(f"The action space: {action_space}")

import matplotlib.pyplot as plt

# plt.imshow(env)
# reset env and see the initial observation
obs = env.reset()
print(f"The initial observation is {obs}")

done = False

while not done:
    # sample a random action from the entire action space
    random_action = env.action_space.sample()

    # take the action and get the new observation space
    new_obs, reward, done, truncated, info = env.step(random_action)
    print(f"the new observation is {new_obs}")

env.close()
# render the plot
plt.imshow(env)

# render method
# doesnt work anymore in new versions of openai gymanasium
# env.render(mode="human")
