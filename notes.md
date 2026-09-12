## OpenAI Gym

- toolkit for RL that provides a wide variety of standarized envs
along with a simple interface for interacting with them.
- basic usage of Gym involves creating an env with `gym.make()`
(resetting it to get an initial state) , then iteratively taking actions with env.step(action) to advance the env
and receive observations, rewards, and a done flag until he episode terminates.
- Each Gym env defines a observation space and a action space to specify the format of inputs and outputs -> eg: an env might have
a discrete action space of size N (n possible actions) and a observation space of certain dimensions, which tells our agent what kind of data
it will receive and what actions it can take.
- Gym by itself doesnt provide the learning algorithm - upto us or an external algorithm like Stable baselines , to implement
an agent that uses the observations and rewards from the environments to learn an optimal policy over time.


### Environments
1. `Env` class
2. this implements a simulator that runs the environment we want to train our agent in.
3. Open AI gym comes packed with a lot of environments, eg: where we can move a car up and down a hill etc.
4. Also provides us the ability to create custom environments as well.

### Interacting with the Environment

- cover functions of the Env class that help the agent interact with the environment
- two such important functions are
  - reset -> this resets the environment to its initial state and returns the observation of the environment corresponding to the initial state
  - step -> takes a action as an input and applies it to the env, which leads to the env transitioning to a new state.

- reset returns 4 things:
  1. observations: the observation of the state of the environment
  2. reward: the reward that we can get from the environment after executing the action that was given as the input to the step function
  3. done: whether the episode has been termianted. if true, we may need to end the simulation or reset the environment to restart the episode.
  4. info: provides additional information depending on the environment, such as the number of lives left, or general information that may be conducive in debugging.

   
### Spaces

- Spaces are DS provided by the Gym library, which describes the valid values of observations and actions in RL environments.
- all of these inherit from the `gym.Space` base class.
- `Box()` space represents an n-dimensional continuous space. This space is bounded, meaning it has specified upper and lower limits for each dimension.
- these bounds define the range of legitimate values that observations can take and can be accessed using the high and low attributes of Box space.
- `Discrete(n)` box describes a discrete space with `[0...n-1]` possible values.

### Wrappers
- Provides us with the functionality to modify various parts of an environment to suit our needs.
- Wrapper class


### 
