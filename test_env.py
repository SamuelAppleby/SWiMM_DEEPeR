import time
from gym_underwater.python_server import PythonServer
from config import *

import numpy as np
from gym_underwater.gym_env import UnderwaterEnv

# construct environment and seed
env = UnderwaterEnv()

print("Doing things for 20 seconds ...")
time.sleep(20)

print("and stopping thread and closing socket")
env.close()

# env.seed(0)
# print("environment ready")

# # how many steps want to test for
# num_test_steps = 10

# # list to store cumulative episodic reward for each episode played out
# episode_rewards = [0.0]

# # counter variable for keeping track of episode length 
# ep_len = 0

# # decide on max episode length 
# max_ep_length = 3000

# # collect initial observation 
# obs = env.reset()

# # mock run where instead of querying model for action decision, randomly sample from action space
# for _ in range(num_test_steps):

#     # sample action
#     action = env.action_space.sample()

#     # step through environment i.e. send action off to be implemented, retrieve next obs, calculate reward, and check if done
#     obs, reward, done, info = env.step(action)

#     # increment episode length counter
#     ep_len += 1

#     # add reward for this step onto the cumulative sum for the current episode
#     episode_rewards[-1] += reward

#     # check episode length
#     if ep_len >= max_ep_length:
#         done = True

#     # periodically print to terminal so can watch episode progression
#     if ep_len % 100 == 0:
#         print("{} steps".format(ep_len))

#     # if episode termination criteria met, finish episode, reset env, and reset log variables
#     if done:
#         obs = env.reset()
#         print("Episode finished. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))
#         episode_rewards.append(0.0)
#         ep_len = 0

# env.close()

# print("Finished!")
# mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)
# num_episodes = len(episode_rewards)
# print("Number of episodes: {}".format(num_episodes))
# print("Average reward: {}".format(mean_reward))

