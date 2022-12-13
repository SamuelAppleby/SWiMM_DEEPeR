import numpy as np
import csv
import datetime

from gym_underwater.gym_env import UnderwaterEnv

# construct environment
env = UnderwaterEnv()

# path to csv file if using
#csv_path = './test_backward.csv'

# length of dummy run
run_length = 200

# list to store total episodic reward per episode
episode_rewards = [0.0]

# counter variable for keeping track of episode length
ep_len = 0

# decide on max episode length 
max_ep_length = 10

# collect initial observation 
obs = env.reset()

# timer for any testing involving time spent comparisons
begin_time = datetime.datetime.now()

# mock run - instead of querying model for action, randomly sample or use hardcoded action
for i in range(run_length):

    # sample action or set hardcoded action
    # action = env.action_space.sample()
    action = [1.0, 0]

    # step through environment 
    obs, reward, done, info = env.step(action)

    # increment episode length counter
    ep_len += 1

    # print to terminal lots of info for checking correctness of reward function
    print("Step: {}, Reward: {:.2f}, Dist: {:.2f}, AngError: {:.2f}".format(ep_len, reward, info["dist"], info["ang_error"]))
    print("RovPos: [{:.2f}, {:.2f}, {:.2f}], TargPos [{:.2f}, {:.2f}, {:.2f}], RovFwd: [{:.2f}, {:.2f}, {:.2f}], TargFwd [{:.2f}, {:.2f}, {:.2f}]"
        .format(info["rov_pos"][0], info["rov_pos"][1], info["rov_pos"][2], info["targ_pos"][0], info["targ_pos"][1], info["targ_pos"][2], 
        info["rov_fwd"][0], info["rov_fwd"][1], info["rov_fwd"][2], info["targ_fwd"][0], info["targ_fwd"][1], info["targ_fwd"][2]))
    
    # or alternatively log to csv
    # row = [ep_len, reward, info["dist"], info["ang_error"], info["rov_pos"][0], info["rov_pos"][1], info["rov_pos"][2], 
    #         info["targ_pos"][0], info["targ_pos"][1], info["targ_pos"][2], info["rov_fwd"][0], info["rov_fwd"][1], info["rov_fwd"][2], 
    #         info["targ_fwd"][0], info["targ_fwd"][1], info["targ_fwd"][2]]
    # with open(csv_path, 'a') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(row)
    #     csv_file.close()

    # add reward for this step onto the cumulative sum for the current episode
    episode_rewards[-1] += reward

    # if episode termination criteria met, reset
    if done or ep_len >= max_ep_length:
        obs = env.reset()
        print("Episode Terminated. Reward: {:.2f} {} Steps".format(episode_rewards[-1], ep_len))
        if i < run_length:
            episode_rewards.append(0.0)
            ep_len = 0

print("TIME TAKEN: ", datetime.datetime.now() - begin_time)

env.close()
print("Finished Run")
mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)
num_episodes = len(episode_rewards)
print("Number of Episodes: {}".format(num_episodes))
print("Reward per Episode: {}".format(episode_rewards))
print("Average Reward: {}".format(mean_reward))