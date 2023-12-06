# generic imports
import argparse
import csv
import os
import sys
import yaml
import gymnasium

# specialist imports
import numpy as np
from stable_baselines3.common import logger
from stable_baselines3.common.utils import set_random_seed, safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_underwater.sim_comms import Protocol

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

# local imports
from stable_baselines3 import SAC
from gym_underwater.utils.utils import make_env
import cmvae_models.cmvae

parser = argparse.ArgumentParser()
parser.add_argument('--host', help='Override the host for network (with port)', default='127.0.0.1:60260', type=str)
parser.add_argument('-tcp', help='Enable tcp', action='store_true')
parser.add_argument('-udp', help='Enable udp', action='store_true')
args = parser.parse_args()

if args.udp and not args.tcp:
    args.protocol = Protocol.UDP
else:
    args.protocol = Protocol.TCP

# Dictionary of available algorithms
ALGOS = {
    'sac': SAC,
}

print("Loading environment configuration ...")
with open(os.path.join(os.pardir, 'configs', 'config.yml'), 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)

# early check on path to trained model if -i arg passed
if env_config['model_path'] != '':
    assert os.path.exists(env_config['model_path']) and os.path.isfile(env_config['model_path']) and env_config['model_path'].endswith('.zip'), \
        'The argument model_path must be a valid path to a .zip file'

# if using pretrained cmvae, create instance of cmvae object and load trained weights from path provided
print("Obs: {}".format(env_config['obs']))
cmvae = None
if env_config['obs'] == 'cmvae':
    print('Loading CMVAE ...')
    cmvae = cmvae_models.cmvae.CmvaeDirect(n_z=10, state_dim=3, res=64, trainable_model=False)  # these args should really be dynamically read in
    cmvae.load_weights(env_config['cmvae_path'])
else:
    print('For inference, must provide a valid cmvae path!')
    quit()

# if seed provided, use it, otherwise use zero
# Note: this stable baselines utility function seeds tensorflow, np.random, and random
set_random_seed(env_config['seed'])

# Set up a reward log if you want one
log_dir = os.path.dirname(env_config['model_path'])
os.environ['OPENAI_LOG_FORMAT'] = 'csv'
os.environ['OPENAI_LOGDIR'] = log_dir
logger.configure()

# Wrap environment with DummyVecEnv to prevent code intended for vectorized envs throwing error
env = DummyVecEnv([make_env(cmvae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_scale'], env_config['debug_logs'], args.protocol, args.host, log_dir, env_config['ep_length_threshold'], seed=env_config.get('seed', 0))])

# load trained model
print("Loading pretrained agent ...")
model = SAC.load(path=env_config['model_path'], env=env)

# Collect first observation
# NB this is calling reset in DummyVecEnv before reset in DonkeyEnv
# DummyVecEnv, like VecEnv, returns list (length 1) of obs, hence why below obs[0] is passed to predict not obs
obs = env.reset()

with open(env_config['reward_log_path'], 'w', newline='') as csv_file:
    best_writer = csv.writer(csv_file)
    best_writer.writerow(['Episode', 'Steps', 'Reward', 'Time', 'Termination'])
    csv_file.close()

episode_times = []

# perform inference for 100 episodes (standard practice)
while len(env.envs[0].episode_lengths) < 5:

    # query model for action decision given obs
    action, _ = model.predict(obs, deterministic=True)

    # step the environment
    obs, reward, terminated, info = env.step(action)

    if terminated:
        # log episodic reward
        print("Episode Reward: {:.2f}".format(info[-1]['episode']['r']))
        print("Episode Length: ", info[-1]['episode']['l'])
        episode_times.append(env.envs[0].episode_times[-1] - env.envs[0].episode_times[-2] if (len(env.envs[0].episode_times) > 1) else env.envs[0].episode_times[-1])
        print("Episode Time: ", episode_times[-1])

        with open(env_config['reward_log_path'], 'a', newline='') as csv_file:
            csv.writer(csv_file).writerow([len(env.envs[0].episode_lengths), info[-1]['episode']['l'], info[-1]['episode']['r'], episode_times[-1], env.envs[0].handler.episode_termination_type])
            csv_file.close()

print("Finished!")

# reset before close
env.reset()

print("Number of episodes: ", len(env.envs[0].episode_lengths))
print("Total reward: ", np.sum(env.envs[0].episode_returns))
print("Average reward: ", safe_mean(env.envs[0].episode_returns))
print("Reward std: ", np.std(env.envs[0].episode_returns))
print("Total time: ", np.sum(episode_times))
print("Average Episode Time: ", safe_mean(episode_times))

env.envs[0].close()
