# generic imports
import os
import sys
import yaml
import warnings

# specialist imports
import gym
import numpy as np
from stable_baselines import logger
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

# local imports
from gym_underwater.algos import SAC
from gym_underwater.utils import make_env 
import cmvae_models.cmvae 

# Remove warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# TODO: add other algorithms
# dictionary of available algorithms
ALGOS = {
    'sac': SAC,
}

print("Loading environment configuration ...")
with open(os.path.abspath(os.path.join(os.pardir, 'Configs', 'env', 'config.yml')), 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)

algo = env_config['algo']
seed = env_config['seed']
policy_path = env_config['policy_path']
assert os.path.isfile(policy_path), "No model found at this path: {}".format(policy_path)

set_global_seeds(seed)

# set up a reward log if want one
log_dir = os.path.dirname(policy_path)
os.environ["OPENAI_LOG_FORMAT"] = 'csv'
os.environ["OPENAI_LOGDIR"] = os.path.abspath(log_dir)
logger.configure()

# load trained vae weights if trained vae was used for rl training
vae = None
if env_config['vae_path'] != '':
    print("Loading VAE ...")
    vae = cmvae_models.cmvae.CmvaeDirect(n_z=10, state_dim=3, res=64, trainable_model=False)
    vae.load_weights(env_config['vae_path'])

# wrap environment with DummyVecEnv to prevent code intended for vectorized envs throwing error
env = DummyVecEnv([make_env(vae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_scale'], env_config['debug_logs'], log_dir, seed=seed)]) 

# load trained model
model = ALGOS[algo].load(policy_path)

# collect first observation 
# NB this is calling reset in DummyVecEnv before reset in DonkeyEnv
# DummyVecEnv, like VecEnv, returns list (length 1) of obs, hence why below obs[0] is passed to predict not obs
obs = env.reset()

running_reward = 0.0
ep_len = 0
ep_rewards = []

# perform inference for 100 episodes (standard practice)
while len(ep_rewards) <= 100:

    # query model for action decision given obs
    action, _ = model.predict(obs[0], deterministic=True) 

    # NB this is going to be passed to step in DummyVecEnv before step in DonkeyEnv
    # hence why it needs to be wrapped in a list 
    action = [action]                             

    # step the environment
    obs, reward, done, infos = env.step(action)

    # add reward for step to cumulative episodic reward       
    running_reward += reward

    # increment episode length
    ep_len += 1

    if done or ep_len >= 3000:
        # log episodic reward
        ep_rewards.append(running_reward)
        print("Episode Reward: {:.2f}".format(running_reward[0]))
        print("Episode Length: ", ep_len)
        running_reward = 0.0
        ep_len = 0

print("Finished!")

# reset before close
env.reset()

print("Number of episodes: ", len(ep_rewards))
print("Average reward: ", np.mean(ep_rewards))
print("Reward std: ", np.std(ep_rewards))

# close sim or command line hangs - indexing is to unwrap wrapper
env.envs[0].close()