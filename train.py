import argparse
import os
import glob
import warnings
import time
from collections import OrderedDict
import yaml
from gym_underwater.gym_env import UnderwaterEnv

# Remove warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

from stable_baselines.common import set_global_seeds
from stable_baselines import SAC
from stable_baselines.common.schedules import constfn
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

from config import *

# TODO: add other algorithms
ALGOS = {
    'sac': SAC,
}

parser = argparse.ArgumentParser()
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                    default='', type=str)
parser.add_argument('-f', '--model-folder', help='Base filepath for saving outputs', type=str, default='models')
parser.add_argument('-tb', '--tensorboard-folder', help='Filepath for saving tensorboard files', default='', type=str)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                    type=int)
args = parser.parse_args()

# TODO: look into what this actually does, whether it is different to env.seed below, how many times has to be called etc
#set_global_seeds(args.seed)

# early check on path to trained model if -i arg passed
if args.trained_agent != "":
    assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .pkl file"

# load hyperparameters from yaml file into dict 
with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

# generate path for saving models and saved_hyperparams
algo_specific_path = os.path.join(args.model_folder, args.algo)
run_id = 0
for path in glob.glob(algo_specific_path + "/[0-9]*"):
        run_num = path.split("/")[-1]
        if run_num.isdigit() and int(run_num) > run_id:
            run_id = int(run_num)
run_specific_path = os.path.join(algo_specific_path, run_id + 1)

# generate path for TB files
if args.tensorboard_folder == '':
    tb_path = None
else:
    tb_algo_specific_path = os.path.join(args.tensorboard_folder, args.algo)
    tb_path = os.path.join(tb_algo_specific_path, run_id + 1)

# TODO: implement linear and other LR schedules and include as option 
if isinstance(hyperparams['learning_rate'], float):
    hyperparams['learning_rate'] = constfn(hyperparams['learning_rate'])
else:
    raise ValueError('Invalid value for learning rate: {}'.format(hyperparams['learning_rate']))

# extract number of time steps want to train for 
n_timesteps = int(hyperparams['n_timesteps'])
# once extracted delete since not hyperparam expected by model initialiser
del hyperparams['n_timesteps'] 

# process normalisation hyperparams
normalize = False
normalize_kwargs = {}
if 'normalize' in hyperparams.keys():
    normalize = hyperparams['normalize']
    if isinstance(normalize, str):
        normalize_kwargs = eval(normalize)
        normalize = True
    del hyperparams['normalize'] 

# use temp dir for logs generated by stable baselines' Monitor wrapper
log_dir = "/tmp/gym/{}/".format(int(time.time()))
os.makedirs(log_dir, exist_ok=True)

# create instance of environment
env = UnderwaterEnv()

# environment seeded with randomly generated seed on initialisation but overwrite if seed provided in yaml 
#if 'seed' in hyperparams.keys():
    #env.seed(hyperparams.get('seed', 0))

# wrap environment with stable baseline's Monitor wrapper
env = Monitor(env, log_dir, allow_early_resets=True)

# wrap environment with DummyVecEnv to prevent code intended for vectorized envs throwing error
env = DummyVecEnv(env) 

# if normalising, wrap environment with VecNormalize wrapper from SB
if normalize:
    env = VecNormalize(env, **normalize_kwargs)

# if training on top of trained model, load trained model
if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
    # Continue training
    print("Loading pretrained agent")
    # Policy should not be changed
    del hyperparams['policy'] # network architecture already set so don't need

    model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                  tensorboard_log=tb_path, verbose=1, **hyperparams)

    exp_folder = args.trained_agent.split('.pkl')[0]
    if normalize:
        print("Loading saved running average")
        env.load_running_average(exp_folder)

else:
    # Train an agent from scratch
    model = ALGOS[args.algo](env=env, tensorboard_log=tb_path, verbose=1, **hyperparams)
