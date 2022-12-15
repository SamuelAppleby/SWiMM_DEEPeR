"""
Parent script for initiating a training run
"""

# generic imports
import argparse
import os
import glob
import warnings
import time
from collections import OrderedDict
import yaml

# specialist imports
from stable_baselines.common import set_global_seeds
from stable_baselines.common.schedules import constfn
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

# local imports
from algos import SAC
from gym_underwater.gym_env import UnderwaterEnv
from Configs.config import *

# remove warnings
# TODO: terminal still flooded with warnings, try and remove
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

# TODO: add other algorithms
# dictionary of available algorithms
ALGOS = {
    'sac': SAC,
}

parser = argparse.ArgumentParser()
parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('--obs', help='Observation type', default='image', type=str, required=False, choices=list(OBS))
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training', default='', type=str)
parser.add_argument('-f', '--base-filepath', help='Base filepath for saving outputs and logs', default=os.path.join('gym_underwater' + os.sep, 'Logs'), type=str)
parser.add_argument('-tb', '--tensorboard', help='Turn on/off Tensorboard logging', default=True, type=bool)
parser.add_argument('-l', '--logging', help='Turn on/off saving out Monitor logs NB off still writes but to tmp', default=True, type=bool)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
args = parser.parse_args()


# --------------------------- Utils ------------------------#

def make_env(obs, log_d, seed=None):
    """
    Makes instance of environment, seeds and wraps with Monitor
    """

    def _init():
        # TODO: is below needed again?
        set_global_seeds(seed)
        # create instance of environment
        env_inst = UnderwaterEnv(obs)
        print("Environment ready")
        # environment seeded with randomly generated seed on initialisation but overwrite if seed provided in yaml 
        if seed > 0:
            # TODO: what is this doing and how is it different to set_global_seeds
            env_inst.seed(seed)
        # wrap environment with SB's Monitor wrapper
        wrapped_env = Monitor(env_inst, log_d, allow_early_resets=True)
        return wrapped_env

    return _init


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress, _):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func


def middle_drop(initial_value):
    """
    Similar to stable_baselines.common.schedules middle_drop, but func returns actual LR value not multiplier.
    Produces linear schedule but with a drop half way through training to a constant schedule at 1/10th initial value.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress, _):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        eps = 0.5
        if progress < eps:
            return initial_value * 0.1
        return progress * initial_value

    return func


def accelerated_schedule(initial_value):
    """
    Custom schedule, starts as linear schedule but once mean_reward (episodic reward averaged over the last 100 episodes)
    surpasses a threshold, schedule remains annealing but at the tail end toward an LR of zero, by taking
    1/10th of the progress before multiplying.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress, mean_reward):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        rew_threshold = 1000
        if mean_reward >= rew_threshold:
            return (progress * 0.1) * initial_value
        return progress * initial_value

    return func


# ---------------------------- Main script ----------------------------------#

# early check on path to trained model if -i arg passed
if MODEL != "":
    assert MODEL.endswith('.pkl') and os.path.isfile(MODEL), \
        "The argument trained_agent must be a valid path to a .pkl file"

# load hyperparameters from yaml file into dict 
print("Loading hyperparameters ...")
with open('hyperparams/{}.yml'.format(ALGO), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)['UnderwaterEnv']

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
# the reason for a second variable is that certain keys will be dropped from 'hyperparams' in prep for passing to model initialiser
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

# TODO: look into what this actually does, whether it is different to env.seed, how many times has to be called etc
# if seed provided in yaml, use it, otherwise use zero
set_global_seeds(hyperparams.get('seed', 0))

# generate filepaths according to base/algo/run/... where run number is generated dynamically 
print("Generating filepaths ...")
algo_specific_path = os.path.join(BASE_FILEPATH, ALGO)
run_id = 0
# if run is first run for algo, this for loop won't execute
for path in glob.glob(algo_specific_path + "/[0-9]*"):
    run_num = path.split("/")[-1]
    if run_num.isdigit() and int(run_num) > run_id:
        run_id = int(run_num)
run_specific_path = os.path.join(algo_specific_path, str(run_id + 1))
os.makedirs(run_specific_path, exist_ok=True)
print("Outputs and logs will be saved to {}/... ".format(run_specific_path))

# generate path for TB files
if not TB:
    tb_path = None
else:
    tb_path = os.path.join(run_specific_path, 'tb_logs')

# generate path for Monitor logs
if not MONITOR:
    log_dir = "/tmp/gym/{}/".format(int(time.time()))
else:
    log_dir = os.path.join(run_specific_path, 'monitor_logs')
os.makedirs(log_dir, exist_ok=True)

if isinstance(hyperparams['learning_rate'], str):
    schedule, initial_value = hyperparams['learning_rate'].split('_')
    initial_value = float(initial_value)
    if schedule == 'md':
        hyperparams['learning_rate'] = middle_drop(initial_value)
    elif schedule == 'acc':
        hyperparams['learning_rate'] = accelerated_schedule(initial_value)
    else:
        hyperparams['learning_rate'] = linear_schedule(initial_value)
elif isinstance(hyperparams['learning_rate'], float):
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

# wrap environment with DummyVecEnv to prevent code intended for vectorized envs throwing error
env = DummyVecEnv([make_env(OBS, log_dir, seed=hyperparams.get('seed', 0))])

# if normalising, wrap environment with VecNormalize wrapper from SB
if normalize:
    env = VecNormalize(env, **normalize_kwargs)

# if training on top of trained model, load trained model
if MODEL.endswith('.pkl') and os.path.isfile(MODEL):
    # Continue training
    print("Loading pretrained agent ...")
    # Policy should not be changed
    del hyperparams['policy']  # network architecture already set so don't need

    model = ALGOS[ALGO].load(MODEL, env=env, tensorboard_log=tb_path, verbose=1, **hyperparams)

    exp_folder = MODEL.split('.pkl')[0]
    if normalize:
        print("Loading saved running average ...")
        env.load(exp_folder, env)

else:
    # Train an agent from scratch
    print("Training from scratch: initialising new model ...")
    model = ALGOS[ALGO](env=env, tensorboard_log=tb_path, verbose=1, **hyperparams)

kwargs = {}
if LOG_INTERVAL > -1:
    kwargs = {'log_interval': LOG_INTERVAL}

# TODO: implement callbacks

# train model
print("Starting training run ...")
model.learn(n_timesteps, **kwargs)

# send messsage via server

# Close the connection properly
env.reset()

# exit scene?

# Save trained model as .pkl - NOTE set cloudpickle to False to save model as json
model.save(run_specific_path, cloudpickle=True)

# Save hyperparams
with open(os.path.join(run_specific_path, 'config.yml'), 'w') as f:
    yaml.dump(saved_hyperparams, f)

if normalize:
    # Important: save the running average, for testing the agent we need that normalization
    env.save(run_specific_path)

# close sim or command line hangs - indexing is to unwrap wrapper
env.envs[0].close()
