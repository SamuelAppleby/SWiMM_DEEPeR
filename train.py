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
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training', default='', type=str)
parser.add_argument('-f', '--base-filepath', help='Base filepath for saving outputs and logs', default='gym_underwater/Logs', type=str)
parser.add_argument('-tb', '--tensorboard', help='Turn on/off Tensorboard logging', default=True, type=bool)
parser.add_argument('-l', '--logging', help='Turn on/off saving out Monitor logs NB off still writes but to tmp', default=True, type=bool)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
parser.add_argument('--print-freq', help='Print number of steps to terminal at this interval', default=10, type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
args = parser.parse_args()


# early check on path to trained model if -i arg passed
if args.trained_agent != "":
    assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
        "The argument trained_agent must be a valid path to a .pkl file"

# load hyperparameters from yaml file into dict 
print("Loading hyperparameters ...")
with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)['UnderwaterEnv']

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
# the reason for a second variable is that certain keys will be dropped from 'hyperparams' in prep for passing to model initialiser
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

# TODO: look into what this actually does, whether it is different to env.seed, how many times has to be called etc
# if seed provided in yaml, use it, otherwise use zero
set_global_seeds(hyperparams.get('seed', 0))

# generate filepaths according to base/algo/run/... where run number is generated dynamically 
print("Generating filepaths ...")
algo_specific_path = os.path.join(args.base_filepath, args.algo)
run_id = 0
# if run is first run for args.algo, this for loop won't execute
for path in glob.glob(algo_specific_path + "/[0-9]*"):
    run_num = path.split("/")[-1]
    if run_num.isdigit() and int(run_num) > run_id:
        run_id = int(run_num)
run_specific_path = os.path.join(algo_specific_path, str(run_id + 1))
os.makedirs(run_specific_path, exist_ok=True)
print("Outputs and logs will be saved to {}/... ".format(run_specific_path))

# generate path for TB files
if not args.tensorboard:
    tb_path = None
else:
    tb_path = os.path.join(run_specific_path, 'tb_logs')

# generate path for Monitor logs
if not args.logging:
    log_dir = "/tmp/gym/{}/".format(int(time.time()))
else:
    log_dir = os.path.join(run_specific_path, 'monitor_logs')
os.makedirs(log_dir, exist_ok=True)

def linear_schedule(init_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        init_value = float(init_value)

    def func(progress, _):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * init_value

    return func

if isinstance(hyperparams['learning_rate'], str):
    schedule, initial_value = hyperparams['learning_rate'].split('_')
    initial_value = float(initial_value)
    if schedule == 'lin':
        hyperparams['learning_rate'] = linear_schedule(initial_value)
    else:
        raise ValueError('Schedule not implemented')
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

# DummyVecEnv below expects callable as argument so defining function for creating environment instance
def make_env(log_d, seed=None):
    """
    Makes instance of environment, seeds and wraps with Monitor
    """

    def _init():

        # TODO: is below needed again?
        set_global_seeds(seed)
        # create instance of environment
        env_inst = UnderwaterEnv()
        print("Environment ready")
        # environment seeded with randomly generated seed on initialisation but overwrite if seed provided in yaml 
        if seed > 0:
            # TODO: what is this doing and how is it different to set_global_seeds
            env_inst.seed(seed)
        # wrap environment with SB's Monitor wrapper
        wrapped_env = Monitor(env_inst, log_d, allow_early_resets=True)
        return wrapped_env

    return _init

# wrap environment with DummyVecEnv to prevent code intended for vectorized envs throwing error
env = DummyVecEnv([make_env(log_dir, seed=hyperparams.get('seed', 0))])

# if normalising, wrap environment with VecNormalize wrapper from SB
if normalize:
    env = VecNormalize(env, **normalize_kwargs)

# if training on top of trained model, load trained model
if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
    # Continue training
    print("Loading pretrained agent ...")
    # Policy should not be changed
    del hyperparams['policy'] # network architecture already set so don't need

    model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                  tensorboard_log=tb_path, verbose=1, **hyperparams)

    exp_folder = args.trained_agent.split('.pkl')[0]
    if normalize:
        print("Loading saved running average ...")
        env.load_running_average(exp_folder)

else:
    # Train an agent from scratch
    print("Training from scratch: initialising new model ...")
    model = ALGOS[args.algo](env=env, tensorboard_log=tb_path, verbose=1, **hyperparams)

kwargs = {}
if args.log_interval > -1:
    kwargs = {'log_interval': args.log_interval}

# TODO: implement callbacks

# train model
print("Starting training run ...")
model.learn(n_timesteps, **kwargs)

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
    env.save_running_average(run_specific_path)

# close sim or command line hangs - indexing is to unwrap wrapper
env.envs[0].close()