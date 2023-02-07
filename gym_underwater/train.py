"""
Parent script for initiating a training run
"""

# generic imports
import argparse
import glob
import os
import warnings
import time
from collections import OrderedDict
import yaml
import sys

# specialist imports
from stable_baselines.common import set_global_seeds
from stable_baselines.common.schedules import constfn
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

# local imports
from gym_underwater.algos import SAC
from gym_underwater.gym_env import UnderwaterEnv
import cmvae_models.cmvae

# remove warnings
# TODO: terminal still flooded with warnings, try and remove
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

parser = argparse.ArgumentParser()
parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('--obs', help='Observation type', default='image', type=str, required=False, choices=list(env_config['obs']))
parser.add_argument('--img_scale', help='Image scale', default=[64, 64, 3], nargs='+', type=int, required=False, choices=list(env_config['obs']))
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training', default='', type=str)
parser.add_argument('-f', '--base-filepath', help='Base filepath for saving outputs and logs', default=os.path.join('gym_underwater' + os.sep, 'Logs'), type=str)
parser.add_argument('-tb', '--tensorboard', help='Turn on/off Tensorboard logging', action='store_true')
parser.add_argument('-l', '--logging', help='Turn on/off saving out Monitor logs NB off still writes but to tmp', default=True, type=bool)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
args = parser.parse_args()
args.img_scale = tuple(args.img_scale)


# --------------------------- Utils ------------------------#

def make_env(vae, obs, opt_d, max_d, img_scale, debug_logs, log_d, seed=None):
    """
    Makes instance of environment, seeds and wraps with Monitor
    """

    def _init():
        # TODO: is below needed again?
        set_global_seeds(seed)
        # create instance of environment
        env_inst = UnderwaterEnv(vae, obs, opt_d, max_d, img_scale, debug_logs)
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
if env_config['model'] != "":
    assert env_config['model'].endswith('.pkl') and os.path.isfile(env_config['model']), \
        "The argument trained_agent must be a valid path to a .pkl file"

# if using pretrained vae, create instance of vae object and load trained weights from path provided
vae = None
if env_config['vae_path'] != '':
    print("Loading VAE ...")
    vae = cmvae_models.cmvae.CmvaeDirect(n_z=10, state_dim=3, res=64, trainable_model=False)  # these args should really be dynamically read in
    vae.load_weights(env_config['vae_path'])
else:
    print("Learning from pixels...")

# load hyperparameters from yaml file into dict 
print("Loading hyperparameters ...")
with open(os.path.abspath(os.path.join(os.pardir, 'Configs', 'hyperparams', '{}.yml'.format(env_config['algo']))), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)['UnderwaterEnv']

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
# the reason for a second variable is that certain keys will be dropped from 'hyperparams' in prep for passing to model initialiser
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

# if using vae, save out which model file and which feature dims were used
if vae is not None:
    saved_hyperparams['vae_path'] = env_config['vae_path']
    saved_hyperparams['z_size'] = vae.z_size

# TODO: look into what this actually does, whether it is different to env.seed, how many times has to be called etc
# if seed provided in yaml, use it, otherwise use zero
set_global_seeds(hyperparams.get('seed', 0))

# generate filepaths according to base/algo/run/... where run number is generated dynamically 
print("Generating filepaths ...")
algo_specific_path = os.path.abspath(os.path.join(os.pardir, "Logs", env_config['algo']))
run_id = 0
# if run is first run for algo, this for loop won't execute
for path in glob.glob(algo_specific_path + "/[0-9]*"):
    run_num = path.split(os.sep)[-1]
    if run_num.isdigit() and int(run_num) > run_id:
        run_id = int(run_num)
run_specific_path = os.path.abspath(os.path.join(algo_specific_path, str(run_id + 1)))
os.makedirs(run_specific_path, exist_ok=True)

print("Outputs and logs will be saved to {}".format(run_specific_path))

# generate path for tb files
if not env_config['tb']:
    tb_path = None
else:
    tb_path = os.path.abspath(run_specific_path)

# generate path for Monitor logs
if not env_config['monitor']:
    log_dir = os.path.abspath(os.path.join('tmp', 'gym', '{}'.format(int(time.time()))))
else:
    log_dir = os.path.abspath(run_specific_path)

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
env = DummyVecEnv([make_env(vae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_scale'], env_config['debug_logs'], log_dir, seed=hyperparams.get('seed', 0))])

# if normalising, wrap environment with VecNormalize wrapper from SB
if normalize:
    env = VecNormalize(env, **normalize_kwargs)

# if training on top of trained model, load trained model
if env_config['model'].endswith('.pkl') and os.path.isfile(env_config['model']):
    # Continue training
    print("Loading pretrained agent ...")
    # Policy should not be changed
    del hyperparams['policy']  # network architecture already set so don't need

    model = ALGOS[env_config['algo']].load(env_config['model'], env=env, tensorboard_log=tb_path, verbose=1, **hyperparams)

    exp_folder = env_config['model'].split('.pkl')[0]
    if normalize:
        print("Loading saved running average ...")
        env.load(exp_folder, env)

else:
    # Train an agent from scratch
    print("Training from scratch: initialising new model ...")
    model = ALGOS[env_config['algo']](env=env, tensorboard_log=tb_path, verbose=1, **hyperparams)

kwargs = {}
if env_config['log_interval'] > -1:
    kwargs = {'log_interval': env_config['log_interval']}

# TODO: implement callbacks

# train model
print("Starting training run ...")
model.learn(n_timesteps, **kwargs)

# send message via server

# Close the connection properly
env.reset()

# exit scene?

# Save trained model as .pkl - NOTE set cloudpickle to False to save model as json
model.save(os.path.abspath(os.path.join(run_specific_path, 'model.pkl')), cloudpickle=True)

# Save hyperparams
with open(os.path.abspath(os.path.join(run_specific_path, 'config.yml')), 'w') as f:
    yaml.dump(saved_hyperparams, f)

if normalize:
    # Important: save the running average, for testing the agent we need that normalization
    env.save(run_specific_path)

# close sim or command line hangs - indexing is to unwrap wrapper
env.envs[0].close()
