"""
Parent script for initiating a training run
"""
# generic imports
import argparse
import glob
import os
import time
from collections import OrderedDict
import yaml
import sys

# specialist imports
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import SAC

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, par_dir)

# local imports
import cmvae_models.cmvae
from gym_underwater.utils.utils import make_env, middle_drop, accelerated_schedule, linear_schedule
from gym_underwater.algos.callbacks import SwimCallback
from gym_underwater.sim_comms import Protocol

# TODO Pytorch determinism and seeding
print('Loading environment configuration ...')
with open(os.path.join(par_dir, 'configs', 'config.yml'), 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
    # tf.keras.utils.set_random_seed(env_config['seed'])

# early check on path to trained model if -i arg passed
if env_config['model_path'] != '':
    assert os.path.exists(env_config['model_path']) and os.path.isfile(env_config['model_path']) and env_config['model_path'].endswith('.zip'), \
        'The argument model_path must be a valid path to a .zip file'

# if using pretrained vae, create instance of vae object and load trained weights from path provided

print("Obs: {}".format(env_config['obs']))
cmvae = None
if env_config['obs'] == 'cmvae':
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'configs', 'cmvae_config.yml'), 'r') as f:
        cmvae_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        n_z = cmvae_config['n_z']
        latent_space_constraints = cmvae_config['latent_space_constraints']
        print('Loading CMVAE ...')
        if latent_space_constraints:
            cmvae = cmvae_models.cmvae.CmvaeDirect(n_z=n_z, seed=env_config['seed'])
        else:
            cmvae = cmvae_models.cmvae.Cmvae(n_z=n_z, gate_dim=3, seed=env_config['seed'])
        cmvae.load_weights(env_config['cmvae_path'])

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
    'sac': SAC
}

# load hyperparameters from yaml file into dict
print('Loading hyperparameters ...')
with open(os.path.join(par_dir, 'configs', '{}.yml'.format(env_config['algo'])), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)['UnderwaterEnv']

hyperparams['seed'] = env_config['seed']

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
# the reason for a second variable is that certain keys will be dropped from 'hyperparams' in prep for passing to model initialiser
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

# if using vae, save out which model file and which feature dims were used
if cmvae is not None:
    saved_hyperparams['cmvae_path'] = env_config['cmvae_path']
    saved_hyperparams['z_size'] = int(cmvae.q_img.dense2.units / 2)

# generate filepaths according to base/algo/run/... where run number is generated dynamically 
print("Generating filepaths ...")
algo_specific_path = str(os.path.join(par_dir, "logs", env_config['algo']))
run_id = -1
# if run is first run for algo, this for loop won't execute
for path in glob.glob(algo_specific_path + "/[0-9]*"):
    run_num = path.split(os.sep)[-1]
    if run_num.isdigit() and int(run_num) > run_id:
        run_id = int(run_num)
run_specific_path = os.path.join(algo_specific_path, str(run_id + 1))
os.makedirs(run_specific_path, exist_ok=True)

hyperparams['tensorboard_log'] = run_specific_path

if not env_config['monitor']:
    log_dir = os.path.join('tmp', 'gym', '{}'.format(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)
else:
    log_dir = run_specific_path

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
    hyperparams['learning_rate'] = constant_fn(hyperparams['learning_rate'])
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

img_res = env_config['img_res'] if cmvae is None else cmvae_config['img_res']

# wrap environment with DummyVecEnv to prevent code intended for vectorized envs throwing error
env = DummyVecEnv([make_env(cmvae, env_config['obs'], env_config['opt_d'], env_config['max_d'], img_res, env_config['debug_logs'],
                            args.protocol, args.host, log_dir, env_config['ep_length_threshold'], env_config['seed'])])

# if normalising, wrap environment with VecNormalize wrapper from SB
if normalize:
    env = VecNormalize(env, **normalize_kwargs)

# if training on top of trained model, load trained model
if os.path.isfile(env_config['model_path']):
    # Continue training
    print("Loading pretrained agent ...")
    # Policy should not be changed
    del hyperparams['policy']  # network architecture already set so don't need
    model = ALGOS[env_config['algo']].load(path=env_config['model_path'], env=env, **hyperparams)

    if normalize:
        print("Loading saved running average ...")
        exp_folder = env_config['model'].split('.zip')[0]
        env.load(exp_folder, env)
else:
    # Train an agent from scratch
    print("Training from scratch: initialising new model ...")
    model = ALGOS[env_config['algo']](env=env, **hyperparams)

kwargs = {'total_timesteps': n_timesteps, 'callback': SwimCallback(), 'log_interval': env_config['log_interval'], 'tb_log_name': env_config['algo'], 'reset_num_timesteps': True, 'progress_bar': True}

# off_policy_algorithm forces no csv output, so recreate the function and set a custom logger
save_path, format_strings = model.tensorboard_log, ['stdout']

if model.tensorboard_log is not None and SummaryWriter is None:
    if SummaryWriter is None:
        raise ImportError('Trying to log data to tensorboard but tensorboard is not installed.')
    # latest_run_id = get_latest_run_id(model.tensorboard_log, kwargs['tb_log_name'])
    # if not kwargs['reset_num_timesteps']:
    #     # Continue training in the same directory
    #     latest_run_id -= 1
    # save_path = os.path.join(save_path, f"{kwargs['tb_log_name']}_{latest_run_id + 1}")
    if model.verbose >= 1:
        format_strings = ['stdout', 'tensorboard', 'csv']
    else:
        format_strings = ['tensorboard']
elif model.verbose == 0:
    format_strings = [""]

model.set_logger(configure(save_path, format_strings))

# Train model
print("Starting training run ...")
model.learn(**kwargs)

# Close the connection properly
env.reset()

# Save final model, regardless of state
model.save(os.path.join(str(model.tensorboard_log), 'final_model'))

# Save hyperparams
with open(os.path.join(run_specific_path, 'config.yml'), 'w') as f:
    yaml.dump(saved_hyperparams, f)

if normalize:
    # Important: save the running average, for testing the agent we need that normalization
    env.save(run_specific_path)

env.envs[0].close()
