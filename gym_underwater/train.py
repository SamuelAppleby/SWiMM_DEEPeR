"""
Parent script for initiating a training run
"""
import argparse
# generic imports
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
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

from gym_underwater.python_server import Protocol

# code to go up a directory so higher level modules can be imported
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)

# local imports
from gym_underwater.algos import SAC
from gym_underwater.utils import make_env, middle_drop, accelerated_schedule, linear_schedule, create_callback
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
parser.add_argument('-h', '--host', help='Override the host for network (with port)', default='127.0.0.1:60260', type=str)
parser.add_argument('-tcp', help='Enable tcp', action='store_true')
parser.add_argument('-udp', help='Enable udp', action='store_true')
args = parser.parse_args()

if (args.tcp and args.udp) or args.tcp:
    args.protocol = Protocol.TCP
else:
    args.protocol = Protocol.UDP

# early check on path to trained model if -i arg passed
if env_config['model'] != "":
    assert env_config['model'].endswith('.pkl') and os.path.isfile(env_config['model']), \
        "The argument trained_agent must be a valid path to a .pkl file"

# if using pretrained vae, create instance of vae object and load trained weights from path provided
print("Obs: {}".format(env_config['obs']))
vae = None
if env_config['obs'] == 'vae':

    if env_config['vae_path'] is '':
        print('For vae training, must provide a valid vae path!')
        quit()

    print("Loading VAE ...")
    vae = cmvae_models.cmvae.CmvaeDirect(n_z=10, state_dim=3, res=64, trainable_model=False)  # these args should really be dynamically read in
    vae.load_weights(env_config['vae_path'])

# load hyperparameters from yaml file into dict
print("Loading hyperparameters ...")
with open(os.path.abspath(os.path.join(os.pardir, 'Configs', 'hyperparams', '{}.yml'.format(env_config['algo']))), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)['UnderwaterEnv']

# add seed provided by config
hyperparams['seed'] = env_config['seed']

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
# the reason for a second variable is that certain keys will be dropped from 'hyperparams' in prep for passing to model initialiser
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

# if using vae, save out which model file and which feature dims were used
if vae is not None:
    saved_hyperparams['vae_path'] = env_config['vae_path']
    saved_hyperparams['z_size'] = vae.z_size

# if seed provided, use it, otherwise use zero
# Note: this stable baselines utility function seeds tensorflow, np.random, and random
seed = hyperparams.get('seed', 0)
set_global_seeds(seed)

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
env = DummyVecEnv([make_env(vae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_scale'], env_config['debug_logs'], args.protocol, args.host, log_dir, seed=seed)])

# if normalising, wrap environment with VecNormalize wrapper from SB
if normalize:
    env = VecNormalize(env, **normalize_kwargs)

# if training on top of trained model, load trained model
if env_config['model'].endswith('.pkl') and os.path.isfile(env_config['model']):
    
    # Continue training
    print("Loading pretrained agent ...")
    
    # Policy should not be changed
    del hyperparams['policy']  # network architecture already set so don't need

    # if config file provides path to existing tensorboard events file, overwrite auto generated tb_path
    if env_config['tb_path'] != "":
        tb_path = env_config['tb_path']

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

if env_config['algo'] == 'sac':
    kwargs.update({'callback': create_callback(env_config['algo'], run_specific_path, reward_threshold=2200, verbose=1)})

# train model
print("Starting training run ...")
model.learn(n_timesteps, **kwargs)

# send message via server

# Close the connection properly
env.reset()

# exit scene?

# Save trained model as .pkl - NOTE set cloudpickle to False to save model as json
model.save(os.path.abspath(os.path.join(run_specific_path, 'finalmodel.pkl')), cloudpickle=True)

# Save hyperparams
with open(os.path.abspath(os.path.join(run_specific_path, 'config.yml')), 'w') as f:
    yaml.dump(saved_hyperparams, f)

if normalize:
    # Important: save the running average, for testing the agent we need that normalization
    env.save(run_specific_path)

# close sim or command line hangs - indexing is to unwrap wrapper
env.envs[0].close()
