"""
Parent script for initiating a training run
"""
import os
from collections import OrderedDict
from typing import List
import yaml

from cmvae_models.cmvae import Cmvae, CmvaeDirect

from stable_baselines3.common.utils import constant_fn, configure_logger

from gym_underwater.args import args
from gym_underwater.utils.utils import make_env, middle_drop, accelerated_schedule, linear_schedule, get_wrapper_class, ALGOS, get_callback_list, convert_train_freq

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print('Loading environment configuration ...')
with open(os.path.join(par_dir, 'configs', 'config.yml'), 'r') as f:
    env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
    # TODO Pytorch determinism and seeding
    # tf.keras.utils.set_random_seed(env_config['seed'])

# early check on path to trained model if -i arg passed
if env_config['model_path'] != '':
    assert os.path.exists(env_config['model_path']) and os.path.isfile(env_config['model_path']) and env_config['model_path'].endswith('.zip'), \
        'The argument model_path must be a valid path to a .zip file'

# if using pretrained vae, create instance of vae object and load trained weights from path provided

print('Obs: {}'.format(env_config['obs']))
cmvae = None
if env_config['obs'] == 'cmvae':
    with open(os.path.join(par_dir, 'configs', 'cmvae_config.yml'), 'r') as f:
        cmvae_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        n_z = cmvae_config['n_z']
        latent_space_constraints = cmvae_config['latent_space_constraints']
        print('Loading CMVAE ...')
        if latent_space_constraints:
            cmvae = CmvaeDirect(n_z=n_z, seed=env_config['seed'])
        else:
            cmvae = Cmvae(n_z=n_z, gate_dim=3, seed=env_config['seed'])
        cmvae.load_weights(env_config['cmvae_path'])

# load hyperparameters from yaml file into dict
print('Loading hyperparameters ...')
with open(os.path.join(par_dir, 'configs', '{}.yml'.format(env_config['algo'])), 'r') as f:
    hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)['UnderwaterEnv']
    if isinstance(hyperparams['train_freq'], List):
        hyperparams['train_freq'] = tuple(hyperparams['train_freq'])

hyperparams.update({
    'seed': env_config['seed']
})

# this ordered (alphabetical) dict will be saved out alongside model so know which hyperparams were used for training
# the reason for a second variable is that certain keys will be dropped from 'hyperparams' in prep for passing to model initializer
saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
del saved_hyperparams['callback']
del saved_hyperparams['env_wrapper']

# if using vae, save out which model file and which feature dims were used
if cmvae is not None:
    saved_hyperparams['cmvae_path'] = env_config['cmvae_path']
    saved_hyperparams['z_size'] = int(cmvae.q_img.dense2.units / 2)

# generate filepaths according to base/algo/run/... where run number is generated dynamically
print("Generating filepaths ...")

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

kwargs = {'total_timesteps': hyperparams['total_timesteps'], 'log_interval': hyperparams['log_interval'], 'tb_log_name': env_config['algo'], 'reset_num_timesteps': True}
del hyperparams['total_timesteps']
del hyperparams['log_interval']

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=str(os.path.join(par_dir, 'logs', env_config['algo'])), tb_log_name=env_config['algo'], reset_num_timesteps=kwargs['reset_num_timesteps'])
hyperparams.update({
    'tensorboard_log': logger.dir
})

hyperparams['train_freq'] = convert_train_freq(hyperparams['train_freq'])

env = make_env(cmvae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_res'] if cmvae is None else cmvae_config['img_res'],
               hyperparams['tensorboard_log'] if env_config['debug_logs'] else None, args.protocol, args.host, env_config['seed'])

# Wrapping
env_wrapper = get_wrapper_class(hyperparams, tensorboard_log=hyperparams['tensorboard_log'])
if 'env_wrapper' in hyperparams.keys():
    del hyperparams['env_wrapper']

if env_wrapper is not None:
    env = env_wrapper(env)

# Callbacks
callbacks = get_callback_list(hyperparams, env, tensorboard_log=hyperparams['tensorboard_log'])
if 'callback' in hyperparams.keys():
    del hyperparams["callback"]

kwargs.update({
    'callback': callbacks
})

if os.path.isfile(env_config['model_path']):
    print('Loading pretrained agent ...')
    del hyperparams['policy']  # network architecture already set so don't need
    model = ALGOS[env_config['algo']].load(path=env_config['model_path'], env=env, **hyperparams)
else:
    # Train an agent from scratch
    print('Training from scratch: initialising new model ...')
    model = ALGOS[env_config['algo']](env=env, **hyperparams)

model.set_logger(logger)
env.unwrapped.wait_until_client_ready()

print('Starting training run ...')
model.learn(**kwargs)
model.save(os.path.join(str(model.tensorboard_log), 'final_model'))

# Save hyperparams
with open(os.path.join(hyperparams['tensorboard_log'], 'config.yml'), 'w') as f:
    yaml.dump(saved_hyperparams, f)

env.close()
