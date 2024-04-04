"""
Parent script for initiating a training run
"""
import os

from stable_baselines3.common.utils import constant_fn, configure_logger
from gym_underwater.args import args
from gym_underwater.utils.utils import make_env, middle_drop, accelerated_schedule, linear_schedule, load_cmvae_config, load_environment_config, load_hyperparams, load_model, load_callbacks, \
    ENVIRONMENT_TO_LOAD, duplicate_directory

import torch
import tensorflow as tf

assert torch.cuda.is_available() and len(tf.config.list_physical_devices('GPU')) > 0

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_config = load_environment_config(par_dir, seed_tensorflow=True, seed_sb=True)
hyperparams = load_hyperparams(par_dir, env_config['algo'], ENVIRONMENT_TO_LOAD, env_config['seed'])

match hyperparams['learning_rate']:
    case str():
        schedule, initial_value = hyperparams['learning_rate'].split('_')
        initial_value = float(initial_value)
        match schedule:
            case 'md':
                hyperparams['learning_rate'] = middle_drop(initial_value)
            case 'acc':
                hyperparams['learning_rate'] = accelerated_schedule(initial_value)
            case _:
                hyperparams['learning_rate'] = linear_schedule(initial_value)
    case float():
        hyperparams['learning_rate'] = constant_fn(hyperparams['learning_rate'])
    case _:
        raise ValueError('Invalid value for learning rate: {}'.format(hyperparams['learning_rate']))

cmvae, cmvae_config = None, None
if env_config['obs'] == 'cmvae':
    cmvae, cmvae_config = load_cmvae_config(par_dir, True, env_config['seed'])

kwargs = {'total_timesteps': hyperparams['total_timesteps'], 'log_interval': hyperparams['log_interval'], 'tb_log_name': env_config['algo'], 'reset_num_timesteps': True}
del hyperparams['total_timesteps']
del hyperparams['log_interval']

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=str(os.path.join(par_dir, 'logs', env_config['algo'])), tb_log_name=env_config['algo'], reset_num_timesteps=kwargs['reset_num_timesteps'])
hyperparams.update({
    'tensorboard_log': logger.dir
})

# Also performs environment wrapping
env = make_env(cmvae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_res'] if cmvae is None else cmvae_config['img_res'],
               hyperparams['tensorboard_log'] if env_config['debug_logs'] else None, par_dir, args.protocol, args.host, env_config['seed'], inference_only=False)

# If model_path_train is None, will load a new agent
model = load_model(env, env_config['algo'], env_config['model_path_train'], hyperparams)
model.set_logger(logger)

callbacks = load_callbacks(par_dir, env, hyperparams['tensorboard_log'], inference_only=False)
kwargs.update({
    'callback': callbacks
})

model.env.envs[0].unwrapped.wait_until_client_ready()

duplicate_directory(os.path.join(par_dir, 'configs'), os.path.join(hyperparams['tensorboard_log'], 'configs'))

print('Starting training run ...')
model.learn(**kwargs)
model.save(os.path.join(str(model.tensorboard_log), 'final_model'))

model.env.close()
