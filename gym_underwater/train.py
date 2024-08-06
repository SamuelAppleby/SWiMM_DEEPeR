"""
Parent script for initiating a training run
"""
import os
import yaml

from gym_underwater.enums import TrainingType, ObservationType

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_global_config.yml'), 'r') as f:
    cmvae_global_config = yaml.load(f, Loader=yaml.UnsafeLoader)

import torch
from torch import nn as nn
import tensorflow as tf

if cmvae_global_config['use_cpu_only']:
    tf.config.set_visible_devices([], 'GPU')

from stable_baselines3.common.utils import constant_fn, configure_logger

from constants import IP_HOST, PORT_TRAIN, ENVIRONMENT_TO_LOAD
from utils import make_env, middle_drop, accelerated_schedule, linear_schedule, load_environment_config, load_hyperparams, load_callbacks, \
    load_cmvae_inference_config, output_devices, parse_command_args, tensorflow_seeding, duplicate_directory, load_pretrained_model, load_new_model, load_cmvae, preprocess_action_noise, \
    output_command_line_arguments, convert_observation_type

from stable_baselines3.common.vec_env import DummyVecEnv

env_config = load_environment_config(project_dir)

obs = convert_observation_type(env_config['obs'])

cmvae_inference_config = load_cmvae_inference_config(project_dir)

parse_command_args(env_config, cmvae_inference_config)

# NB Very important, _setup_model (for both on/off-policy algorithms) will call every seeding operation (see stable_baselines3.common.base_class.set_random_seed)
tensorflow_seeding(env_config['seed'])

# Also adds ['seed'] to hyperparams
hyperparams = load_hyperparams(project_dir, env_config['algorithm'], ENVIRONMENT_TO_LOAD, env_config['seed'])

for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
    if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
        hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

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

kwargs = {
    'total_timesteps': hyperparams['total_timesteps'],
    'log_interval': hyperparams['log_interval'],
    'tb_log_name': env_config['algorithm'],
    'reset_num_timesteps': hyperparams['reset_num_timesteps'],
    'progress_bar': hyperparams['progress_bar']
}

if 'total_timesteps' in hyperparams:
    del hyperparams['total_timesteps']
if 'log_interval' in hyperparams:
    del hyperparams['log_interval']
if 'reset_num_timesteps' in hyperparams:
    del hyperparams['reset_num_timesteps']
if 'progress_bar' in hyperparams:
    del hyperparams['progress_bar']

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=str(os.path.join(project_dir, 'models', env_config['algorithm'])), tb_log_name=env_config['algorithm'], reset_num_timesteps=kwargs['reset_num_timesteps'])
hyperparams.update({
    'tensorboard_log': logger.dir,
})

cmvae = load_cmvae(cmvae_global_config=cmvae_global_config, weights_path=cmvae_inference_config['weights_path'])

# Also performs environment wrapping
env = DummyVecEnv([make_env(cmvae=cmvae, obs=obs, img_res=env_config['img_res'], tensorboard_log=hyperparams['tensorboard_log'], debug_logs=env_config['debug_logs'], ip=IP_HOST, port=(PORT_TRAIN+i), training_type=TrainingType.TRAINING, seed=((env_config['seed']+i) if env_config['seed'] is not None else None)) for i in range(env_config['n_envs'])])

hyperparams = preprocess_action_noise(hyperparams, env)

# If pre_trained_model_path is None, will load a new agent
if env_config['pre_trained_model_path'] is not None:
    model = load_pretrained_model(env=env, algorithm_name=env_config['algorithm'], model_path=env_config['pre_trained_model_path'], hyperparams=hyperparams)
else:
    model = load_new_model(env=env, algorithm_name=env_config['algorithm'], hyperparams=hyperparams)

model.set_logger(logger)

callbacks = load_callbacks(project_dir, env, hyperparams['tensorboard_log'])

kwargs.update({
    'callback': callbacks
})

model.learn(**kwargs)
model.save(os.path.join(model.tensorboard_log, 'final_model'))

config_dir = os.path.join(hyperparams['tensorboard_log'], 'configs')

hyperparams_to_exclude = [file for file in os.listdir(os.path.join(project_dir, 'configs', 'hyperparams')) if file != f'{env_config["algorithm"].lower()}.yml']

duplicate_directory(os.path.join(project_dir, 'configs'), config_dir, dirs_to_exclude=None, files_to_exclude=(hyperparams_to_exclude + ['cmvae_training_config.yml']))
output_devices(config_dir, tensorflow_device=True, torch_device=True)
output_command_line_arguments(config_dir)
