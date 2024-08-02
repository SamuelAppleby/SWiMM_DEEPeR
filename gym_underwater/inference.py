"""
Parent script for initiating an inference run
"""
import os
import yaml
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_underwater.enums import TrainingType

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_global_config.yml'), 'r') as f:
    cmvae_global_config = yaml.load(f, Loader=yaml.UnsafeLoader)

import torch
import tensorflow as tf

if cmvae_global_config['use_cpu_only']:
    tf.config.set_visible_devices([], 'GPU')

from typing import List, Dict
from stable_baselines3.common.utils import configure_logger

from gym_underwater.constants import IP_HOST, PORT_INFERENCE, ENVIRONMENT_TO_LOAD

from gym_underwater.utils import make_env, load_environment_config, load_cmvae_inference_config, output_devices, duplicate_directory, \
    parse_command_args, tensorflow_seeding, load_pretrained_model, get_class_by_name, load_cmvae

env_config = load_environment_config(project_dir)
assert env_config['n_envs'] == 1, 'When running in inference mode, can only run 1 environment'
assert os.path.isfile(env_config['pre_trained_model_path']) and env_config['pre_trained_model_path'].endswith('.zip'), 'The argument pre_trained_model_path must be a valid path to a .zip file'

cmvae_inference_config = load_cmvae_inference_config(project_dir)

parse_command_args(env_config, cmvae_inference_config)

# NB Very important, _setup_model (for both on/off-policy algorithms) will call every seeding operation (see stable_baselines3.common.base_class.set_random_seed)
tensorflow_seeding(env_config['seed'])

cmvae = load_cmvae(cmvae_global_config=cmvae_global_config, weights_path=cmvae_inference_config['weights_path'])

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=os.path.join(os.path.dirname(env_config['pre_trained_model_path']), 'inference'), tb_log_name=env_config['algorithm'], reset_num_timesteps=True)

with open(os.path.join(project_dir, 'configs', 'callbacks.yml'), 'r') as f:
    callback_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)[ENVIRONMENT_TO_LOAD]
    eval_callback = next(filter(lambda x: isinstance(x, Dict) and 'gym_underwater.callbacks.SwimEvalCallback' in x.keys(), callback_wrapper_config), None)

assert eval_callback is not None, 'Must provide a SwimEvalCallback object in callbacks.yml'

if isinstance(eval_callback, dict):
    assert len(eval_callback) == 1, (
        "You have an error in the formatting "
        f"of your YAML file near {eval_callback}. "
        "You should check the indentation."
    )
    callback_dict = eval_callback

    callback_name = next(iter(callback_dict.keys()))
    kwargs = callback_dict[callback_name]

    for param in ['callback_after_eval', 'callback_on_new_best']:
        del kwargs[param]
else:
    kwargs = {}

callback_class = get_class_by_name(callback_name)

env = DummyVecEnv([make_env(cmvae=cmvae, obs=env_config['obs'], img_res=env_config['img_res'], tensorboard_log=logger.dir, debug_logs=env_config['debug_logs'], ip=IP_HOST, port=(PORT_INFERENCE+i), training_type=TrainingType.INFERENCE, seed=(env_config['seed']+i)) for i in range(env_config['n_envs'])])

kwargs.update({
    'eval_env': env,
    'log_path': logger.dir
})

for freq in ['eval_freq', 'eval_inference_freq']:
    if freq in kwargs and isinstance(kwargs[freq], List):
        kwargs[freq] = tuple(kwargs[freq])

eval_callback = callback_class(**kwargs)

model = load_pretrained_model(env=env, algorithm_name=env_config['algorithm'], model_path=env_config['pre_trained_model_path'])

model.set_logger(logger)

# We have to manually initialise the callbacks as we want to ensure a consistent flow across training and evaluation, but callbacks are only initialised during setup_learn
eval_callback = model._init_callback(eval_callback, progress_bar=False)
eval_callback.evaluate()

config_dir = os.path.join(logger.dir, 'configs')

hyperparams_to_exclude = [file for file in os.listdir(os.path.join(project_dir, 'configs', 'hyperparams')) if file != f'{env_config["algorithm"].lower()}.yml']

duplicate_directory(os.path.join(project_dir, 'configs'), config_dir, dirs_to_exclude=['hyperparams'], files_to_exclude=(hyperparams_to_exclude + ['cmvae_training_config.yml', 'cmvae_global_config.yml']))
output_devices(config_dir, tensorflow_device=True, torch_device=True)

model.env.close()
