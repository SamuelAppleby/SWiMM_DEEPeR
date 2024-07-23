import argparse
import importlib
import json
import os

import shutil
from typing import Dict, Type, Any, Optional, Callable, List

import gymnasium
import keras
import numpy as np
import tensorflow as tf
import torch
import yaml

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from cmvae_models.cmvae import CmvaeDirect, Cmvae
from gym_underwater.callbacks import convert_train_freq
from gym_underwater.constants import ENVIRONMENT_TO_LOAD, IP_HOST, PORT_TRAIN, PORT_INFERENCE, ALGOS
from gym_underwater.gym_env import UnderwaterEnv


def make_env(cmvae: tf.keras.Model, obs: str, img_res, tensorboard_log: str, debug_logs: bool = False, ip: str = IP_HOST, port: int = PORT_TRAIN, seed: int = None) -> gymnasium.Env:
    """
    Makes instance of environment, seeds and wraps with Monitor
    """
    uenv = UnderwaterEnv(obs=obs, img_res=img_res, tensorboard_log=tensorboard_log, debug_logs=debug_logs, ip=ip, port=port, seed=seed, cmvae=cmvae)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'env_wrapper.yml'), 'r') as f:
        env_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)

        if env_wrapper_config is not None:
            uenv_conf = env_wrapper_config[ENVIRONMENT_TO_LOAD]
            env_wrapper = get_wrapper_class(uenv_conf, monitor_filename=os.path.join(tensorboard_log, 'training_monitor.csv' if port == PORT_TRAIN else 'testing_monitor.csv'))

            if env_wrapper is not None:
                uenv = env_wrapper(uenv)

    uenv.unwrapped.wait_until_client_ready()
    return uenv


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
    Produces linear schedule but with a drop halfway through training to a constant schedule at 1/10th initial value.

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


# We require that Monitor is the very last wrapper to execute, as it relies on values obtained from TimeLimit
def custom_wrapper_sort(item):
    if isinstance(item, dict):
        item = next(iter(item.keys()))
    match item:
        case 'gymnasium.wrappers.time_limit.TimeLimit':
            return 0
        case 'stable_baselines3.common.monitor.Monitor':
            return np.inf
        case _:
            return 0


# Adapted from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/utils.py#L47
def get_wrapper_class(wrapper_list: List[Any], monitor_filename: str = None) -> Optional[Callable[[gymnasium.Env], gymnasium.Env]]:
    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if wrapper_list is None or len(wrapper_list) == 0:
        return None

    wrapper_list.sort(key=custom_wrapper_sort)

    wrapper_classes = []
    wrapper_kwargs = []
    # Handle multiple wrappers
    for wrapper_name in wrapper_list:
        # Handle keyword arguments
        if isinstance(wrapper_name, dict):
            assert len(wrapper_name) == 1, (
                'You have an error in the formatting '
                f'of your YAML file near {wrapper_name}. '
                'You should check the indentation.'
            )
            wrapper_dict = wrapper_name
            wrapper_name = next(iter(wrapper_dict.keys()))
            kwargs = wrapper_dict[wrapper_name]
        else:
            kwargs = {}
        wrapper_module = importlib.import_module(get_module_name(wrapper_name))
        wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
        wrapper_classes.append(wrapper_class)

        if issubclass(wrapper_class, Monitor):
            kwargs.update({
                'filename': monitor_filename
            })

        wrapper_kwargs.append(kwargs)

    def wrap_env(env: gymnasium.Env) -> gymnasium.Env:
        """
        :param env:
        :return:
        """
        for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
            env = wrapper_class(env, **kwargs)
        return env

    return wrap_env


def get_class_by_name(name: str) -> Type:
    """
    Imports and returns a class given the name, e.g. passing
    'stable_baselines3.common.callbacks.CheckpointCallback' returns the
    CheckpointCallback class.

    :param name:
    :return:
    """

    def get_module_name(name: str) -> str:
        return ".".join(name.split(".")[:-1])

    def get_class_name(name: str) -> str:
        return name.split(".")[-1]

    module = importlib.import_module(get_module_name(name))
    return getattr(module, get_class_name(name))


# SwimCallback must be last, as that is the one restarting physics
def custom_callback_sort(item):
    if isinstance(item, dict):
        item = next(iter(item.keys()))
    match item:
        case 'gym_underwater.callbacks.SwimCallback':
            return 0
        case 'gym_underwater.callbacks.SwimProgressBarCallback':
            return 1
        case 'gym_underwater.callbacks.SwimEvalCallback':
            return np.inf
        case _:
            return 0


def get_callback_list(callback_list: List[Any], env: gymnasium.Env = None, tensorboard_log: str = None) -> List[BaseCallback]:
    callbacks: List[BaseCallback] = []

    # We need to order our callbacks in a precedential manner
    callback_list.sort(key=custom_callback_sort)

    # Handle multiple wrappers
    for callback_name in callback_list:
        # Handle keyword arguments
        if isinstance(callback_name, dict):
            assert len(callback_name) == 1, (
                "You have an error in the formatting "
                f"of your YAML file near {callback_name}. "
                "You should check the indentation."
            )
            callback_dict = callback_name

            callback_name = next(iter(callback_dict.keys()))
            kwargs = callback_dict[callback_name]
        else:
            kwargs = {}

        callback_class = get_class_by_name(callback_name)

        if issubclass(callback_class, EvalCallback):
            eval_env = make_env(cmvae=env.unwrapped.cmvae, obs=env.unwrapped.obs, img_res=env.unwrapped.handler.img_res, tensorboard_log=env.unwrapped.tensorboard_log,
                                debug_logs=env.unwrapped.handler.debug_logs, ip=IP_HOST, port=PORT_INFERENCE, seed=env.unwrapped.seed)

            kwargs.update({
                'eval_env': eval_env,
                'log_path': tensorboard_log,
                'best_model_save_path': tensorboard_log
            })

            for freq in ['eval_freq', 'eval_inference_freq']:
                if freq in kwargs and isinstance(kwargs[freq], List):
                    kwargs[freq] = tuple(kwargs[freq])

            for c_name in ['callback_on_new_best', 'callback_after_eval']:
                recurse_callbacks(c_name, kwargs, env, tensorboard_log)

        callbacks.append(callback_class(**kwargs))

    return callbacks


def recurse_callbacks(nested_item: str = None, kwargs: Dict[str, Any] = None, env: gymnasium.Env = None, tensorboard_log: str = None):
    # No valid nested key was found
    if nested_item not in kwargs.keys():
        return

    assert nested_item in kwargs.keys()
    nest_list = []

    assert isinstance(kwargs[nested_item], dict), (
        f'There is an error within the {nested_item}'
        'callback definition, check the configuration file'
    )

    nested_dict = kwargs[nested_item]

    nested_dict_name = next(iter(nested_dict.keys()))
    nested_kwargs = nested_dict[nested_dict_name]

    nested_class = get_class_by_name(nested_dict_name)

    for nest in nest_list:
        recurse_callbacks(nest, nested_kwargs, env, tensorboard_log)

    kwargs.update({
        nested_item: nested_class(**nested_kwargs)
    })


def load_cmvae(cmvae_global_config: Dict[str, Any], weights_path: str = None) -> keras.models.Model:
    if cmvae_global_config['latent_space_constraints']:
        cmvae = CmvaeDirect(n_z=cmvae_global_config['n_z'], img_res=cmvae_global_config['img_res'])
    else:
        cmvae = Cmvae(n_z=cmvae_global_config['n_z'], gate_dim=3)

    if weights_path is not None:
        cmvae.load_weights(weights_path).expect_partial()
        cmvae.img_res = tuple(cmvae.img_res)

    if cmvae_global_config['deterministic']:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        if len(tf.config.list_logical_devices('GPU')) > 0:
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    return cmvae


def load_cmvae_training_config(project_dir: str) -> Dict[str, Any]:
    with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_training_config.yml'), 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_cmvae_inference_config(project_dir: str) -> Dict[str, Any]:
    with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_inference_config.yml'), 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_hyperparams(project_dir: str, algorithm_name: str, environment_name: str, seed: int = None) -> Dict[str, Any]:
    print('Loading hyperparameters ...')
    config_dir = os.path.join(project_dir, 'configs', 'hyperparams', f'{algorithm_name.lower()}.yml')

    assert os.path.exists(config_dir) and os.path.isfile(config_dir), f'Must provide a valid path to an algorithm config file: {config_dir} does not exist'

    with open(config_dir, 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)[environment_name]
        if isinstance(hyperparams['train_freq'], List):
            hyperparams['train_freq'] = tuple(hyperparams['train_freq'])

        hyperparams['train_freq'] = convert_train_freq(hyperparams['train_freq'])

        hyperparams.update({
            'seed': seed
        })
        return hyperparams


def load_environment_config(project_dir: str) -> Dict[str, Any]:
    print('Loading environment configuration ...')
    with open(os.path.join(project_dir, 'configs', 'env_config.yml'), 'r') as f:
        env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        return env_config


# Braces and belts, could optimize by calling each seeding function per module but
# would likely cause future issues, so we can simply call the helper functions
def tensorflow_seeding(seed: int) -> None:
    if seed is None:
        return

    assert isinstance(seed, int), f'{seed} is not a valid seed, please provide a valid integer'
    # Tensorflow seeding (random.seed(seed), np.random.seed(seed), tf.random.set_seed(seed))
    tf.keras.utils.set_random_seed(seed)


def load_new_model(env: gymnasium.Env, algorithm_name: str, hyperparams: Dict[str, Any] = None) -> BaseAlgorithm:
    print('Training from scratch: initialising new model ...')
    assert algorithm_name in ALGOS, f'Algorithm {algorithm_name} is not supported, please choose from {ALGOS}'
    return ALGOS[algorithm_name](env=env, **hyperparams)


def load_pretrained_model(env: gymnasium.Env, algorithm_name: str, model_path: str, hyperparams: Dict[str, Any] = None):
    print('Loading pretrained agent ...')
    assert algorithm_name in ALGOS, f'Algorithm {algorithm_name} is not supported, please choose from {ALGOS}'

    assert os.path.isfile(model_path) and model_path.endswith('.zip'), 'The argument model_path_train must be a valid path to a .zip file'
    if hyperparams is not None:
        del hyperparams['policy']  # network architecture already set so don't need
        model = ALGOS[algorithm_name].load(path=model_path, env=env, **hyperparams)
    else:
        model = ALGOS[algorithm_name].load(path=model_path, env=env)

    return model


def load_callbacks(project_dir: str, env: gymnasium.Env, tensorboard_log: str) -> List[BaseCallback]:
    with open(os.path.join(project_dir, 'configs', 'callbacks.yml'), 'r') as f:
        callback_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)[ENVIRONMENT_TO_LOAD]
        return get_callback_list(callback_list=callback_wrapper_config, env=env, tensorboard_log=tensorboard_log)


def duplicate_directory(src_dir: str, dst_dir: str, files_to_exclude: List[str] = None, dirs_to_exclude: List[str] = None):
    if files_to_exclude is None:
        files_to_exclude = []
    if dirs_to_exclude is None:
        dirs_to_exclude = []

    def _exclude_func(directory, contents):
        excluded_files = set(files_to_exclude)
        excluded_dirs = set(dirs_to_exclude)

        return [content for content in contents if content in excluded_files or content in excluded_dirs and os.path.isdir(os.path.join(directory, content))]

    try:
        shutil.copytree(src_dir, dst_dir, ignore=_exclude_func)
    except FileExistsError:
        raise FileExistsError(f'Destination directory "{dst_dir}" already exists.')


def save_configs(configs: Dict[str, Dict[str, Any]]) -> None:
    for key, value in configs.items():
        with open(key, 'w') as file:
            if key.endswith('.yml'):
                yaml.dump(value, file)
            elif key.endswith('.json'):
                json.dump(value, file, indent=2)


def output_devices(output_dir: str, tensorflow_device=False, torch_device=False):
    with open(os.path.join(output_dir, 'devices.txt'), 'w') as file:
        if tensorflow_device:
            file.write('TENSORFLOW\n')
            for device_name in tf.config.list_physical_devices():
                file.write(device_name.name + "\n")
        if torch_device:
            file.write('PYTORCH\n')
            for i in range(torch.cuda.device_count()):
                file.write(torch.cuda.get_device_name(i) + "\n")


def count_files_in_directory(directory: str):
    file_count = 0

    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            file_count += 1

    return file_count


def count_directories_in_directory(directory: str):
    directory_count = 0

    # Iterate through all items in the directory
    for item in os.listdir(directory):
        # Check if it's a directory
        if os.path.isdir(os.path.join(directory, item)):
            directory_count += 1

    return directory_count


def parse_command_args(env_config: Dict[str, Any], cmvae_inference_config=None) -> None:
    parser = argparse.ArgumentParser(description='Process a seed parameter.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed', required=False)
    parser.add_argument('--algorithm', type=str, default=None, help='Reinforcement Learning Algorithm (choose from "sac", "ppo" or "ddpg")', required=False)
    parser.add_argument('--weights_path', type=str, default=None, help='Path to cmvae weights', required=False)
    args = parser.parse_args()

    if args.seed is not None:
        env_config['seed'] = args.seed

    if args.algorithm is not None:
        env_config['algorithm'] = args.algorithm

    if args.weights_path is not None:
        cmvae_inference_config['weights_path'] = args.weights_path
