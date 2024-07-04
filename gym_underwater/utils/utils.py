import argparse
import importlib
import json
import os
import shutil
from typing import Dict, Type, Any, Optional, Callable, List, Tuple

import gymnasium
import numpy as np
import tensorflow as tf
import torch
import yaml

from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.monitor import Monitor

from cmvae_models.cmvae import CmvaeDirect, Cmvae
from gym_underwater.constants import ENVIRONMENT_TO_LOAD, IP_HOST, PORT_TRAIN, PORT_INFERENCE
from gym_underwater.enums import Protocol
from gym_underwater.gym_env import UnderwaterEnv


def make_env(cmvae, obs, img_res, tensorboard_log, debug_logs=False, protocol=Protocol.TCP, ip=IP_HOST, port=PORT_TRAIN, seed=None) -> gymnasium.Env:
    """
    Makes instance of environment, seeds and wraps with Monitor
    """
    uenv = UnderwaterEnv(obs=obs, img_res=img_res, tensorboard_log=tensorboard_log, debug_logs=debug_logs, protocol=protocol, ip=ip, port=port, seed=seed, cmvae=cmvae)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'configs', 'env_wrapper.yml'), 'r') as f:
        env_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)

        if env_wrapper_config is not None:
            uenv_conf = env_wrapper_config[ENVIRONMENT_TO_LOAD]
            env_wrapper = get_wrapper_class(uenv_conf,
                                            monitor_filename=os.path.join(tensorboard_log, 'training_monitor.csv') if port == PORT_TRAIN else os.path.join(tensorboard_log, 'evaluation_monitor.csv'))

            if env_wrapper is not None:
                uenv = env_wrapper(uenv)

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


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3
}


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


nested_keys = ['callback_on_new_best', 'callback_after_eval']
eval_callback_name = ''


# SwimCallback must be last, as that is the one restarting physics
def custom_callback_sort(item):
    if isinstance(item, dict):
        item = next(iter(item.keys()))
    match item:
        case 'gym_underwater.callbacks.SwimEvalCallback':
            return 0
        case 'gym_underwater.callbacks.SwimProgressBarCallback':
            return 1
        case 'gym_underwater.callbacks.SwimCallback':
            return np.inf
        case _:
            return 0


def get_callback_list(callback_list: List[Any], env: gymnasium.Env, tensorboard_log: str = None) -> List[BaseCallback]:
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
                                debug_logs=env.unwrapped.handler.debug_logs, protocol=env.unwrapped.handler.protocol, ip=IP_HOST, port=PORT_INFERENCE, seed=env.unwrapped.seed)

            eval_env.unwrapped.wait_until_client_ready()

            kwargs.update({
                'eval_env': eval_env,
                'log_path': tensorboard_log,
                'best_model_save_path': tensorboard_log
            })

            for freq in ['eval_freq', 'eval_inference_freq']:
                if isinstance(kwargs[freq], List):
                    kwargs[freq] = tuple(kwargs[freq])

            for nested in nested_keys:
                if nested in kwargs.keys():
                    assert isinstance(kwargs[nested], dict), (
                        f'There is an error within the {nested}'
                        'callback definition, check the configuration file'
                    )

                    nested_dict = kwargs[nested]

                    nested_dict_name = next(iter(nested_dict.keys()))
                    nested_kwargs = nested_dict[nested_dict_name]

                    nested_class = get_class_by_name(nested_dict_name)

                    kwargs.update({
                        nested: nested_class(**nested_kwargs)
                    })

        callbacks.append(callback_class(**kwargs))

    return callbacks


def convert_train_freq(train_freq) -> TrainFreq:
    """
    Convert `train_freq` parameter (int or tuple)
    to a TrainFreq object.
    """
    if isinstance(train_freq, TrainFreq):
        return train_freq

    # The value of the train frequency will be checked later
    if not isinstance(train_freq, tuple):
        train_freq = (train_freq, 'step')
    try:
        train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))  # type: ignore[assignment]
    except ValueError as e:
        raise ValueError(
            f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
        ) from e

    if not isinstance(train_freq[0], int):
        raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

    return TrainFreq(*train_freq)


def load_cmvae_global_config(project_dir, weights_path=None) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_global_config.yml'), 'r') as f:
        cmvae_global_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        if cmvae_global_config['latent_space_constraints']:
            cmvae = CmvaeDirect(n_z=cmvae_global_config['n_z'], img_res=cmvae_global_config['img_res'])
        else:
            cmvae = Cmvae(n_z=cmvae_global_config['n_z'], gate_dim=3)

        if weights_path is not None:
            cmvae.load_weights(weights_path).expect_partial()
            cmvae.img_res = tuple(cmvae.img_res)

        if cmvae_global_config['use_cpu_only']:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        if cmvae_global_config['deterministic']:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            if len(tf.config.list_physical_devices('GPU')) > 0:
                os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        return cmvae, cmvae_global_config


def load_cmvae_training_config(project_dir) -> Dict[str, Any]:
    with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_training_config.yml'), 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_cmvae_inference_config(project_dir) -> Dict[str, Any]:
    with open(os.path.join(project_dir, 'configs', 'cmvae', 'cmvae_inference_config.yml'), 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


def load_hyperparams(project_dir, algorithm_name, environment_name, seed=None) -> Dict[str, Any]:
    print('Loading hyperparameters ...')
    with open(os.path.join(project_dir, 'configs', 'hyperparams', f'{algorithm_name}.yml'), 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)[environment_name]
        if isinstance(hyperparams['train_freq'], List):
            hyperparams['train_freq'] = tuple(hyperparams['train_freq'])

        hyperparams['train_freq'] = convert_train_freq(hyperparams['train_freq'])

        hyperparams.update({
            'seed': seed
        })
        return hyperparams


def load_environment_config(project_dir) -> Dict[str, Any]:
    print('Loading environment configuration ...')
    with open(os.path.join(project_dir, 'configs', 'env_config.yml'), 'r') as f:
        env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        return env_config


# Braces and belts, could optimize by calling each seeding function per module but
# would likely cause future issues, so we can simply call the helper functions
def tensorflow_seeding(seed) -> None:
    if seed is None:
        return

    assert isinstance(seed, int), f'{seed} is not a valid seed, please provide a valid integer'
    # Tensorflow seeding (random.seed(seed), np.random.seed(seed), tf.random.set_seed(seed))
    tf.keras.utils.set_random_seed(seed)


def load_model(env, algorithm_name, model_path=None, hyperparams=None) -> BaseAlgorithm:
    if model_path is not None:
        print('Loading pretrained agent ...')
        assert os.path.isfile(model_path) and model_path.endswith('.zip'), 'The argument model_path_train must be a valid path to a .zip file'
        if hyperparams is not None:
            del hyperparams['policy']  # network architecture already set so don't need
            model = ALGOS[algorithm_name].load(path=model_path, env=env, **hyperparams)
        else:
            model = ALGOS[algorithm_name].load(path=model_path, env=env)
    else:
        print('Training from scratch: initialising new model ...')
        model = ALGOS[algorithm_name](env=env, **hyperparams)

    return model


def load_callbacks(project_dir: str, env: gymnasium.Env, tensorboard_log: str) -> List[BaseCallback]:
    with open(os.path.join(project_dir, 'configs', 'callbacks.yml'), 'r') as f:
        callback_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)[ENVIRONMENT_TO_LOAD]
        return get_callback_list(callback_list=callback_wrapper_config, env=env, tensorboard_log=tensorboard_log)


def duplicate_directory(src_dir, dst_dir, files_to_exclude=None, dirs_to_exclude=None):
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


def output_devices(output_dir, tensorflow_device=False, torch_device=False):
    with open(os.path.join(output_dir, 'devices.txt'), 'w') as file:
        if tensorflow_device:
            file.write('TENSORFLOW\n')
            for device_name in tf.config.list_physical_devices():
                file.write(device_name.name + "\n")
        if torch_device:
            file.write('PYTORCH\n')
            for i in range(torch.cuda.device_count()):
                file.write(torch.cuda.get_device_name(i) + "\n")


def count_files_in_directory(directory):
    file_count = 0

    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)):
            file_count += 1

    return file_count


def count_directories_in_directory(directory):
    directory_count = 0

    # Iterate through all items in the directory
    for item in os.listdir(directory):
        # Check if it's a directory
        if os.path.isdir(os.path.join(directory, item)):
            directory_count += 1

    return directory_count


def parse_command_args(env_config, cmvae_inference_config=None) -> None:
    parser = argparse.ArgumentParser(description='Process a seed parameter.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed', required=False)
    parser.add_argument('--weights_path', type=str, default=None, help='Path to cmvae weights', required=False)
    args = parser.parse_args()

    if args.seed is not None:
        env_config['seed'] = args.seed

    if args.weights_path is not None:
        cmvae_inference_config['weights_path'] = args.weights_path
