import importlib
from typing import Dict, Type, Any, Optional, Callable, List

import gymnasium
import numpy as np

from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.monitor import Monitor

from gym_underwater.callbacks import SwimEvalCallback
from gym_underwater.enums import Protocol
from gym_underwater.gym_env import UnderwaterEnv


def make_env(cmvae, obs, opt_d, max_d, img_res, tensorboard_log, protocol=Protocol.TCP, host='127.0.0.1:60260', seed=None) -> UnderwaterEnv:
    """
    Makes instance of environment, seeds and wraps with Monitor
    """

    return UnderwaterEnv(cmvae, obs, opt_d, max_d, img_res, tensorboard_log, protocol, host, seed)


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


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3
}


# We require that SwimMonitor is the very last wrapper to execute, as it relies on values obtained from others (at the moment, SwimTimeLimit)
def custom_wrapper_sort(item):
    if isinstance(item, dict):
        item = next(iter(item.keys()))
    match item:
        case 'gym_underwater.env_wrappers.swim_time_limit.SwimTimeLimit':
            return 0
        case 'gym_underwater.env_wrappers.swim_monitor.SwimMonitor':
            return np.inf
        case _:
            return 0


# Adapted from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/utils.py#L47
def get_wrapper_class(hyperparams: Dict[str, Any], key: str = "env_wrapper", tensorboard_log: str = None) -> Optional[Callable[[gymnasium.Env], gymnasium.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    Works also for VecEnvWrapper with the key "vec_env_wrapper".

    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - rl_zoo3.wrappers.PlotActionWrapper
        - rl_zoo3.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if key in hyperparams.keys():
        wrapper_name = hyperparams.get(key)

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_names.sort(key=custom_wrapper_sort)

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
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

            if wrapper_class is Monitor:
                kwargs.update({
                    'filename': tensorboard_log
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
    else:
        return None


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


# We require that callbacks are sorted into a custom list, as those at the head have dependencies on the values of those further nested
# Specifically, if we want to inject some evaluation. we need to ensure that SwimEvalCallback is the last callback
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


def get_callback_list(hyperparams: Dict[str, Any], env: gymnasium.Env, key: str = "callback", tensorboard_log: str = None) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - rl_zoo3.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    callbacks: List[BaseCallback] = []

    if key in hyperparams.keys():
        callback_name = hyperparams.get(key)

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # We need to order our callbacks in a precedential manner
        callback_names.sort(key=custom_callback_sort)

        # Handle multiple wrappers
        for callback_name in callback_names:
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

            if (callback_class is EvalCallback) or (callback_class is SwimEvalCallback):
                kwargs.update({
                    'eval_env': env,
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
        train_freq = (train_freq, "step")
    try:
        train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))  # type: ignore[assignment]
    except ValueError as e:
        raise ValueError(
            f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
        ) from e

    if not isinstance(train_freq[0], int):
        raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

    return TrainFreq(*train_freq)
