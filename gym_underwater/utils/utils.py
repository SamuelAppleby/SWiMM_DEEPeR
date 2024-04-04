import importlib
import os
import shutil
from typing import Dict, Type, Any, Optional, Callable, List, Union, Tuple

import gymnasium
import numpy as np
import tensorflow as tf
import yaml
from gymnasium import Env

from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import should_collect_more_steps, set_random_seed
from stable_baselines3.common.vec_env import VecEnv

from cmvae_models.cmvae import CmvaeDirect, Cmvae
from gym_underwater.enums import Protocol
from gym_underwater.gym_env import UnderwaterEnv

ENVIRONMENT_TO_LOAD = 'UnderwaterEnv'


def make_env(cmvae, obs, opt_d, max_d, img_res, tensorboard_log, project_dir, protocol=Protocol.TCP, host='127.0.0.1:60260', seed=None, inference_only=False) -> Env:
    """
    Makes instance of environment, seeds and wraps with Monitor
    """
    uenv = UnderwaterEnv(cmvae, obs, opt_d, max_d, img_res, tensorboard_log, protocol, host, seed)
    with open(os.path.join(project_dir, 'configs', 'env_wrapper.yml'), 'r') as f:
        env_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)[ENVIRONMENT_TO_LOAD]
        env_wrapper = get_wrapper_class(env_wrapper_config, tensorboard_log=tensorboard_log, inference_only=inference_only)
        if env_wrapper is not None:
            env = env_wrapper(uenv)

    return env


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
def get_wrapper_class(wrapper_list: List[Any], tensorboard_log: str = None, inference_only: bool = False) -> Optional[Callable[[gymnasium.Env], gymnasium.Env]]:
    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

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
                'filename': tensorboard_log,
                'inference_only': inference_only
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


def get_callback_list(callback_list: List[Any], env: gymnasium.Env, tensorboard_log: str = None, inference_only: bool = False) -> List[BaseCallback]:
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
                    if not inference_only:
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
                    else:
                        del kwargs[nested]

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


# Code adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py#L337. This is to fix the issue where we
# have stagnant observations due to the interruption of the rollout collection.
def should_collect_more_steps_vec(
        train_freq: TrainFreq,
        num_collected_steps: np.ndarray[int],
        num_collected_episodes: np.ndarray[int],
        count_targets: np.ndarray[int]
) -> bool:
    """
    Helper used in ``collect_rollouts()`` of off-policy algorithms
    to determine the termination condition.

    :param train_freq: How much experience should be collected before updating the policy.
    :param num_collected_steps: The number of already collected steps.
    :param num_collected_episodes: The number of already collected episodes.
    :param count_targets: The number of targets to collect.
    :return: Whether to continue or not collecting experience
        by doing rollouts of the current policy.
    """
    if train_freq.unit == TrainFrequencyUnit.STEP:
        return bool((num_collected_steps < count_targets).any())

    elif train_freq.unit == TrainFrequencyUnit.EPISODE:
        return bool((num_collected_episodes < count_targets).any())

    else:
        raise ValueError(
            "The unit of the `train_freq` must be either TrainFrequencyUnit.STEP "
            f"or TrainFrequencyUnit.EPISODE not '{train_freq.unit}'!"
        )


def evaluate_policy(
        model: "type_aliases.PolicyPredictor",
        env: Union[gymnasium.Env, VecEnv],
        eval_inference_freq: TrainFreq = TrainFreq(1, TrainFrequencyUnit.EPISODE),
        deterministic: bool = False,
        render: bool = False,
        callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
        warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param eval_inference_freq: Number of episode/steps to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Returns ([float], [int]), first list containing per-episode rewards and
        second containing per-episode lengths(in number of steps).
    """
    if not isinstance(env, VecEnv):
        print('[evaluate_policy] Wrapping the env in a DummyVecEnv')
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    step_counts = np.zeros(n_envs, dtype='int')
    episode_counts = np.zeros(n_envs, dtype='int')

    count_targets = np.array([(eval_inference_freq.frequency + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype='int')

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while should_collect_more_steps_vec(eval_inference_freq, step_counts, episode_counts, count_targets):
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic
        )

        new_observations, rewards, dones, infos = env.step(actions)

        current_rewards += rewards
        current_lengths += 1

        step_counts += 1
        for i in range(n_envs):
            if should_collect_more_steps(eval_inference_freq, step_counts[i] - 1, episode_counts[i]):
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                # Even if wrapped with a SwimMonitor, we cannot use the monitor values as we supress logging for evaluation episodes
                if dones[i] or ((eval_inference_freq.unit == TrainFrequencyUnit.STEP) and (step_counts[i] == eval_inference_freq.frequency)):
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    current_rewards[i] = 0
                    current_lengths[i] = 0

                    print('[INFERENCE] Episode finished. \nReward: {:.2f} \nSteps: {}'.format(episode_rewards[-1], episode_lengths[-1]))
                    episode_counts[i] += 1

        observations = new_observations

        if render:
            env.render()

    return episode_rewards, episode_lengths


def load_cmvae_config(project_dir, load_weights=False, seed=None) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    print('Loading CMVAE ...')
    with open(os.path.join(project_dir, 'configs', 'cmvae_config.yml'), 'r') as f:
        cmvae_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        if cmvae_config['latent_space_constraints']:
            cmvae = CmvaeDirect(n_z=cmvae_config['n_z'], seed=seed)
        else:
            cmvae = Cmvae(n_z=cmvae_config['n_z'], gate_dim=3, seed=seed)

        # TODO Investigate saving the model using save() and then we can avoid below (there must have been a reason)
        if load_weights:
            cmvae.load_weights(cmvae_config['weights_path']).expect_partial()

        if cmvae_config['use_cpu']:
            os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

        if cmvae_config['deterministic']:
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            if len(tf.config.list_physical_devices('GPU')) > 0:
                os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

        return cmvae, cmvae_config


def load_hyperparams(project_dir, algorithm_name, environment_name, seed=None) -> Dict[str, Any]:
    print('Loading hyperparameters ...')
    with open(os.path.join(project_dir, 'configs', 'hyperparams', '{}.yml'.format(algorithm_name)), 'r') as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)[environment_name]
        if isinstance(hyperparams['train_freq'], List):
            hyperparams['train_freq'] = tuple(hyperparams['train_freq'])

        hyperparams['train_freq'] = convert_train_freq(hyperparams['train_freq'])

        hyperparams.update({
            'seed': seed
        })
        return hyperparams


def load_environment_config(project_dir, seed_tensorflow=True, seed_sb=True) -> Dict[str, Any]:
    print('Loading environment configuration ...')
    with open(os.path.join(project_dir, 'configs', 'env_config.yml'), 'r') as f:
        env_config = yaml.load(f, Loader=yaml.UnsafeLoader)
        if env_config['seed'] is not None:
            global_seeding(env_config['seed'], tensorflow=seed_tensorflow, sb=seed_sb)
        return env_config


# Braces and belts, could optimize by calling each seeding function per module but
# would likely cause future issues, so we can simply call the helper functions
def global_seeding(seed, tensorflow=True, sb=True) -> None:
    assert isinstance(seed, int), f'{seed} is not a valid seed, please provide a valid integer'
    if tensorflow:
        # Tensorflow seeding (random.seed(seed), np.random.seed(seed), tf.random.set_seed(seed))
        tf.keras.utils.set_random_seed(seed)
    if sb:
        # SB3 seeding (random.seed(seed), np.random.seed(seed), th.manual_seed(seed))
        set_random_seed(seed, True)


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


def load_callbacks(project_dir, env, tensorboard_log, inference_only=False) -> List[BaseCallback]:
    with open(os.path.join(project_dir, 'configs', 'callbacks.yml'), 'r') as f:
        callback_wrapper_config = yaml.load(f, Loader=yaml.UnsafeLoader)[ENVIRONMENT_TO_LOAD]
        return get_callback_list(callback_wrapper_config, env, tensorboard_log=tensorboard_log, inference_only=inference_only)


def duplicate_directory(src_dir, dst_dir):
    try:
        shutil.copytree(src_dir, dst_dir)
    except FileExistsError:
        raise FileExistsError(f'Destination directory "{dst_dir}" already exists.')
