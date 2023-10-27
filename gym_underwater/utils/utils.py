from stable_baselines3.common.monitor import Monitor
from gym_underwater.gym_env import UnderwaterEnv


def make_env(vae, obs, opt_d, max_d, img_scale, debug_logs, protocol, host, log_d, ep_len_threshold, seed):
    """
    Makes instance of environment, seeds and wraps with Monitor
    """

    def _init():
        # create instance of environment
        env_inst = UnderwaterEnv(vae, obs, opt_d, max_d, img_scale, debug_logs, protocol, host, ep_len_threshold)
        print("Environment ready")
        if seed > 0:
            # seed the environment
            env_inst.seed(seed)
        # wrap environment with SB's Monitor wrapper
        wrapped_env = Monitor(env_inst, log_d, allow_early_resets=True)
        return wrapped_env

    return _init


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
