"""
Parent script for initiating a training run
"""
import os

from stable_baselines3.common.utils import constant_fn, configure_logger

from gym_underwater.constants import IP_HOST, PORT_TRAIN
from gym_underwater.enums import Protocol
from gym_underwater.utils.utils import make_env, middle_drop, accelerated_schedule, linear_schedule, load_cmvae_global_config, load_environment_config, load_hyperparams, load_model, load_callbacks, \
    ENVIRONMENT_TO_LOAD, load_cmvae_inference_config, output_devices, duplicate_directory

from stable_baselines3 import SAC, PPO

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_config = load_environment_config(project_dir, seed_tensorflow=True, seed_sb=True)
hyperparams = load_hyperparams(project_dir, env_config['algo'], ENVIRONMENT_TO_LOAD, env_config['seed'])

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

assert env_config['obs'] == 'cmvae', 'For training, must provide a valid cmvae path'

cmvae_inference_config = load_cmvae_inference_config(project_dir)
cmvae, _ = load_cmvae_global_config(project_dir, weights_path=cmvae_inference_config['weights_path'], seed=env_config['seed'])

kwargs = {'total_timesteps': hyperparams['total_timesteps'], 'log_interval': hyperparams['log_interval'], 'tb_log_name': env_config['algo'], 'reset_num_timesteps': True}
del hyperparams['total_timesteps']
del hyperparams['log_interval']

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=str(os.path.join(project_dir, 'models', env_config['algo'])), tb_log_name=env_config['algo'], reset_num_timesteps=kwargs['reset_num_timesteps'])
hyperparams.update({
    'tensorboard_log': logger.dir
})

exe_args = ['ip', IP_HOST, 'port', str(PORT_TRAIN), 'modeServerControl', 'debugLogs']

# Also performs environment wrapping
env = make_env(cmvae=cmvae, obs=env_config['obs'], opt_d=env_config['opt_d'], max_d=env_config['max_d'], img_res=env_config['img_res'], tensorboard_log=hyperparams['tensorboard_log'] if env_config['debug_logs'] else None, protocol=Protocol.TCP, ip=IP_HOST, port=PORT_TRAIN, seed=env_config['seed'], exe_args=exe_args, cancel_event=None, read_write_thread_other=None)
env.unwrapped.wait_until_client_ready()

# If model_path_train is None, will load a new agent
model = load_model(env, env_config['algo'], env_config['model_path_train'], hyperparams)
model.set_logger(logger)

callbacks = load_callbacks(project_dir, env, hyperparams['tensorboard_log'])

kwargs.update({
    'callback': callbacks
})

print('Starting training run...')
model.learn(**kwargs)
model.save(os.path.join(str(model.tensorboard_log), 'final_model'))

duplicate_directory(os.path.join(project_dir, 'configs'), os.path.join(hyperparams['tensorboard_log'], 'configs'), dirs_to_exclude=None, files_to_exclude=['cmvae_training_config.yml', 'cmvae_global_config.yml'])

output_devices(os.path.join(hyperparams['tensorboard_log'], 'configs'), tensorflow_device=True, torch_device=True)

model.env.close()
