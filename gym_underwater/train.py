"""
Parent script for initiating a training run
"""
import os

from stable_baselines3.common.utils import constant_fn, configure_logger

from gym_underwater.constants import IP_HOST, PORT_TRAIN
from gym_underwater.utils.utils import make_env, middle_drop, accelerated_schedule, linear_schedule, load_cmvae_global_config, load_environment_config, load_hyperparams, load_callbacks, \
    ENVIRONMENT_TO_LOAD, load_cmvae_inference_config, output_devices, parse_command_args, tensorflow_seeding, duplicate_directory, load_pretrained_model, load_new_model

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_config = load_environment_config(project_dir)
cmvae_inference_config = load_cmvae_inference_config(project_dir)

parse_command_args(env_config, cmvae_inference_config)

# NB Very important, _setup_model (for both on/off-policy algorithms) will call every seeding operation (see stable_baselines3.common.base_class.set_random_seed)
tensorflow_seeding(env_config['seed'])

# Also adds ['seed'] to hyperparams
hyperparams = load_hyperparams(project_dir, env_config['algorithm'], ENVIRONMENT_TO_LOAD, env_config['seed'])

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

cmvae, _ = load_cmvae_global_config(project_dir, weights_path=cmvae_inference_config['weights_path'])

kwargs = {'total_timesteps': hyperparams['total_timesteps'], 'log_interval': hyperparams['log_interval'], 'tb_log_name': env_config['algorithm'], 'reset_num_timesteps': True}
del hyperparams['total_timesteps']
del hyperparams['log_interval']

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=str(os.path.join(project_dir, 'models', env_config['algorithm'])), tb_log_name=env_config['algorithm'], reset_num_timesteps=kwargs['reset_num_timesteps'])
hyperparams.update({
    'tensorboard_log': logger.dir,
})

# Also performs environment wrapping
env = make_env(cmvae=cmvae, obs=env_config['obs'], img_res=env_config['img_res'], tensorboard_log=hyperparams['tensorboard_log'], debug_logs=env_config['debug_logs'], ip=IP_HOST, port=PORT_TRAIN, seed=env_config['seed'])

# If model_path_train is None, will load a new agent
model = load_pretrained_model(env, env_config['model_path_train'], hyperparams) if env_config['model_path_train'] is not None else load_new_model(env, env_config['algorithm'], hyperparams)
model.set_logger(logger)

callbacks = load_callbacks(project_dir, env, hyperparams['tensorboard_log'])

kwargs.update({
    'callback': callbacks
})

print('[TRAINING START]')

model.learn(**kwargs)
model.save(os.path.join(model.tensorboard_log, 'final_model'))

config_dir = os.path.join(hyperparams['tensorboard_log'], 'configs')
duplicate_directory(os.path.join(project_dir, 'configs'), config_dir, dirs_to_exclude=None, files_to_exclude=['cmvae_training_config.yml'])
output_devices(config_dir, tensorflow_device=True, torch_device=True)
