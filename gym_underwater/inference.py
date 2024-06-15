"""
Parent script for initiating an inference run
"""
import os

from stable_baselines3.common.utils import configure_logger

from gym_underwater.callbacks import SwimEvalCallback
from gym_underwater.constants import IP_HOST, PORT_INFERENCE
from gym_underwater.enums import Protocol
from gym_underwater.utils.utils import make_env, load_model, load_callbacks, load_environment_config, load_cmvae_inference_config, load_cmvae_global_config, output_devices, duplicate_directory

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_config = load_environment_config(project_dir, seed_tensorflow=True, seed_sb=True)

assert os.path.isfile(env_config['model_path_inference']) and env_config['model_path_inference'].endswith('.zip'), 'The argument model_path_inference must be a valid path to a .zip file'

assert env_config['obs'] == 'cmvae', 'For inference, must provide a valid cmvae path'

cmvae_inference_config = load_cmvae_inference_config(project_dir)
cmvae, _ = load_cmvae_global_config(project_dir, weights_path=cmvae_inference_config['weights_path'], seed=env_config['seed'])

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=os.path.join(os.path.dirname(env_config['model_path_inference']), 'inference'), tb_log_name=f'{env_config["algo"]}', reset_num_timesteps=True)

# Also performs environment wrapping
env = make_env(cmvae=cmvae, obs=env_config['obs'], opt_d=env_config['opt_d'], max_d=env_config['max_d'], img_res=env_config['img_res'], tensorboard_log=logger.dir, debug_logs=env_config['debug_logs'], protocol=Protocol.TCP, ip=IP_HOST, port=PORT_INFERENCE, seed=env_config['seed'], cancel_event=None, read_write_thread_other=None)

model = load_model(env, env_config['algo'], env_config['model_path_inference'])
model.set_logger(logger)

callbacks = load_callbacks(project_dir, env, logger.dir)
eval_callback = list(filter(lambda x: isinstance(x, SwimEvalCallback), callbacks))
assert len(eval_callback) == 1, 'When running inference you must provide a SwimEvalCallback for evaluation'

model.env.envs[0].unwrapped.wait_until_client_ready()

# We have to manually initialise the callbacks as we want to ensure a consistent flow across
# training and evaluation, but callbacks are only initialised during setup_learn
callback = model._init_callback(callbacks, False)
eval_callback[0].evaluate()

dirs_to_exclude = ['hyperparams']
files_to_exclude = ['cmvae_training_config.yml', 'cmvae_global_config.yml']
duplicate_directory(os.path.join(project_dir, 'configs'), os.path.join(logger.dir, 'configs'), dirs_to_exclude=dirs_to_exclude, files_to_exclude=files_to_exclude)

output_devices(logger.dir, tensorflow_device=True, torch_device=True)

model.env.close()
