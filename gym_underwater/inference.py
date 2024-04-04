"""
Parent script for initiating an inference run
"""
import os

from stable_baselines3.common.utils import configure_logger

from gym_underwater.callbacks import SwimEvalCallback
from gym_underwater.utils.utils import make_env, load_model, load_callbacks, load_environment_config, load_cmvae_config
from gym_underwater.args import args

par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

env_config = load_environment_config(par_dir, seed_tensorflow=True, seed_sb=True)

assert os.path.isfile(env_config['model_path_inference']) and env_config['model_path_inference'].endswith('.zip'), 'The argument model_path_inference must be a valid path to a .zip file'

assert env_config['obs'] == 'cmvae', 'For inference, must provide a valid cmvae path'

cmvae, cmvae_config = load_cmvae_config(par_dir, True, env_config['seed'])

# Define the logger first to avoid reduplicating code caused by the file search in learn()
logger = configure_logger(verbose=1, tensorboard_log=str(os.path.join(par_dir, 'logs', env_config['algo'])), tb_log_name=env_config['algo'], reset_num_timesteps=True)

# Also performs environment wrapping
env = make_env(cmvae, env_config['obs'], env_config['opt_d'], env_config['max_d'], env_config['img_res'] if cmvae is None else cmvae_config['img_res'],
               logger.dir if env_config['debug_logs'] else None, par_dir, args.protocol, args.host, env_config['seed'], inference_only=True)

model = load_model(env, env_config['algo'], env_config['model_path_inference'])
model.set_logger(logger)

callbacks = load_callbacks(par_dir, env, logger.dir, inference_only=True)
eval_callback = list(filter(lambda x: isinstance(x, SwimEvalCallback), callbacks))
assert len(eval_callback) == 1, 'When running inference you must provide a SwimEvalCallback for evaluation'

model.env.envs[0].unwrapped.wait_until_client_ready()

# We have to manually initialise the callbacks as we want to ensure a consistent flow across
# training and evaluation, but callbacks are only initialised during setup_learn
callback = model._init_callback(callbacks, False)
eval_callback[0].evaluate(inference_only=True)

model.env.close()
