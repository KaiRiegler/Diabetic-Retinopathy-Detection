import os
import gin
import logging
import wandb
import tensorflow as tf

from input_pipeline.datasets import load
from train import Trainer
from utils import utils_params, utils_misc
from main import evaluate, generate_grad_cam_heatmap


def train_func():
    '''
    Executes the training and evaluation pipeline with Weights & Biases integration for Sweeps.
    '''
    with wandb.init() as run:

        gin.clear_config()

        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')

        # generate folder structures
        run_paths = utils_params.gen_run_folder()

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings([os.path.join(os.path.dirname(__file__), 'configs/config.gin')], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())
        config = utils_params.config_to_dict(run_paths['path_gin'])

        # add config parameters to wandb config
        for key, value in config.items():
            run.config[key] = value

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = load()

        # model
        model = utils_misc.choose_model_to_train()

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, run=run)
        for _ in trainer.train():
            continue
        
        checkpoint = tf.train.Checkpoint(model=model)
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)
        generate_grad_cam_heatmap(model, ds_test)

sweep_config = {
    'name': 'sweep_10_effnetV2',
    'project': 'sweep_effnet_with_scaling_param', 
    'entity': 'your-entity',
    'method': 'grid',
    'metric': {
        'name': 'validate/acc',
        'goal': 'maximize'
    },
    'parameters': {
        # 'Trainer.lr': {
        #     'values': [0.00001, 0.000001, 0.001]
        # },
        # 'Trainer.total_steps': {
        #     'values': [500, 1000]
        # },
        # 'build_pretrained_model.unfreezed_layers': {
            # 'values': [0, 3, 121, 194, 237, 249, 261, 264, 267] 
            # 'values': [0, 3, 136, 224, 282, 301, 320, 328, 331]
            # 'values': [0, 3, 226, 357, 445, 471, 497, 508, 511] 
        'Trainer.total_steps': {
            'values': [6e2, 8e2, 1e3, 1.2e3, 1.4e3]
        },
        'EfficientNetV2.width_factor': {
            'values': [0.2, 0.4, 0.5, 0.6 ,0.8, 1.0]
        },
        'EfficientNetV2.depth_factor': {
            'values': [0.2, 0.4, 0.5, 0.6 ,0.8, 1.0]
        },
    }
}

# start sweep
os.environ["WANDB_MODE"] = "online"
sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train_func)
