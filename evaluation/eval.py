import tensorflow as tf
import numpy as np
import logging
import wandb
from sklearn.metrics import ConfusionMatrixDisplay 
from matplotlib import pyplot as plt

from evaluation.metrics import Metrics

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    '''
    Evaluates a given model on a test dataset and logs relevant metrics.

    Parameters:
        model (tf.keras.Model): The trained model to evaluate.
        checkpoint (tf.train.Checkpoint): Checkpoint to restore the latest trained weights.
        ds_test (tf.data.Dataset): The test dataset.
        ds_info (dict): Information about the dataset.
        run_paths (dict): Dictionary containing paths for saving checkpoints.

    Returns:
        None.
    '''
    manager = tf.train.CheckpointManager(checkpoint, run_paths['path_ckpts_train'], max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    logging.info(f'Evaluation on {model.name} with {manager.latest_checkpoint}.')

    eval_metrics = Metrics("eval")
    eval_predictions = []
    eval_labels = []

    for idx, (images, labels) in enumerate(ds_test):
        predictions = eval_step(images, labels, model, eval_metrics)
        eval_predictions.append(predictions.numpy())
        eval_labels.append(labels.numpy())

    template = 'Evaluation: Accuracy: {}, Recall: {}, Precision: {}, F1: {}'
    logging.info(template.format(eval_metrics.result("accuracy") * 100,
                                 eval_metrics.result("recall"),
                                 eval_metrics.result("precision"),
                                 eval_metrics.result("f1")))
    
    # Write summary to wandb
    wandb.log({'acc': eval_metrics.result("accuracy") * 100,
                'recall': eval_metrics.result("recall"),
                'precision': eval_metrics.result("precision"),
                'f1': eval_metrics.result("f1")},
                commit=False)
    
    labels = np.concatenate(eval_labels)
    predictions = np.concatenate(eval_predictions)
    pred = np.eye(2)[predictions.reshape(-1)] # one hot encoding for roc and prc
    ConfusionMatrixDisplay.from_predictions(labels, predictions)

    wandb.log({"conf": wandb.plot.confusion_matrix(y_true=labels, preds=predictions, class_names=['0', '1']),
                "roc": wandb.plot.roc_curve(labels, pred),
                "Confusion matrix": wandb.Image(plt)})
    return None

@tf.function
def eval_step(images, labels, model, eval_metrics):
    '''
    Performs a single evaluation step on a batch of images.

    Parameters:
        images (tf.Tensor): A batch of input images.
        labels (tf.Tensor): Corresponding ground truth labels.
        model (tf.keras.Model): The trained model for evaluation.
        eval_metrics (Metrics): Object that updates evaluation metrics.

    Returns:
        tf.Tensor: The predicted labels for the batch.
    '''
    predictions = model(images, training=False)

    eval_metrics.update_acc(labels, predictions)

    predictions = tf.math.argmax(predictions, axis=1)      
    eval_metrics.update_state(labels, predictions)

    return predictions