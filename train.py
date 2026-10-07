import gin
import tensorflow as tf
import numpy as np
import logging
import wandb
from evaluation.metrics import  Metrics

@gin.configurable
class Trainer(object):
    '''
    A class for training and validating a deep learning model with logging and checkpointing.

        Attributes:
                model (tf.keras.Model): The neural network model.
                ds_train (tf.data.Dataset): Training dataset.
                ds_val (tf.data.Dataset): Validation dataset.
                ds_info (dict): Information about the dataset.
                run_paths (dict): Paths for logs, checkpoints, and configuration.
                lr (float): Initial learning rate.
                total_steps (int): Total number of training steps.
                log_interval (int): Steps between logging training progress.
                ckpt_interval (int): Steps between saving checkpoints.
                run (wandb.Run): Weights & Biases run instance.
    '''
    def __init__(self, model, ds_train, ds_val, ds_info, run_paths, lr, total_steps, log_interval, ckpt_interval, run):

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Metrics
        self.train_metrics = Metrics("train")
        self.train_predictions = []
        self.train_labels = []

        self.val_metrics = Metrics("val")
        self.val_predictions = []
        self.val_labels = []

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval 
        self.run = run     
        
        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer, step=tf.Variable(1))
        self.manager = tf.train.CheckpointManager(self.ckpt, run_paths['path_ckpts_train'], max_to_keep=1)

    @tf.function
    def train_step(self, images, labels):
        '''
        Performs a single training step.

        Parameters:
            images (tf.Tensor): Batch of training images.
            labels (tf.Tensor): Corresponding labels.

        Returns:
            tf.Tensor: Predicted class labels.
        '''
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_metrics.update_loss(loss)
        self.train_metrics.update_acc(labels, predictions)

        predictions = tf.math.argmax(predictions, axis=1)      
        self.train_metrics.update_state(labels, predictions)  
        return predictions

    @tf.function
    def val_step(self, images, labels):
        '''
        Performs a single validation step.

        Parameters:
            images (tf.Tensor): Batch of validation images.
            labels (tf.Tensor): Corresponding labels.

        Returns:
            tf.Tensor: Predicted class labels.
        '''
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_metrics.update_loss(t_loss)
        self.val_metrics.update_acc(labels, predictions)

        predictions = tf.math.argmax(predictions, axis=1)      
        self.val_metrics.update_state(labels, predictions)  
        return predictions

    def train(self):
        '''
        Executes the training loop with logging and checkpointing.

        Yields:
            float: Validation accuracy at each logging interval.
        '''
        # restoring latest checkpoint
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            logging.info(f'Restored checkpoint from {self.manager.latest_checkpoint}.')

        for (images, labels) in self.ds_train:
            self.ckpt.step.assign_add(1)
            step = int(self.ckpt.step)

            predictions = self.train_step(images, labels)
            self.train_predictions.append(predictions.numpy())
            self.train_labels.append(labels.numpy())

            if step % self.log_interval == 0:
                # Reset test metrics
                self.val_metrics.reset_states()
                self.val_predictions = []
                self.val_labels = []
                
                for val_images, val_labels in self.ds_val:
                    predictions = self.val_step(val_images, val_labels)
                    self.val_predictions.append(predictions.numpy())
                    self.val_labels.append(val_labels.numpy())

                template = 'Train: Step {}, Loss: {}, Accuracy: {}, Recall: {}, Precision: {}, F1: {}'
                logging.info(template.format(step,
                                             self.train_metrics.result("loss"),
                                             self.train_metrics.result("accuracy") * 100,
                                             self.train_metrics.result("recall"),
                                             self.train_metrics.result("precision"),
                                             self.train_metrics.result("f1")))
                template = 'Validation: Step {}, Loss: {}, Accuracy: {}, Recall: {}, Precision: {}, F1: {}'
                logging.info(template.format(step,
                                             self.val_metrics.result("loss"),
                                             self.val_metrics.result("accuracy") * 100,
                                             self.val_metrics.result("recall"),
                                             self.val_metrics.result("precision"),
                                             self.val_metrics.result("f1")))
                
                # Write summary to wandb
                wandb.log({'train/acc': self.train_metrics.result("accuracy") * 100,
                            'train/loss': self.train_metrics.result("loss"),
                            'train/recall': self.train_metrics.result("recall"),
                            'train/precision': self.train_metrics.result("precision"),
                            'train/f1': self.train_metrics.result("f1"),
                            'validate/acc': self.val_metrics.result("accuracy") * 100,
                            'validate/loss': self.val_metrics.result("loss"),
                            'validate/recall': self.val_metrics.result("recall"),
                            'validate/precision': self.val_metrics.result("precision"),
                            'validate/f1': self.val_metrics.result("f1")
                            }, 
                            commit=False)
                labels = np.concatenate(self.train_labels)
                predictions = np.concatenate(self.train_predictions)
                pred = np.eye(2)[predictions.reshape(-1)] # one hot encoding for roc and prc
                wandb.log({"train/conf": wandb.plot.confusion_matrix(y_true=labels, preds=predictions, class_names=['0', '1']),
                           "train/roc": wandb.plot.roc_curve(labels, pred)}, 
                           commit=False)

                labels = np.concatenate(self.val_labels)
                predictions = np.concatenate(self.val_predictions)
                pred = np.eye(2)[predictions.reshape(-1)] # one hot encoding for roc and prc
                wandb.log({"validate/conf": wandb.plot.confusion_matrix(y_true=labels, preds=predictions, class_names=['0', '1']),
                           "validate/roc": wandb.plot.roc_curve(labels, pred)})

                # Reset train metrics
                self.train_metrics.reset_states()
                self.train_predictions = []
                self.train_labels = []

                yield self.val_metrics.result("accuracy").numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.manager.save()
                artifact = wandb.Artifact(name="model-"+self.run.name, type='model')
                artifact.add_dir(local_path=self.run_paths["path_model_id"])
                self.run.log_artifact(artifact)
                return self.val_metrics.result("accuracy").numpy()
