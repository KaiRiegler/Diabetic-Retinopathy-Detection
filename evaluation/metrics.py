import tensorflow as tf

class Metrics():
    '''
    A class to compute and store various evaluation metrics. 
        
        Attributes:
                name (str): Name prefix for the metric group.
    '''
    def __init__(self, name):
        self.name = name
        self.metrics = {"loss": tf.keras.metrics.Mean(name=self.name+'_loss'),
                        "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(name=self.name+'_accuracy'),
                        "recall": tf.keras.metrics.Recall(name=self.name+'_recall'),
                        "precision": tf.keras.metrics.Precision(name=self.name+'_precision')
                        }
        
    def update_state(self, labels, predictions):
        '''
        Updates recall and precision metrics.

        Parameters:
            labels (tf.Tensor): True labels.
            predictions (tf.Tensor): Predicted labels.
        '''
        for k, m in self.metrics.items():
            if k != "loss" and k != "accuracy":
                m.update_state(labels, predictions)

    def update_loss(self, loss):
        '''
        Updates the loss metric.

        Parameters:
            loss (tf.Tensor): The computed loss value.
        '''
        self.metrics["loss"].update_state(loss)

    def update_acc(self, labels, predictions):
        '''
        Updates the accuracy metric.

        Parameters:
            labels (tf.Tensor): True labels.
            predictions (tf.Tensor): Predicted labels.
        '''
        self.metrics["accuracy"].update_state(labels, predictions)

    def result(self, metric):
        '''
        Retrieves the computed value for a specific metric.

        Parameters:
            metric (str): Name of the metric.

        Returns:
            tf.Tensor: Computed metric value.
        '''
        if metric == "f1":
            result = self.calc_f1_score()
        else:
            result = self.metrics[metric].result()
        return result
    
    def reset_states(self):
        '''
        Resets all tracked metrics.
        '''
        for k, m in self.metrics.items():
            m.reset_states()
        
    def calc_f1_score(self):
        '''
        Computes the F1 score using precision and recall.

        Returns:
            tf.Tensor: Computed F1 score.
        '''
        return 2 * (self.metrics["precision"].result() * self.metrics["recall"].result()) / (self.metrics["precision"].result() + self.metrics["recall"].result())     
    