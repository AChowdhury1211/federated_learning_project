import os
import typing
import tensorflow as tf
from tensorflow.keras import optimizers, losses
import numpy as np
import tensorflow_federated as tff
import config
from utils import Model, Dataset, FederatedDataset, Metrics, init_function, utility

class Training:
    def __init__(self, model: Model, dataset: Dataset, federatedDataset: FederatedDataset, 
                model_dir: str, strategy: str) -> None:
       
        self.model = model
        self.dataset_ = dataset
        self.federatedDataset_ = federatedDataset
        self.model_dir = model_dir
        self.strategy = strategy

    def normal_convnet_training(self) -> Metrics:
       
        self.model.compile(loss=losses.sparse_categorical_crossentropy,
                        optimizer=optimizers.SGD(learning_rate=config.LEARNING_RATE),
                        metrics=['accuracy'])
        history = self.model.fit(self.dataset_.x_train, self.dataset_.y_train, 
                batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=1,
                validation_data=(self.dataset_.x_test, self.dataset_.y_test))
        train_accuracy = list(np.array(history.history['accuracy']))
        train_loss = list(np.array(history.history['loss']))
        val_accuracy = list(np.array(history.history['val_accuracy']))
        val_loss = list(np.array(history.history['val_loss']))
        save_path = os.path.join(config.model_dir, config.model_name)
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

        return Metrics(train_accuracy, train_loss, val_accuracy, val_loss)

    def federated_convnet_training_random(self) -> Metrics:
     
        init_fn = init_function(self.model, self.dataset_.processed_data)
        iterative_process = tff.learning.build_federated_averaging_process(
                    init_fn,
                    client_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    server_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    use_experimental_simulation_loop=True)
        
        state = iterative_process.initialize()
        print("FL Training with random initialization started...")
        metrics_instance: Metrics = utility(self.model, state, iterative_process, 
                                            self.federatedDataset_.federated_train_data,
                                            self.dataset_.x_test, self.dataset_.y_test)
        return metrics_instance

    def federated_convnet_training_pretrained(self) -> Metrics:
     
        pretrained_model = tf.keras.models.load_model(config.model_dir / config.model_name)
        init_fn = init_function(pretrained_model, self.dataset_.processed_data)
        iterative_process = tff.learning.build_federated_averaging_process(
                    init_fn,
                    client_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    server_optimizer_fn=lambda: optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    use_experimental_simulation_loop=True)
        
        state = iterative_process.initialize()
        
        state = tff.learning.state_with_new_model_weights(
                    state,
                    trainable_weights=[v.numpy() for v in pretrained_model.trainable_weights],
                    non_trainable_weights=[
                    v.numpy() for v in pretrained_model.non_trainable_weights])
        print("FL Training with pretrained models started...")
        metrics_instance: Metrics = utility(pretrained_model, state, iterative_process,
                                            self.federatedDataset_.federated_train_data,
                                            self.dataset_.x_test, self.dataset_.y_test)
        return metrics_instance

   
    TRAINING_STRATEGIES: dict = {
            "normal": normal_convnet_training,
            "fl_random": federated_convnet_training_random,
            "fl_pretrained": federated_convnet_training_pretrained,
            }

    def train(self) -> Metrics:
        return self.TRAINING_STRATEGIES[self.strategy]()