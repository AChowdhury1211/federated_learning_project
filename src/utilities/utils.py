import os
import numpy as np
from typing import Any, List, Tuple
from dataclasses import dataclass, field
import tensorflow_federated as tff
from tensorflow import reshape
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, losses, metrics
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import pickle
from collections import OrderedDict
import config

@dataclass
class Dataset:
    
    x_train: np.ndarray = None
    y_train: np.array = None
    x_test: np.ndarray = None
    y_test: np.array = None

@dataclass
class FederatedDataset:
    
    federated_train_data: Any = None
    processed_data: Any = None

@dataclass
class Metrics:
    
    train_accuracy: list = field(default_factory=lambda: [])
    train_loss: list = field(default_factory=lambda: [])
    validation_accuracy: list = field(default_factory=lambda: [])
    validation_loss: list = field(default_factory=lambda: [])

@dataclass
class Preprocess:
    
    dataset_: Dataset
    federatedDataset_: FederatedDataset = None
    img_rows: int = 28
    img_cols: int = 28

    def preprocess_normal_dataset(self) -> Dataset:
        
        x_train = self.dataset_.x_train.reshape(self.dataset_.x_train.shape[0], 
                                                self.img_rows, self.img_cols, 1)
        x_test = self.dataset_.x_test.reshape(self.dataset_.x_test.shape[0], 
                                                self.img_rows, self.img_cols, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = self.dataset_.y_train.astype(np.int32).reshape(self.dataset_.x_train.shape[0], 1)
        y_test = self.dataset_.y_test.astype(np.int32).reshape(self.dataset_.x_test.shape[0], 1)
        return Dataset(x_train, y_train, x_test, y_test)

    def preprocess_federated_dataset(self) -> Tuple[Dataset, FederatedDataset]:
        
        images_per_set = int(np.floor(len(self.dataset_.x_train)/config.SPLIT))
        train_dataset_dict = OrderedDict()

        for i in range(1, config.SPLIT+1):
            client_name = "client_" + str(i)
            start_point = images_per_set * (i-1)
            end_point = images_per_set * i
            data = OrderedDict((('label', self.dataset_.y_train[start_point:end_point]), 
                                            ('pixels', self.dataset_.x_train[start_point:end_point])))
            train_dataset_dict[client_name] = data

        train_dataset_tensor = tff.simulation.FromTensorSlicesClientData(train_dataset_dict)
        tf_train_dataset: tf.data.Dataset = train_dataset_tensor.create_tf_dataset_for_client(
                                            train_dataset_tensor.client_ids[0])

        def process(dataset: tf.data.Dataset) -> Any:
            def flatten_batch(element) -> OrderedDict:
                
                return OrderedDict(
                    x=reshape(element['pixels'], [-1, 28, 28, 1]),
                    y=reshape(element['label'], [-1, 1]))

            return dataset.repeat(config.NUM_EPOCHS_FL).shuffle(images_per_set).batch(
                   config.BATCH_SIZE_FL).map(flatten_batch).prefetch(config.PREFETCH_BUFFER_FL)

        processed_dataset = process(tf_train_dataset)
        federated_train_data = process(train_dataset_tensor.create_tf_dataset_for_client(x) 
                                        for x in train_dataset_tensor.client_ids)
        normal_dataset: Dataset = self.preprocess_normal_dataset()
        
        return normal_dataset, FederatedDataset(federated_train_data, processed_dataset)

@dataclass
class Model:
   
    input_shape: Tuple[int]
    num_classes: int

    def build_convnet_model(self) -> Sequential:
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=self.input_shape))
        model.add(tfa.layers.GroupNormalization(groups=8, axis=3))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(tfa.layers.GroupNormalization(groups=8, axis=3))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.summary()
        return model

def create_dir(sub_dir: str, method_name: str) -> Tuple[str, str]:
    
    this_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
    model_dir = os.path.join(parent_dir, sub_dir, method_name)
    result_dir = os.path.join(sub_dir, method_name)

    try:
        os.makedirs(model_dir)
        os.makedirs(result_dir)
    except OSError as error:
        print(error) 

    return model_dir, result_dir

def init_function(model: Model, 
                  processed_dataset: FederatedDataset) -> tff.computation:
    
    return tff.learning.from_keras_model(model,
                        input_spec=processed_dataset.element_spec,
                        loss=losses.SparseCategoricalCrossentropy(),
                        metrics=[metrics.SparseCategoricalAccuracy()])

def utility(model: Model, state: tff.computation, iterative_process: Any, 
            federatedTrainData: FederatedDataset, 
            x_test: Dataset, y_test: Dataset) -> Metrics:
    
    for round_num in range(1, config.NUM_ROUNDS+1):
            print(f"Round: {round_num}:")
            state, metrics = iterative_process.next(state, federatedTrainData)
            tff_metric_list = list(metrics.items())
            tff_metrics = list(tff_metric_list[2][1].items())
            eval_metrics = evaluate(model, state, x_test, y_test)
            print(f"Metrics: {tff_metrics} \n Eval loss : {eval_metrics[0]} and Eval accuracy : {eval_metrics[1]}")
            train_accuracy = float(tff_metrics[0][1])
            train_loss = float(tff_metrics[1][1])
            val_loss, val_accuracy = eval_metrics
            return Metrics(train_accuracy, train_loss, val_accuracy, val_loss)

def evaluate(model: Model, state: tff.computation, x_test: Dataset, y_test: Dataset) -> List:
   
    model.compile(optimizer=optimizers.SGD(learning_rate = config.LEARNING_RATE),
                    loss=losses.SparseCategoricalCrossentropy(),
                    metrics=[metrics.SparseCategoricalAccuracy()])
    state.model.assign_weights_to(model)
    eval_metrics = model.evaluate(x_test, y_test, verbose=0)
    return eval_metrics

def plot(X: List, y: Metrics, title: str, label_1: str, label_2: str) -> None:
   
    fig = plt.figure(figsize=(13, 9))
    plt.title(title)
    plt.scatter(X, y, label = label_1)
    plt.plot(X, y, format='-', label = label_2)
    plt.xlabel("No of rounds")
    plt.ylabel("Values")
    plt.legend()

def save(metrics: Metrics, save_dir: str) -> None:
    
    for metric_name, values in metrics.__dict__.items():
        with open(os.path.join(save_dir, metric_name), "wb") as file:
            pickle.dump(values, file)
            print(f"Metrics saved as pickle file to {save_dir}")