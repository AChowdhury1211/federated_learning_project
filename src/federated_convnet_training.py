import numpy as np
from tensorflow.keras.datasets import mnist
from utilities.utils import Dataset, Model, Preprocess, Metrics, create_dir, save, plot
import utilities.config as config
from utilities.train import Training

np.random.seed(0)

if __name__ == "__main__":
   
    model_dir, result_dir = create_dir('saved_models', config.method_name)

   
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    dataset_ = Dataset(x_tr, y_tr, x_te, y_te)
    preprocess_ = Preprocess(dataset_)
    preprocessed_normal_dataset, preprocessed_fl_dataset_ = preprocess_.preprocess_federated_dataset()

  
    model = Model(config.INPUT_SHAPE, config.NUM_CLASSES).build_convnet_model()
    metrics_: Metrics = Training(model, preprocessed_normal_dataset, preprocessed_fl_dataset_, model_dir, config.STRATEGY).train()

   
    plot(list(range(1, 101)), metrics_.train_accuracy, config.PLOT_TITLE, "Train accuracy points", "Train accuracy")
    plot(list(range(1, 101)), metrics_.validation_accuracy, config.PLOT_TITLE, "Validation accuracy points", "Validation accuracy")
    save(metrics_, result_dir)