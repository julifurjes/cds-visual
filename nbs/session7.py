# generic tools
import numpy as np

# tools from sklearn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
#from tensorflow.keras.utils import plot_model

# matplotlib
import matplotlib.pyplot as plt

def prepare_data():
    data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    # normalise data
    data = data.astype("float")/255.0
    # split data
    (X_train, X_test, y_train, y_test) = train_test_split(data,
                                                        labels, 
                                                        test_size=0.2)
    # convert labels to one-hot encoding
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    return X_train, X_test, y_train, y_test

def neural_network():
    # define architecture 784x256x128x10
    model = Sequential()
    model.add(Dense(256, 
                    input_shape=(784,), 
                    activation="relu"))
    model.add(Dense(128, 
                    activation="relu"))
    model.add(Dense(10, 
                    activation="softmax"))
    model.summary() # show summary
    #plot_model(model, show_shapes=True, show_layer_names=True)
    return model

def optimisation(main_model):
    # train model using SGD
    sgd = SGD(0.01)
    main_model.compile(loss="categorical_crossentropy", 
                  optimizer=sgd, 
                  metrics=["accuracy"])
    
def train_model(main_model, x_training, y_training, x_testing, y_testing):
    history = main_model.fit(x_training, y_training, 
                    validation_data=(x_testing, y_testing), 
                    epochs=10, 
                    batch_size=32)
    return history

def vis_plot(history_res):
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, 10), history_res.history["loss"], label="train_loss")
    plt.plot(np.arange(0, 10), history_res.history["val_loss"], label="val_loss", linestyle=":")
    plt.plot(np.arange(0, 10), history_res.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, 10), history_res.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.tight_layout()
    plt.legend()
    plt.show()
    
def classifier(main_model, x_testing, y_testing):
    # evaluate network
    print("[INFO] evaluating network...")
    predictions = model.predict(x_testing, batch_size=32)
    print(classification_report(y_testing.argmax(axis=1), 
                                predictions.argmax(axis=1), 
                                target_names=[str(x) for x in lb.classes_]))
    
def main():
    X_train, X_test, y_train, y_test = prepare_data()
    model = neural_network()
    optimisation(model)
    history = train_model(model, X_train, X_test, y_train, y_test)
    vis_plot(history)
    classifier(model, X_test, y_test)
    
main()
    