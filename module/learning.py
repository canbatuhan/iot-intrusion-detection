import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Input, ZeroPadding1D
from tensorflow.keras.layers import MaxPooling1D, Add, AveragePooling1D
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from keras.initializers import glorot_uniform, he_normal
from tensorflow.keras.optimizers import Adam
from keras.metrics import Recall, Precision
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

class DeepLearningPipeline:
    def __init__(self, dataframe, label_name):
        X = dataframe.drop([label_name], axis=1).to_numpy()
        Y = dataframe[label_name].to_numpy()

        label_encoder = LabelEncoder().fit(Y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train)
        del X
        del Y

        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_val   = self.X_val.reshape(self.X_val.shape[0], self.X_val.shape[1], 1)
        self.X_test  = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        self.num_classes = len(np.unique(self.y_train))
        self.input_shape = self.X_train.shape[1:]
        
        self.y_train = label_encoder.transform(self.y_train)
        self.y_train = to_categorical(self.y_train, num_classes=self.num_classes)
    
        self.y_val  = label_encoder.transform(self.y_val)
        self.y_val = to_categorical(self.y_val, num_classes=self.num_classes)

        self.y_test  = label_encoder.transform(self.y_test)
        self.y_test = to_categorical(self.y_test, num_classes=self.num_classes)
        
        self.model = Sequential()
        self.model.add(Conv1D(32, 3, activation='relu', padding='same', input_shape=self.input_shape))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(64, 3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Conv1D(128, 3, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(2))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def build(self):
        opt = Adam(learning_rate=0.001)
        self.model.compile(optimizer=opt, loss=tf.keras.metrics.categorical_crossentropy, 
                           metrics=['accuracy', Recall(), Precision(), f1_score])

    def train(self, num_epochs, batch_size):
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
        lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode="min", verbose=1, min_lr=0)

        callbacks = [early_stopping, lr_reduce]
        self.history = self.model.fit(self.X_train, self.y_train,
                                      validation_data = (self.X_val, self.y_val),
                                      epochs          = num_epochs,
                                      callbacks       = callbacks,
                                      batch_size      = batch_size)

    def evaluate(self):
        y_hat = self.model.predict(self.X_test)
        y_hat = np.argmax(y_hat, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        y_true_ohe = to_categorical(y_true, num_classes=self.num_classes)
        y_hat_ohe  = to_categorical(y_hat, num_classes=self.num_classes)
        return y_true_ohe, y_hat_ohe