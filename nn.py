import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from config import *
import os
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if GPU: 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print("Reading file")
data = pd.read_hdf("data/data_ml.h5", key='table',mode='r')

train_features = TRAIN_FEATURES
print(train_features)

data_train = data.query(" gen_split == 'train' ")
data_val   = data.query(" gen_split == 'val'   ")

y_train = data_train['label'].values
y_val = data_val['label'].values

num_classes = len(data["sample"].unique())
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes)

X_train = data_train[train_features].values

print("Standard Scaler")
scaler = StandardScaler().fit(X_train)
with open(f'models/ae_scaler.pck','wb') as f:
    pickle.dump(scaler,f)

X_train = scaler.transform(data_train[train_features].values)

X_val = scaler.transform(data_val[train_features].values)
print(X_train.shape, X_val.shape)

print("Define NN and train")

model = Sequential()
model.add(Dense(15, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(15, activation='relu')),
model.add(Dense(15, activation='relu')),
model.add(Dense(15, activation='relu')),
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

EarlyStopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights= True )

history = model.fit(X_train, 
                    y_train_onehot, 
                    epochs=100, batch_size=128, 
                    validation_data=(X_val, y_val_onehot), 
                    callbacks= EarlyStopping)

model.save(f"models/model.h5")
model.save(f"models/model", save_format="tf")



print("Plotting history")
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'models/history_loss.png', transparent=False)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'models/history_acc.png', transparent=False)
plt.show()
