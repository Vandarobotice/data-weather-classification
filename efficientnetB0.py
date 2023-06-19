#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# In[2]:


image_size = (224,224)
batch_size = 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "F:/classy/train/",
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "F:/classy/validation/",
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# In[3]:


def tensorboard_callback(directory, name):
    log_dir = directory + "/" + name
    t_c = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
    return t_c

def model_checkpoint(directory, name):
    log_dir = directory + "/" + name
    m_c = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,
                                             monitor="val_accuracy",
                                             save_best_only=True,
                                             save_weights_only=True,
                                             verbose=1)
    return m_c


# In[4]:


wp='F:/classy/weights/efficientnetb0_notop.h5'
base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False,weights=wp)
base_model.trainable = False


# In[5]:


from tensorflow.keras import layers

inputs = layers.Input(shape = (224,224,3), name='inputLayer')
x = base_model(inputs, training = False)
x = layers.GlobalAveragePooling2D(name='poolingLayer')(x)
x = layers.Dense(5, name='outputLayer')(x)
outputs = layers.Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)

model = tf.keras.Model(inputs, outputs, name = "FeatureExtractionModel")


# In[6]:


model.summary()


# In[7]:


for lnum, layer in enumerate(model.layers):
    print(lnum, layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# In[8]:


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["accuracy"])

hist_model = model.fit(train_data,
                       epochs = 3,
                       steps_per_epoch=len(train_data),
                       validation_data=test_data,
                       validation_steps=int(0.1 * len(test_data)),
                       callbacks=[tensorboard_callback("Tensorboard","model"),model_checkpoint("Checkpoints","model.ckpt")])


# In[ ]:





# In[9]:


base_model.trainable = True


# In[10]:


for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


# In[11]:


for lnum, layer in enumerate(model.layers[1].layers[-10:]):
    print(lnum, layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# In[12]:


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics = ["accuracy"])
hist_model_tuned = model.fit(train_data,
                             epochs=5,
                             steps_per_epoch=len(train_data),
                             validation_data=test_data,
                             validation_steps=int(0.1*len(test_data)),
                             initial_epoch=hist_model.epoch[-1],
                             callbacks=[tensorboard_callback("Tensorboard", "model_tuned"), model_checkpoint("Checkpoints", "model_tuned.ckpt")])


# In[13]:


model_tuned_results = model.evaluate(test_data)


# In[14]:


def compare_histories(original_history, new_history, initial_epochs):
    """
    Compares two model history objects.
    """
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    
    #val_acc = original_history.history["val_accuracy"]
    #val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    #total_val_acc = val_acc + new_history.history["val_accuracy"]
    #total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(9, 9))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    #plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start of Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    #plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start of Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# In[15]:


compare_histories(hist_model, hist_model_tuned, initial_epochs=3)


# In[16]:


preds = model.predict(test_data, verbose = 1)


# In[26]:


pred_labels = tf.argmax(preds, axis=1)
pred_labels[:10]


# In[27]:


test_labels = np.concatenate([y for x, y in test_data], axis=0)
test_labels[:10]


# In[19]:


# Step 1
test_image_batches = []
for images, labels in test_data.take(-1):
    test_image_batches.append(images.numpy())

# Step 2
test_images = [item for sublist in test_image_batches for item in sublist]
len(test_images)


# In[20]:


from sklearn.metrics import classification_report
report = classification_report(test_labels, pred_labels, output_dict=True)

#check a small slice of the dictionary
import itertools
dict(itertools.islice(report.items(), 5))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




