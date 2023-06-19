#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


# In[3]:


image_size = (380,380)
batch_size = 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/gadmin/Desktop/farshid/dataset_train/",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/gadmin/Desktop/farshid/dataset_validation/",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)



test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/gadmin/Desktop/farshid/dataset_test/",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# In[ ]:


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


# In[ ]:


wp='/home/gadmin/Desktop/farshid/efficientnetb4_notop.h5'
base_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,weights=wp)
base_model.trainable = False


# In[ ]:


from tensorflow.keras import layers

inputs = layers.Input(shape = (380,380,3), name='inputLayer')
x = base_model(inputs, training = False)
x = layers.GlobalAveragePooling2D(name='poolingLayer')(x)
x = layers.Dense(6, name='outputLayer')(x)
outputs = layers.Activation(activation="softmax", dtype=tf.float32, name='activationLayer')(x)

model = tf.keras.Model(inputs, outputs, name = "FeatureExtractionModel")


# In[ ]:


model.summary()


# In[ ]:


for lnum, layer in enumerate(model.layers):
    print(lnum, layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# In[ ]:


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["accuracy"])

hist_model = model.fit(train_data,
                       epochs = 30,
                       steps_per_epoch=len(train_data),
                       validation_data=validation_data,
                       #validation_steps=int(0.1 * len(validation_data)),
                       callbacks=[tensorboard_callback("Tensorboard","model"),model_checkpoint("Checkpoints","model.ckpt")])


# In[ ]:


base_model.trainable = True


# In[ ]:


for layer in model.layers[1].layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


# In[ ]:


for lnum, layer in enumerate(model.layers[1].layers[-10:]):
    print(lnum, layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# In[ ]:


model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics = ["accuracy"])
hist_model_tuned = model.fit(train_data,
                             epochs=50,
                             steps_per_epoch=len(train_data),
                             validation_data=validation_data,
                             #validation_steps=int(0.1*len(validation_data)),
                             initial_epoch=hist_model.epoch[-1],
                             callbacks=[tensorboard_callback("Tensorboard", "model_tuned"), model_checkpoint("Checkpoints", "model_tuned.ckpt")])


# In[ ]:


model_tuned_results = model.evaluate(test_data)


# In[ ]:


def compare_histories(original_history, new_history, initial_epochs):
    """
    Compares two model history objects.
    """
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]
    
    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    plt.figure(figsize=(9, 9))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start of Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start of Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# In[ ]:


compare_histories(hist_model, hist_model_tuned, initial_epochs=31)


# In[ ]:





from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image



# In[5]:
test_df=pd.read_csv("/home/gadmin/Desktop/farshid/test.csv")
y_true = test_df['label']
path=test_df['path']
y_pred=[]
for img_path in test_df['path']:
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img, 0)
    
    pred = model.predict(img_array,verbose=0)
    pred_labels = tf.argmax(pred,axis=1)
    pred_labels = pred_labels.numpy()
    y_pred.append(pred_labels)





# In[ ]:
from sklearn.metrics import confusion_matrix
import os
conf_mat = confusion_matrix(y_true, y_pred)

# In[ ]:
train_dir='/home/gadmin/Desktop/farshid/dataset_train/'
classes = os.listdir(train_dir)

# In[ ]:

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# In[ ]:
np.set_printoptions(precision=2)

fig1 = plt.figure(figsize=(7,6))
plot_confusion_matrix(conf_mat, classes=classes, title='Confusion matrix, without normalization')
#fig1.savefig('../cm_wo_norm.jpg')
plt.show()

# In[ ]:
np.set_printoptions(precision=2)

fig2 = plt.figure(figsize=(7,6))
plot_confusion_matrix(conf_mat, classes=classes, normalize = True, title='Normalized Confusion matrix')
#fig2.savefig('../cm_norm.jpg')
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
report = classification_report(y_true, y_pred, output_dict=True)

#check a small slice of the dictionary
import itertools
report=dict(itertools.islice(report.items(), 6))
print(report)
