#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import os
import matplotlib.pyplot as plt

# In[1]:

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# In[2]:

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# In[3]:

img_width=256; img_height=256
batch_size=8


# In[ ]:


# In[4]:

TRAINING_DIR = "F:/classy/train/"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=30,
                                   zoom_range=0.4,
                                   horizontal_flip=True
                                   )

# In[5]:

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=(img_height, img_width)
                                                    )


# In[ ]:


# In[6]:

VALIDATION_DIR = "F:/classy/validation/"

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# In[7]:

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              target_size=(img_height, img_width)
                                                             )


# In[ ]:


# In[8]:

callbacks = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')
#best_model_file = '.../resnet50_drop_batch_best_weights_256.h5'
#best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# In[9]:

wp='F:/classy/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet50_base = ResNet50(include_top=False, weights=wp,
                         input_tensor=None, input_shape=(img_height, img_width,3))


# In[ ]:


# In[10]:

print('Adding new layers...')
output = resnet50_base.get_layer(index = -1).output  
output = Flatten()(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(5, activation='softmax')(output)
print('New layers added!')


# In[ ]:


# In[11]:

resnet50_model = Model(resnet50_base.input, output)
for layer in resnet50_model.layers[:-7]:
    layer.trainable = False

resnet50_model.summary()


# In[ ]:


# In[12]:

resnet50_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics =['accuracy'])


# In[ ]:


# In[13]:

history = resnet50_model.fit_generator(train_generator,
                              epochs=5,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks=callbacks
                              )


# In[ ]:


# In[15]:

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

# In[16]:

fig = plt.figure(figsize=(20,10))
plt.plot(epochs, acc, 'r', label="Training Accuracy")
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy of ResNet50')
plt.legend(loc='lower right')
plt.show()
#fig.savefig('.../Accuracy_curve_resnet50_drop_batch_256.jpg')

# In[17]:

fig2 = plt.figure(figsize=(20,10))
plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss of ResNet50')
#fig2.savefig('.../Loss_curve_resnet50_drop_batch_256.jpg')


# In[ ]:


model_tuned_results = resnet50_model.evaluate(validation_generator)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




