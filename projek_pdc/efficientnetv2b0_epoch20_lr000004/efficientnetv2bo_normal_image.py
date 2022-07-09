# %%
import numpy as np
import pickle
import cv2
from os import listdir

from tensorflow import keras
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
import tensorflow_hub as hub
from keras import layers
from keras.models import Sequential
import tensorflow_addons as tfa


# %%
EPOCHS = 20
INIT_LR = 4*1e-5
default_image_size = tuple((224, 224))
image_size = 0
directory_root = 'segment_img_hsv'
width=224
height=224
depth=3

# %%
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

# %%
image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for disease_folder in root_dir :
        plant_disease_image_list = listdir(f"{directory_root}/{disease_folder}")
        
        for image in plant_disease_image_list:
            image_directory = f"{directory_root}/{disease_folder}/{image}"
            if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True or image_directory.endswith(".jpeg") == True or image_directory.endswith(".png") == True:
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

# %%
print(len(image_list))

# %%
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

# %%
np_image_list = np.array(image_list, dtype=np.float16) / 225.0



# %%
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30, 
    width_shift_range=0.15,
    height_shift_range=0.15, 
    shear_range=0.15, 
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=tfa.image.gaussian_filter2d,
    fill_mode="nearest"
)
train = train_datagen.flow_from_directory("segment/train",
                            batch_size=100,
                            class_mode='categorical',
                            target_size=(224, 224),
                            shuffle=True
                           )

# %%
valid_datagen = ImageDataGenerator(rescale=1. / 255)
valid = valid_datagen.flow_from_directory("segment/valid",
                            batch_size=100,
                            class_mode='categorical',
                            target_size=(224, 224),
                            shuffle=True
                           )

# %%
test_datagen = ImageDataGenerator(rescale=1. / 255)
test = test_datagen.flow_from_directory("segment/test",
                            batch_size=1,
                            class_mode='categorical',
                            target_size=(224, 224),
                            shuffle=True
                           )

# %%
# Exponential decay
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn
exponential_decay_fn = exponential_decay(lr0=INIT_LR, s=5)
lr_scheduler_ed = keras.callbacks.LearningRateScheduler(exponential_decay_fn)

# %%
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint('efficientnetb0_epoch50_lr00004.h5', verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto') 

# %%
efn = EfficientNetV2B0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    classes=n_classes,
    classifier_activation='softmax'
)

model = Sequential()
model.add(efn)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))


# %%
opt = keras.optimizers.Adam(INIT_LR, beta_1=0.9, beta_2=0.999)
model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=['accuracy'])

# %%
history = model.fit(
    train, 
    validation_data=test,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint]
)

# %%
model_json = model.to_json()
with open("efficientnetb0_epoch50_lr00004.json", "w") as json_file:
    json_file.write(model_json)

# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()






# %%
from keras.models import load_model, model_from_json


json_file = open('efficientnetb0_epoch50_lr00004', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# load weights into new model
loaded_model.load_weights("efficientnetb0_epoch10_lr000004.json.h5")

# To Evaluate the model with test images
score = loaded_model.evaluate(test, batch_size=1, verbose=1)

# %%
# T0 print the Classification Report
y_pred = loaded_model.predict(test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test.classes, y_pred))





