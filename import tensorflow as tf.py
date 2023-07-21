import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# base model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = Dense(1024, activation='relu')(x)

# and a logistic layer -- we'll assume we have 200 classes for different objects in living room
predictions = Dense(200, activation='softmax')(x)

# model to be trained
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# data preparation
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# assuming you have a folder with all images, and the images are named by their class
train_gen = data_gen.flow_from_directory('pathtoimages', target_size=(224, 224), batch_size=32, subset='training')
val_gen = data_gen.flow_from_directory('pathtoimages', target_size=(224, 224), batch_size=32, subset='validation')

# early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# train the model on the new data for a few epochs
model.fit(train_gen, epochs=100, validation_data=val_gen, callbacks=[early_stopping])
