import tensorflow as tf
from tensorflow import keras

def build_model(num_class = 17):
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.02),
        keras.layers.RandomZoom(0.02)
    ])

    model = keras.models.Sequential()

    model.add(data_augmentation)

    model.add(keras.layers.Conv2D(32,(3,3),activation='relu', input_shape = (150,150,3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2)))

    model.add(keras.layers.Conv2D(64,(3,3), activation= 'relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(2,2))

    model.add(keras.layers.Conv2D(128,(3,3),activation= 'relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(2,2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(num_class,activation="softmax"))

    return model

def compile_model(model):
    model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model