import tensorflow as tf
from tensorflow import keras

def build_model(num_class=17):

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(150,150,3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False 

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_class, activation='softmax')
    ])

    return model

def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model