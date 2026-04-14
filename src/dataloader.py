import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (150,150)
BATCH_SIZE = 32

def load_data():
    train_ds = keras.utils.image_dataset_from_directory("data/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    validation_ds = keras.utils.image_dataset_from_directory("data/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE)
    test_ds = keras.utils.image_dataset_from_directory("data/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE)

    normalization_layer = keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y : (normalization_layer(x),y))
    validation_ds = validation_ds.map(lambda x, y : (normalization_layer(x),y))
    test_ds = test_ds.map(lambda x, y : (normalization_layer(x),y))
    return train_ds, validation_ds, test_ds