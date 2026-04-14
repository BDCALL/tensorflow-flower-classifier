from dataloader import load_data
from model import build_model, compile_model
from plot import plot_history

train_ds, valid_ds, test_ds = load_data()
model = build_model(17)
model = compile_model(model)

history = model.fit(train_ds, validation_data = valid_ds, epochs = 30, verbose=2)
plot_history(history)

model.save("flowers_model.h5")

# model = tf.keras.models.load_model("flowers.model.h5")
_, test_acc = model.evaluate(test_ds)
print("test accuracy : ", test_acc)