import tensorflow as tf
from translearn_model import build_model, compile_model
from dataloader import load_data
from plot import plot_history

train_ds, val_ds, test_ds = load_data()

model = build_model(num_class=17)
model = compile_model(model)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    verbose=2
)

plot_history(history)

model.save("flowers_mobilenet.keras")

_, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)
