# 🌸 Flower Classification with TensorFlow (CNN vs Transfer Learning)

This project is a deep learning image classifier that identifies 17 types of flowers using TensorFlow.  
It compares two approaches:

- 🧠 Custom CNN (from scratch)
- 🚀 Transfer Learning (MobileNetV2 pretrained on ImageNet)

---

# 📊 Results Summary

| Model Type | Test Accuracy |
|------------|--------------|
| Custom CNN (from scratch) | ~47% |
| Transfer Learning (MobileNetV2) | ~88% |

---

# 🧠 1. Custom CNN Approach

A convolutional neural network was built from scratch with:

- Conv2D layers
- Batch Normalization
- MaxPooling
- Dropout
- Softmax classifier

### Result:
- Accuracy: **~47%**
- Limitation: Small dataset (17 classes, ~80 images each)
- Struggled to learn deep visual features

---

# 🚀 2. Transfer Learning Approach (MobileNetV2)

A pretrained MobileNetV2 model was used with ImageNet weights.

### Architecture:
- Frozen MobileNetV2 base
- Global Average Pooling
- Dense layer (128 units)
- Dropout (0.3)
- Softmax output (17 classes)

### Result:
- Accuracy: **~88%**
- Much faster convergence
- Strong generalisation on small dataset

---

# 📉 Key Insights

### Why Transfer Learning performed better:

- Pretrained on 14M+ images (ImageNet)
- Already understands edges, textures, and shapes
- Requires only task-specific fine-tuning
- Works extremely well on small datasets

---

# 🧪 Reflection

This project highlighted an important lesson in deep learning:

> Increasing model complexity does not always improve performance.

The initial CNN model was heavily tuned with Batch Normalization, dropout, and data augmentation. However, this led to **over-regularisation and underfitting**, reducing accuracy.

In contrast, transfer learning provided a **strong pretrained feature extractor**, allowing the model to focus only on learning flower-specific patterns. This resulted in a significant performance boost from **47% to 88% accuracy**.