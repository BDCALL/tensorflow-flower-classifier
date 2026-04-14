import os
import shutil
import random

# Paths
RAW_DIR = "data/raw/jpg"
BASE_DIR = "data"

train_split = 0.7
valdiation_split = 0.15
test_split = 0.15

num_classes = 17
image_per_class = 80

random.seed(42)

def create_dirs():
    for split in ["train", "val", "test"]:
        for flower_class in range (num_classes):
            path = os.path.join(BASE_DIR,split,f'class_{flower_class}')
            os.makedirs(path,exist_ok=True)

def split_data():
    all_images = sorted(os.listdir(RAW_DIR))
    for flower_class in range (num_classes):
        start = flower_class * image_per_class
        end = start + image_per_class

        class_images = all_images[start:end]
        random.shuffle(class_images)

        train_end = int(len(class_images) * train_split)
        valid_end = train_end + int(len(class_images) * valdiation_split)

        train_imgs = class_images[:train_end]
        validation_imgs = class_images[train_end:valid_end]
        test_imgs = class_images[valid_end:]

        for imgs in train_imgs:
            shutil.copy(os.path.join(RAW_DIR,imgs), os.path.join(BASE_DIR, "train", f'class_{flower_class}', imgs))
        for imgs in validation_imgs:
            shutil.copy(os.path.join(RAW_DIR,imgs), os.path.join(BASE_DIR, "val", f'class_{flower_class}', imgs))
        for imgs in test_imgs:
            shutil.copy(os.path.join(RAW_DIR,imgs), os.path.join(BASE_DIR, "test", f'class_{flower_class}', imgs))


if __name__ == "__main__":
    create_dirs()
    split_data()
    print("All Data Split Complete!")