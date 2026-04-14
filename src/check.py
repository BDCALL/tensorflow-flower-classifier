import os

BASE_DIR = "data"

def check_counts():
    for split in ["train", "val", "test"]:
        print(f"\n📂 {split.upper()}")
        total = 0

        for cls in os.listdir(os.path.join(BASE_DIR, split)):
            class_path = os.path.join(BASE_DIR, split, cls)
            count = len(os.listdir(class_path))
            total += count
            print(f"{cls}: {count}")

        print(f"Total: {total}")

check_counts()

def check_duplicates():
    train_files = set()
    val_files = set()
    test_files = set()

    for cls in os.listdir("data/train"):
        train_files.update(os.listdir(os.path.join("data/train", cls)))

    for cls in os.listdir("data/val"):
        val_files.update(os.listdir(os.path.join("data/val", cls)))

    for cls in os.listdir("data/test"):
        test_files.update(os.listdir(os.path.join("data/test", cls)))

    print("Train ∩ Val:", len(train_files & val_files))
    print("Train ∩ Test:", len(train_files & test_files))
    print("Val ∩ Test:", len(val_files & test_files))

check_duplicates()