import os
import random
import shutil

inputFolder = "Dataset/all"
outputFolder = "Dataset/SplitData"

classes = ["fake", "real"]

# Split ratios

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Remove old SplitData folder

if os.path.exists(outputFolder):
 shutil.rmtree(outputFolder)

# Create folders

os.makedirs(outputFolder + "/train/images", exist_ok=True)
os.makedirs(outputFolder + "/train/labels", exist_ok=True)

os.makedirs(outputFolder + "/val/images", exist_ok=True)
os.makedirs(outputFolder + "/val/labels", exist_ok=True)

os.makedirs(outputFolder + "/test/images", exist_ok=True)
os.makedirs(outputFolder + "/test/labels", exist_ok=True)

# Read dataset

files = os.listdir(inputFolder)

names = []
for file in files:
 if file.endswith(".jpg"):
  names.append(file.split(".")[0])

# Remove duplicates

names = list(set(names))

# Shuffle data

random.shuffle(names)

total = len(names)

train_count = int(total * train_ratio)
val_count = int(total * val_ratio)
test_count = total - train_count - val_count

train_files = names[:train_count]
val_files = names[train_count:train_count + val_count]
test_files = names[train_count + val_count:]

print("Total Images:", total)
print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))

def copy_files(file_list, folder):

 for name in file_list:

    img_src = inputFolder + "/" + name + ".jpg"
    label_src = inputFolder + "/" + name + ".txt"

    img_dst = outputFolder + "/" + folder + "/images/" + name + ".jpg"
    label_dst = outputFolder + "/" + folder + "/labels/" + name + ".txt"

    if os.path.exists(img_src):
        shutil.copy(img_src, img_dst)

    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("Dataset split completed")

# Create data.yaml

yaml_text = "path: ../Dataset/SplitData\n"
yaml_text += "train: train/images\n"
yaml_text += "val: val/images\n"
yaml_text += "test: test/images\n\n"
yaml_text += "nc: 2\n"
yaml_text += "names: ['fake','real']"

with open(outputFolder + "/data.yaml", "w") as f:
 f.write(yaml_text)

print("data.yaml created")
