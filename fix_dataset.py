import os
import yaml

base = r"C:/Users/ADMIN/Desktop/Anti-Spoffing/Dataset/SplitData"

data = {
    "path": base,
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 2,
    "names": ["fake", "real"]
}

with open(base + "/data.yaml", "w") as f:
    yaml.dump(data, f)

print("New data.yaml created")