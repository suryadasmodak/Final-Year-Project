import os

base = r"C:/Users/ADMIN/Desktop/Anti-Spoffing/Dataset/SplitData"

print("Train images:", len(os.listdir(base + "/train/images")))
print("Val images:", len(os.listdir(base + "/val/images")))
print("Test images:", len(os.listdir(base + "/test/images")))