import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np

# ✅ Load FaceNet model
print("Loading FaceNet model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InceptionResnetV1(pretrained=None).eval().to(device)
model.load_state_dict(torch.load("20180402-114759-vggface2.pt", map_location=device), strict=False)

print("Model loaded successfully ✅")


# ✅ Preprocess image
def preprocess_face(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((160, 160))

    img_array = np.array(img)

    img_tensor = torch.tensor(img_array).permute(2, 0, 1).float()

    # Normalize image to [-1, 1]
    img_tensor = (img_tensor - 127.5) / 128.0

    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor


# ✅ Extract embedding
def extract_feature_vector(image_path):
    face_tensor = preprocess_face(image_path)

    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding.squeeze().cpu().numpy()


# ✅ Main test
if __name__ == "__main__":
    print("\n--- Face Feature Extraction Demo ---")

    image_path = "face.jpg"   # Put any face image in folder

    embedding = extract_feature_vector(image_path)

    print("\n✅ Feature Vector Generated Successfully!")
    print("Embedding length:", len(embedding))
    print("First 10 values:", embedding[:10])