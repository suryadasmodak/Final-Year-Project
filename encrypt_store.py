import tenseal as ts
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os
import pickle

# ─── MongoDB Connection ────────────────────────────────────────────────────────
def get_database():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["biometric_db"]
    return db

# ─── CKKS Context Setup ───────────────────────────────────────────────────────
def create_ckks_context():
    """Create and return a CKKS context for homomorphic encryption"""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def save_context(context, path="ckks_context.pkl"):
    """Save CKKS context to file"""
    with open(path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
    print(f"✅ CKKS context saved to {path}")

def load_context(path="ckks_context.pkl"):
    """Load CKKS context from file"""
    with open(path, "rb") as f:
        context = ts.context_from(f.read())
    print(f"✅ CKKS context loaded from {path}")
    return context

# ─── Encryption ───────────────────────────────────────────────────────────────
def encrypt_template(protected_template: np.ndarray, context) -> bytes:
    """
    Encrypt a protected biometric template using CKKS.
    
    Args:
        protected_template: Transformed feature vector (512-dim)
        context: CKKS context
    
    Returns:
        encrypted_bytes: Serialized encrypted template
    """
    # Convert to list of floats
    template_list = protected_template.tolist()

    # Encrypt using CKKS
    encrypted = ts.ckks_vector(context, template_list)

    # Serialize to bytes for storage
    encrypted_bytes = encrypted.serialize()

    return encrypted_bytes

def decrypt_template(encrypted_bytes: bytes, context) -> np.ndarray:
    """
    Decrypt an encrypted template.
    
    Args:
        encrypted_bytes: Serialized encrypted template
        context: CKKS context
    
    Returns:
        decrypted: Decrypted feature vector
    """
    encrypted = ts.ckks_vector_from(context, encrypted_bytes)
    decrypted = np.array(encrypted.decrypt())
    return decrypted

# ─── Enrollment ───────────────────────────────────────────────────────────────
def enroll_user(user_id: str, name: str, protected_template: np.ndarray, context):
    """
    Enroll a new user by encrypting and storing their biometric template.
    
    Args:
        user_id: Unique user ID (e.g., "surya_001")
        name: Full name of the user
        protected_template: Cancelable transformed feature vector
        context: CKKS context
    """
    db = get_database()
    collection = db["enrolled_users"]

    # Check if user already exists
    existing = collection.find_one({"user_id": user_id})
    if existing:
        print(f"⚠️ User '{user_id}' already enrolled! Updating template...")
        collection.delete_one({"user_id": user_id})

    # Encrypt the template
    print(f"🔐 Encrypting template for user: {name}...")
    encrypted_bytes = encrypt_template(protected_template, context)

    # Store in MongoDB
    user_document = {
        "user_id": user_id,
        "name": name,
        "encrypted_template": encrypted_bytes,
        "enrolled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    collection.insert_one(user_document)
    print(f"✅ User '{name}' enrolled successfully in MongoDB!")
    print(f"📦 Encrypted template size: {len(encrypted_bytes)} bytes")

# ─── Retrieve All Users ───────────────────────────────────────────────────────
def get_all_users():
    """Get list of all enrolled users"""
    db = get_database()
    collection = db["enrolled_users"]
    users = collection.find({}, {"user_id": 1, "name": 1, "enrolled_at": 1, "_id": 0})
    return list(users)

def get_user_template(user_id: str) -> bytes:
    """Get encrypted template of a specific user"""
    db = get_database()
    collection = db["enrolled_users"]
    user = collection.find_one({"user_id": user_id})
    if user:
        return user["encrypted_template"]
    return None

# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("🔐 Testing Encryption and MongoDB Storage Module")
    print("=" * 55)

    # Step 1 — Create or load CKKS context
    context_path = "ckks_context.pkl"
    if os.path.exists(context_path):
        context = load_context(context_path)
    else:
        print("🔧 Creating new CKKS context...")
        context = create_ckks_context()
        save_context(context, context_path)

    # Step 2 — Simulate a protected template (normally comes from cancelable_transform.py)
    print("\n📊 Simulating protected biometric template...")
    from cancelable_transform import generate_transform_key, cancelable_transform

    # Simulate FaceNet embedding
    raw_embedding = np.random.randn(512).astype(np.float64)

    # Apply cancelable transformation
    W = generate_transform_key("test_user_001")
    protected = cancelable_transform(raw_embedding, W)
    print(f"✅ Protected template shape: {protected.shape}")

    # Step 3 — Encrypt and store
    print("\n📥 Enrolling test user...")
    enroll_user(
        user_id="test_user_001",
        name="Test User",
        protected_template=protected,
        context=context
    )

    # Step 4 — Show all enrolled users
    print("\n👥 All Enrolled Users:")
    users = get_all_users()
    for user in users:
        print(f"  - {user['name']} (ID: {user['user_id']}) enrolled at {user['enrolled_at']}")

    # Step 5 — Verify encryption/decryption
    print("\n🔍 Verifying encryption/decryption...")
    encrypted_bytes = get_user_template("test_user_001")
    decrypted = decrypt_template(encrypted_bytes, context)
    print(f"✅ Decrypted template shape: {decrypted.shape}")
    print(f"✅ First 5 values (original):  {protected[:5]}")
    print(f"✅ First 5 values (decrypted): {decrypted[:5]}")
    print(f"✅ Match (approx): {np.allclose(protected, decrypted, atol=1e-3)}")