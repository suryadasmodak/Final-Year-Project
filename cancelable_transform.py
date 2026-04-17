import numpy as np
import os

def generate_transform_key(user_id: str, size: int = 512, keys_dir: str = "keys") -> np.ndarray:
    """
    Generate and save a user-specific random orthogonal matrix (transformation key).
    If key already exists for the user, load and return it.
    
    Args:
        user_id: Unique identifier for the user (e.g., name or ID number)
        size: Size of the feature vector (512 for FaceNet)
        keys_dir: Directory to save keys
    
    Returns:
        W: Orthogonal transformation matrix (512x512)
    """
    os.makedirs(keys_dir, exist_ok=True)
    key_path = os.path.join(keys_dir, f"{user_id}_key.npy")

    if os.path.exists(key_path):
        print(f"🔑 Loading existing key for user: {user_id}")
        W = np.load(key_path)
    else:
        print(f"🔑 Generating new key for user: {user_id}")
        # Generate random matrix
        random_matrix = np.random.randn(size, size)
        # Make it orthogonal using QR decomposition
        W, _ = np.linalg.qr(random_matrix)
        # Save the key
        np.save(key_path, W)
        print(f"✅ Key saved to {key_path}")

    return W


def cancelable_transform(feature_vector: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Apply cancelable biometric transformation to a feature vector.
    
    Args:
        feature_vector: Raw FaceNet embedding (512-dim)
        W: Orthogonal transformation matrix
    
    Returns:
        protected_template: Transformed and protected feature vector (512-dim)
    """
    # Step 1 — Normalize the feature vector to unit length
    norm = np.linalg.norm(feature_vector)
    if norm == 0:
        raise ValueError("Feature vector has zero norm!")
    normalized = feature_vector / norm

    # Step 2 — Apply transformation: W × v
    protected_template = W @ normalized

    return protected_template


def revoke_key(user_id: str, keys_dir: str = "keys") -> np.ndarray:
    """
    Revoke existing key and generate a new one for the user.
    Use this if the user's key is compromised.
    
    Args:
        user_id: Unique identifier for the user
        keys_dir: Directory where keys are stored
    
    Returns:
        W_new: New orthogonal transformation matrix
    """
    key_path = os.path.join(keys_dir, f"{user_id}_key.npy")

    # Delete old key
    if os.path.exists(key_path):
        os.remove(key_path)
        print(f"🗑️ Old key for user '{user_id}' has been revoked!")
    else:
        print(f"⚠️ No existing key found for user '{user_id}'")

    # Generate new key
    print(f"🔑 Generating new key for user: {user_id}")
    W_new = generate_transform_key(user_id)
    return W_new


def verify_transformation(vec_a: np.ndarray, vec_b: np.ndarray, W: np.ndarray) -> dict:
    """
    Verify that cosine similarity is preserved after transformation.
    Useful for testing/debugging.
    
    Args:
        vec_a: First feature vector
        vec_b: Second feature vector
        W: Transformation matrix
    
    Returns:
        dict with similarity scores before and after transformation
    """
    # Normalize
    a = vec_a / np.linalg.norm(vec_a)
    b = vec_b / np.linalg.norm(vec_b)

    # Similarity before transformation
    sim_before = float(np.dot(a, b))

    # Transform
    a_transformed = W @ a
    b_transformed = W @ b

    # Normalize transformed vectors
    a_transformed = a_transformed / np.linalg.norm(a_transformed)
    b_transformed = b_transformed / np.linalg.norm(b_transformed)

    # Similarity after transformation
    sim_after = float(np.dot(a_transformed, b_transformed))

    return {
        "similarity_before": round(sim_before, 6),
        "similarity_after": round(sim_after, 6),
        "preserved": abs(sim_before - sim_after) < 1e-6
    }


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Testing Cancelable Transformation Module")
    print("=" * 50)

    # Simulate a FaceNet feature vector
    test_vector = np.random.randn(512).astype(np.float32)

    # Generate key for a test user
    user_id = "surya_001"
    W = generate_transform_key(user_id)

    # Apply transformation
    protected = cancelable_transform(test_vector, W)
    print(f"\n✅ Original vector (first 5): {test_vector[:5]}")
    print(f"✅ Protected template (first 5): {protected[:5]}")
    print(f"✅ Shape preserved: {protected.shape}")

    # Verify cosine similarity is preserved
    test_vector_2 = np.random.randn(512).astype(np.float32)
    result = verify_transformation(test_vector, test_vector_2, W)
    print(f"\n📊 Similarity before transformation: {result['similarity_before']}")
    print(f"📊 Similarity after transformation:  {result['similarity_after']}")
    print(f"📊 Cosine similarity preserved: {result['preserved']}")

    # Test revocation
    print("\n🔄 Testing key revocation...")
    W_new = revoke_key(user_id)
    protected_new = cancelable_transform(test_vector, W_new)
    print(f"✅ New protected template (first 5): {protected_new[:5]}")
    print(f"✅ Templates are different: {not np.allclose(protected, protected_new)}")