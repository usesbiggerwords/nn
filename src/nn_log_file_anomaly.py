from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

normal_logs = [
"INFO user login successful",
"INFO user logout",
"INFO file uploaded successfully",
"INFO database connection established",
"INFO scheduled job completed",
"INFO user updated profile"
]

test_logs = [
"INFO user login successful",
"WARNING disk space low",
"ERROR database connection failed",
"INFO scheduled job completed",
"CRITICAL kernel panic detected"
]

# embed logs
normal_vectors = model.encode(normal_logs)
test_vectors = model.encode(test_logs)

# compute center of normal logs
normal_center = np.mean(normal_vectors, axis=0)

def anomaly_score(vec):
    sim = cosine_similarity([vec],[normal_center])[0][0]
    return 1 - sim

print("Anomaly scores\n")

for log,vec in zip(test_logs,test_vectors):
    score = anomaly_score(vec)
    print(f"{log} -> {score:.3f}")