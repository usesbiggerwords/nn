from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

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

# embed
normal_vec = model.encode(normal_logs)
test_vec = model.encode(test_logs)

# compute center of normal logs
center = np.mean(normal_vec, axis=0)

def anomaly_score(vec):
    sim = cosine_similarity([vec],[center])[0][0]
    return 1 - sim

print("\nAnomaly scores\n")

for log,vec in zip(test_logs,test_vec):
    print(log,"->",round(anomaly_score(vec),3))

# combine vectors for visualization
all_vec = np.vstack([normal_vec,test_vec])

# reduce dimensions
pca = PCA(n_components=2)
points = pca.fit_transform(all_vec)

normal_points = points[:len(normal_logs)]
test_points = points[len(normal_logs):]

# plot
plt.figure(figsize=(8,6))

plt.scatter(normal_points[:,0],normal_points[:,1],label="normal",s=120)
plt.scatter(test_points[:,0],test_points[:,1],label="test",s=120)

# label points
for i,log in enumerate(normal_logs):
    plt.text(normal_points[i,0],normal_points[i,1],log[:20],fontsize=8)

for i,log in enumerate(test_logs):
    plt.text(test_points[i,0],test_points[i,1],log[:20],fontsize=8)

plt.title("Log Embedding Space (PCA Projection)")
plt.legend()
plt.show()