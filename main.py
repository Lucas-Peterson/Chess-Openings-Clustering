import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Download our dataset
path = kagglehub.dataset_download("datasnaek/chess")
print("Path to dataset files:", path)
df = pd.read_csv(path + '/games.csv')
print(f" Dataset loaded. Rows: {len(df)}, Columns: {len(df.columns)}")

# 3. Calculate average game rating
df['mean_rating'] = (df['white_rating'] + df['black_rating']) / 2

# 4. Group by openings (can be replaced with 'opening_eco' for compactness)
debuts = df.groupby('opening_name').agg({
    'mean_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'games_count'}).reset_index()

# 5. Keep only openings that occurred more than 20 times
debuts = debuts[debuts['games_count'] > 20]

# 6. Clustering by average rating
X = debuts[['mean_rating']].values
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
debuts['cluster'] = kmeans.fit_predict(X)

# 7. Visualization
plt.figure(figsize=(10, 6))
for c in range(n_clusters):
    d = debuts[debuts['cluster'] == c]
    plt.scatter(d['mean_rating'], d['games_count'], label=f'Cluster {c}', alpha=0.7)
plt.xlabel('Mean Player Rating')
plt.ylabel('Number of Games')
plt.title('Debuts Clustered by Mean Player Rating')
plt.legend()
plt.grid()
plt.show()

# 8. (optional) View top-5 openings in each cluster
for c in range(n_clusters):
    print(f"\nTop openings in cluster {c}:")
    print(debuts[debuts['cluster'] == c].sort_values('mean_rating', ascending=False).head(5)[['opening_name', 'mean_rating', 'games_count']])

from sklearn.metrics import silhouette_samples
silhouette_vals = silhouette_samples(X, debuts['cluster'])
plt.hist(silhouette_vals, bins=20)
plt.xlabel("Silhouette coefficient")
plt.ylabel("Number of samples")
plt.title("Silhouette score distribution")
plt.show()

# 4. Group by openings (can be replaced with 'opening_eco' for compactness)
debuts = df.groupby('opening_name').agg({
    'mean_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'games_count'}).reset_index()

# 5. Keep only openings that occurred more than 20 times
debuts = debuts[debuts['games_count'] > 20]

# 6. Clustering by average rating
X = debuts[['mean_rating']].values
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
debuts['cluster'] = kmeans.fit_predict(X)

# 7. Visualization
plt.figure(figsize=(10, 6))
for c in range(n_clusters):
    d = debuts[debuts['cluster'] == c]
    plt.scatter(d['mean_rating'], d['games_count'], label=f'Cluster {c}', alpha=0.7)
plt.xlabel('Mean Player Rating')
plt.ylabel('Number of Games')
plt.title('Debuts Clustered by Mean Player Rating')
plt.legend()
plt.grid()
plt.show()

# 8. View top-5 openings in each cluster
for c in range(n_clusters):
    print(f"\nTop openings in cluster {c}:")
    print(debuts[debuts['cluster'] == c].sort_values('mean_rating', ascending=False).head(5)[['opening_name', 'mean_rating', 'games_count']])
