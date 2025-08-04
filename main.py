import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Скачиваем датасет через kagglehub
path = kagglehub.dataset_download("datasnaek/chess")
print("Path to dataset files:", path)

# 2. Загружаем games.csv (уточни путь, если он другой)
df = pd.read_csv(path + '/games.csv')

# 3. Считаем средний рейтинг партии
df['mean_rating'] = (df['white_rating'] + df['black_rating']) / 2import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Скачиваем датасет через kagglehub
path = kagglehub.dataset_download("datasnaek/chess")
print("Path to dataset files:", path)

# 2. Загружаем games.csv (уточни путь, если он другой)
df = pd.read_csv(path + '/games.csv')

# 3. Считаем средний рейтинг партии
df['mean_rating'] = (df['white_rating'] + df['black_rating']) / 2

# 4. Группируем по дебютам (можно заменить на 'opening_eco' для компактности)
debuts = df.groupby('opening_name').agg({
    'mean_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'games_count'}).reset_index()

# 5. Оставляем дебюты, которые встречались больше 20 раз
debuts = debuts[debuts['games_count'] > 20]

# 6. Кластеризация по среднему рейтингу
X = debuts[['mean_rating']].values
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
debuts['cluster'] = kmeans.fit_predict(X)

# 7. Визуализация
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

# 8. (по желанию) Посмотреть топ-5 дебютов в каждом кластере
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


# 4. Группируем по дебютам (можно заменить на 'opening_eco' для компактности)
debuts = df.groupby('opening_name').agg({
    'mean_rating': 'mean',
    'id': 'count'
}).rename(columns={'id': 'games_count'}).reset_index()

# 5. Оставляем дебюты, которые встречались больше 20 раз
debuts = debuts[debuts['games_count'] > 20]

# 6. Кластеризация по среднему рейтингу
X = debuts[['mean_rating']].values
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
debuts['cluster'] = kmeans.fit_predict(X)

# 7. Визуализация
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

# 8. (по желанию) Посмотреть топ-5 дебютов в каждом кластере
for c in range(n_clusters):
    print(f"\nTop openings in cluster {c}:")
    print(debuts[debuts['cluster'] == c].sort_values('mean_rating', ascending=False).head(5)[['opening_name', 'mean_rating', 'games_count']])
