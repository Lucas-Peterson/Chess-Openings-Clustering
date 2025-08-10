# Chess Openings Clustering

This project analyzes chess games from the Kaggle dataset [datasnaek/chess](https://www.kaggle.com/datasets/datasnaek/chess) and groups chess openings by the average player rating of the games in which they appear.

## Steps
1. **Load data** using `kagglehub` and read `games.csv`.
2. **Calculate mean rating** for each game from `white_rating` and `black_rating`.
3. **Aggregate by opening** to find:
   - Average rating (`mean_rating`)
   - Number of games (`games_count`)
4. **Filter rare openings** (fewer than 20 games).
5. **Cluster openings** using **KMeans** on `mean_rating` into 4 groups.
6. **Visualize**:
   - Scatter plot: mean rating vs. popularity
   - Silhouette score distribution
7. **Show top openings** for each cluster.

## Results
- **Cluster 0:** Advanced, theory-heavy openings (e.g., *Ruy Lopez*, *Trompowsky Attack*)
- **Cluster 1:** Mid-to-high rating, creative mix of styles (e.g., *Bird Opening*, *Queen’s Gambit Declined*)
- **Cluster 2:** Lower ratings (~1490), experimental or simple openings (e.g., *French Defense*, *Englund Gambit*)
- **Cluster 3:** Mid-level, solid classical openings (e.g., *Italian Game*, *Slav Defense*)

## Requirements
```bash
pip install pandas matplotlib scikit-learn kagglehub
