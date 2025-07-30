import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# 1. Загрузка датасета
path = kagglehub.dataset_download("arevel/chess-games")
df = pd.read_csv(path + "/lichess_db_standard_rated_2016-07.csv")

# 2. Очистка и выбор признаков
df = df[['WhiteElo', 'BlackElo', 'Opening', 'TimeControl', 'Termination', 'Result']].dropna()

# 3. Удаляем экзотические значения результата
df = df[df['Result'].isin(['1-0', '0-1', '1/2-1/2'])]

# 4. Кодируем целевую переменную
label_map = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}
df['Target'] = df['Result'].map(label_map)

# 5. Преобразуем TimeControl в общее время (основа + инкремент)
def parse_timecontrol(tc):
    try:
        base, inc = tc.split('+')
        return int(base) + int(inc) * 40  # предполагаем 40 ходов в среднем
    except:
        return np.nan

df['TotalTime'] = df['TimeControl'].apply(parse_timecontrol)
df.dropna(inplace=True)

# 6. Категориальные признаки: Opening и Termination
top_openings = df['Opening'].value_counts().nlargest(20).index  # только топ-20 дебютов
df['Opening'] = df['Opening'].apply(lambda x: x if x in top_openings else 'Other')
df = pd.get_dummies(df, columns=['Opening', 'Termination'], drop_first=True)

# 7. Финальные признаки
X = df.drop(columns=['Result', 'Target', 'TimeControl'])
y = df['Target']

# 8. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 9. Масштабирование числовых признаков
num_cols = ['WhiteElo', 'BlackElo', 'TotalTime']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 10. Обучение моделей
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Gradient Boosting": GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:\n")
    print(classification_report(y_test, y_pred, target_names=["White win", "Black win", "Draw"]))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["White", "Black", "Draw"], yticklabels=["White", "Black", "Draw"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
