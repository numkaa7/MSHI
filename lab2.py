import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
np.random.seed(12)
 
# 1. Генерація 1000 значень
n = 1000
X = np.random.uniform(0, 1000, n)
noise = np.random.normal(0, 70, n)
y = 2.0 * X + 100 * np.sin(X / 70) + noise
 
# 2. Нормалізація Min-Max
X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())
 
# 3. Розподіл 80% / 20%
indices = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = indices[:split], indices[split:]
 
X_train = X_norm[train_idx].reshape(-1, 1)
X_test = X_norm[test_idx].reshape(-1, 1)
y_train = y_norm[train_idx]
y_test = y_norm[test_idx]
 
# 4. KNN-регресор
def knn_predict(X_train, y_train, X_query, k):
    predictions = []
    for xq in X_query:
        distances = np.abs(X_train[:, 0] - xq[0])
        nearest = np.argsort(distances)[:k]
        predictions.append(np.mean(y_train[nearest]))
    return np.array(predictions)
 
# 5. Перевірка різних K
results = []
 
for k in range(1, 31):
    pred = knn_predict(X_train, y_train, X_test, k)
 
    mae = np.mean(np.abs(y_test - pred))
    mse = np.mean((y_test - pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum(
        (y_test - np.mean(y_test)) ** 2
    )
 
    results.append([k, mae, mse, rmse, r2])
 
results_df = pd.DataFrame(
    results, columns=["K", "MAE", "MSE", "RMSE", "R2"]
)
 
# 6. Найкраще K за мінімальним MSE
best_row = results_df.loc[results_df["MSE"].idxmin()]
best_k = int(best_row["K"])
best_pred = knn_predict(X_train, y_train, X_test, best_k)
 
print(results_df.to_string(index=False))
print("\nНайкраще K:", best_k)
print("MAE:", best_row["MAE"])
print("MSE:", best_row["MSE"])
print("RMSE:", best_row["RMSE"])
print("R2:", best_row["R2"])
 
# 7. Візуалізація
plt.figure(figsize=(8, 5))
plt.plot(results_df["K"], results_df["MSE"], marker="o")
plt.xlabel("K")
plt.ylabel("MSE")
plt.title("Залежність MSE від K")
plt.grid(True)
plt.show()
 
plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.65)
mn = min(y_test.min(), best_pred.min())
mx = max(y_test.max(), best_pred.max())
plt.plot([mn, mx], [mn, mx], "--")
plt.xlabel("Фактичні значення")
plt.ylabel("Прогнозовані значення")
plt.title(f"Фактичні та прогнозовані значення, K={best_k}")
plt.grid(True)
plt.show()

order = np.argsort(X_test[:, 0])

x_sorted = X_test[order]
pred_sorted = best_pred[order]
y_sorted = y_test[order]

# Повертаємо нормалізовані значення до початкового масштабу
x_original = x_sorted[:, 0] * (X.max() - X.min()) + X.min()
pred_original = pred_sorted * (y.max() - y.min()) + y.min()
y_original = y_sorted * (y.max() - y.min()) + y.min()

# Побудова графіка
plt.figure(figsize=(9, 5))

plt.scatter(
    x_original,
    y_original,
    s=12,
    alpha=0.35,
    label="Тестові дані"
)

plt.plot(
    x_original,
    pred_original,
    linewidth=2,
    label=f"KNN-регресія, K={best_k}"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Результат KNN-регресії на тестовій вибірці")
plt.legend()
plt.grid(True)

plt.show()