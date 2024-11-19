import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Загружаем данные
path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты"
agronomic_data_path = os.path.join(path, "Агрономические данные.xlsx")
climate_data_path = os.path.join(path, "Климат по 10дням.xlsx")
soil_data_path = os.path.join(path, "Почвенные данные.xlsx")
vegetation_data_path = os.path.join(path, "Вегетационный период.xlsx")

agronomic_data = pd.read_excel(agronomic_data_path, decimal=',')
climate_data = pd.read_excel(climate_data_path, decimal=',')
soil_data = pd.read_excel(soil_data_path, decimal=',')
vegetation_data = pd.read_excel(vegetation_data_path, decimal=',')

# Пример фильтрации данных для картофеля в регионе Орловская область
culture = "Картофель"
region_prefix = "Орл"

# Фильтруем агрономические данные
agronomic_filtered = agronomic_data[agronomic_data['Культура'] == culture]
agronomic_filtered = agronomic_filtered[['Год', f'Урожайность ц/га, {region_prefix}',
                                         f'Посевные площади, тыс га, {region_prefix}',
                                         f'Внесено мин. удобрений, тыс. ц., {region_prefix}',
                                         f'Удобрений на посевную площадь, ц/га, {region_prefix}']]

# Фильтруем климатические данные
climate_filtered = climate_data[['Год', f'T, {region_prefix}', f'Po, {region_prefix}', f'U, {region_prefix}',
                                 f'FF, {region_prefix}', f'Tn, {region_prefix}', f'Tx, {region_prefix}',
                                 f'Td, {region_prefix}', f'RRR, {region_prefix}']]

# Фильтруем почвенные данные для региона
soil_filtered = soil_data[[f'Доля почвы в %, {region_prefix}']].copy()
soil_filtered.loc[:, 'Год'] = agronomic_filtered['Год']  # Изменяем с использованием loc

# Фильтруем данные о вегетационном периоде для культуры и региона
vegetation_filtered = vegetation_data[vegetation_data['Культура'] == culture]
vegetation_filtered = vegetation_filtered[['Год', f'Количество дней, {region_prefix}', f'СЭТ, {region_prefix}']]

# Объединяем данные по году
merged_data = agronomic_filtered.merge(climate_filtered, on="Год", how="inner")
merged_data = merged_data.merge(soil_filtered, on="Год", how="left")
merged_data = merged_data.merge(vegetation_filtered, on="Год", how="left")

# Проверяем количество NaN и дублирующих строк
print("Количество NaN в данных:")
print(merged_data.isna().sum())
print(f"Количество дублирующих строк: {merged_data.duplicated().sum()}")

# Удаляем строки с NaN значениями
merged_data = merged_data.dropna()

# Проверка и замена inf значений в X
X = merged_data.drop(columns=[f'Урожайность ц/га, {region_prefix}', 'Год'])
X = X.replace([float('inf'), float('-inf')], pd.NA)  # Заменяем inf на NaN

# Удаляем NaN после замены
X = X.dropna()

# Убедимся, что y соответствует X
y = merged_data[f'Урожайность ц/га, {region_prefix}'].loc[X.index]  # Обеспечиваем соответствие

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель градиентного бустинга
gb_model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Предсказываем урожайность на тестовых данных
y_pred_gb = gb_model.predict(X_test)

# Оцениваем модель градиентного бустинга
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Обучаем модель XGBoost
xgb_model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Предсказываем урожайность на тестовых данных
y_pred_xgb = xgb_model.predict(X_test)

# Оцениваем модель XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Кросс-валидация для обеих моделей
cv_scores_gb = cross_val_score(gb_model, X, y, cv=3)
cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=3)

# Выводим результаты
print("Градиентный бустинг:")
print(f"Mean Squared Error: {mse_gb}")
print(f"R^2 Score: {r2_gb}")
print(f"Cross-Validated R^2 Scores: {cv_scores_gb}")
print(f"Mean Cross-Validated R^2 Score: {cv_scores_gb.mean()}")

print("\nXGBoost:")
print(f"Mean Squared Error: {mse_xgb}")
print(f"R^2 Score: {r2_xgb}")
print(f"Cross-Validated R^2 Scores: {cv_scores_xgb}")
print(f"Mean Cross-Validated R^2 Score: {cv_scores_xgb.mean()}")

# Визуализируем важность признаков для XGBoost
plt.figure(figsize=(10, 6))
plt.title("Feature Importances - XGBoost")
plt.barh(X.columns, xgb_model.feature_importances_, color='b', align='center')
plt.xlabel("Relative Importance")
plt.show()
