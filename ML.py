import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем данные
path = r"C:\Users\Stepan\Documents\РГАУ МСХА\Статьи\Урожайность\Машинное обучение для урожайности\Датасеты"
agronomic_data_path = os.path.join(path, "Агрономические данные.xlsx")
climate_data_path = os.path.join(path, "Климат по 10дням.xlsx")
soil_data_path = os.path.join(path, "Почвенные данные.xlsx")
vegetation_data_path = os.path.join(path, "Вегетационный период.xlsx")

agronomic_data = pd.read_excel(agronomic_data_path, decimal=',')
climate_data = pd.read_excel(climate_data_path, decimal=',')
soil_data = pd.read_excel(soil_data_path, decimal=',')
vegetation_data = pd.read_excel(vegetation_data_path, decimal=',')

# Пример фильтрации данных для картофеля в регионе Орловская область
culture = "Пшеница"
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

# Обучаем модель Random Forest
model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)  # Увеличили количество деревьев
model.fit(X_train, y_train)

# Предсказываем урожайность на тестовых данных
y_pred = model.predict(X_test)

# Оцениваем модель по среднеквадратической ошибке, R^2,ыы MAE и RMSE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2_score(y_test, y_pred)}")


# График остатков
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()


