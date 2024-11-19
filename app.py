import pandas as pd
import os
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

# Уникальные значения культур и регионов
cultures = agronomic_data['Культура'].unique()
regions = ['Орл', 'Став', 'Тат']  # Пример регионов, вы можете добавить свои

# Заголовок приложения
st.title("Анализ урожайности")

# Выбор культуры
selected_culture = st.selectbox("Выберите культуру:", cultures)

# Выбор региона
selected_region = st.selectbox("Выберите регион:", regions)

# Выбор переменной для визуализации
variables = [
    f'Урожайность ц/га, {selected_region}',
    f'Посевные площади, тыс га, {selected_region}',
    f'Внесено мин. удобрений, тыс. ц., {selected_region}',
    f'Удобрений на посевную площадь, ц/га, {selected_region}'
]

selected_variable = st.selectbox("Выберите переменную для визуализации:", variables)

# Фильтрация данных на основе выбора
filtered_data = agronomic_data[agronomic_data['Культура'] == selected_culture]

# Подготовка данных для графика
# Убедимся, что Год является индексом для правильной визуализации
filtered_data.reset_index(drop=True, inplace=True)  # Сброс индексов
filtered_data['Год'] = filtered_data['Год'].astype(str)  # Преобразуем год в строку для оси X

# Визуализация данных с использованием plotly
fig = px.line(filtered_data, x='Год', y=selected_variable, title=f"{selected_culture} - {selected_variable}",
              labels={'Год': 'Год', selected_variable: selected_variable})
fig.update_xaxes(tickvals=filtered_data['Год'], tickangle=45)  # Убедимся, что годы отображаются
fig.update_layout(xaxis_fixedrange=True)  # Зафиксируем ось X

st.write(f"Выбранная переменная: {selected_variable}")
st.plotly_chart(fig)

# Выбор года для отображения вегетационного периода
selected_year = st.selectbox("Выберите год для отображения вегетационного периода:", filtered_data['Год'].unique())

# Фильтруем данные о вегетационном периоде для выбранного года и культуры
vegetation_filtered = vegetation_data[
    (vegetation_data['Культура'] == selected_culture) &
    (vegetation_data['Год'] == int(selected_year))
    ]

# Вывод вегетационного периода
if not vegetation_filtered.empty:
    start_date = vegetation_filtered[f'Начало вегетации, {selected_region}'].values[0]
    end_date = vegetation_filtered[f'Конец вегетации, {selected_region}'].values[0]
    duration = vegetation_filtered[f'Количество дней, {selected_region}'].values[0]

    # Преобразуем даты в нужный формат
    start_date_str = pd.to_datetime(start_date).strftime("%d %B %Y")
    end_date_str = pd.to_datetime(end_date).strftime("%d %B %Y")

    st.write(f"Вегетационный период для {selected_culture} в {selected_year}:")
    st.write(f"Начало: {start_date_str}, Конец: {end_date_str}, \nПродолжительность: {duration} дней")
else:
    st.write(f"Данные о вегетационном периоде для {selected_culture} в {selected_year} отсутствуют.")

# Визуализация почвенных данных круговой диаграммой
soil_filtered = soil_data[[f'Доля почвы в %, {selected_region}']].copy()
soil_filtered['Наименование почвы'] = soil_data['Наименование почвы']  # Сохраним наименование почвы

# Убираем нулевые значения
soil_filtered = soil_filtered[soil_filtered[f'Доля почвы в %, {selected_region}'] > 1.0]

# Создаем круговую диаграмму, если остались данные
if not soil_filtered.empty:
    soil_fig = px.pie(soil_filtered,
                      values=f'Доля почвы в %, {selected_region}',
                      names='Наименование почвы',
                      title='Распределение доли почвы по типам')
    st.plotly_chart(soil_fig)
else:
    st.write("Нет доступных данных для отображения.")

# Корреляционная матрица
st.header("Корреляционная матрица")

agronomic_filtered = agronomic_data[agronomic_data['Культура'] == selected_culture]
agronomic_filtered = agronomic_filtered[
    ['Год', f'Урожайность ц/га, {selected_region}', f'Посевные площади, тыс га, {selected_region}',
     f'Внесено мин. удобрений, тыс. ц., {selected_region}', f'Удобрений на посевную площадь, ц/га, {selected_region}']]

vegetation_filtered = vegetation_data[
    (vegetation_data['Культура'] == selected_culture)
]

vegetation_filtered = vegetation_filtered[['Год', f'Начало вегетации, {selected_region}',
                                           f'Конец вегетации, {selected_region}',
                                           f'Количество дней, {selected_region}', f'СЭТ, {selected_region}']]

# Фильтрация климатических данных по вегетационному периоду
climate_filtered = pd.DataFrame()
for _, row in vegetation_filtered.iterrows():
    year = row['Год']
    start_date = pd.to_datetime(row[f'Начало вегетации, {selected_region}'])
    end_date = pd.to_datetime(row[f'Конец вегетации, {selected_region}'])

    climate_year = climate_data[
        (climate_data['Год'] == year) &
        (climate_data['Дата'] >= start_date) &
        (climate_data['Дата'] <= end_date)
    ]
    if not climate_year.empty:
        climate_year_avg = climate_year.mean(numeric_only=True)
        climate_year_avg['Год'] = year
        climate_filtered = pd.concat([climate_filtered, climate_year_avg.to_frame().T], ignore_index=True)

climate_filtered = climate_filtered[[
    f'T, {selected_region}', f'Po, {selected_region}', f'U, {selected_region}',
    f'FF, {selected_region}', f'Tn, {selected_region}', f'Tx, {selected_region}',
    f'Td, {selected_region}', f'RRR, {selected_region}', f'Tg, {selected_region}', 'Год']]

merged_data = agronomic_filtered.merge(vegetation_filtered, on="Год", how="inner")
merged_data = merged_data.merge(climate_filtered, on="Год", how="inner")
merged_data_no_year = merged_data.drop(columns=['Год'])

# Рассчитываем и визуализируем корреляционную матрицу
corr_matrix = merged_data_no_year.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title(f'Корреляционная матрица для {selected_culture} в регионе {selected_region}')
st.pyplot(plt.gcf())
plt.close()

# Статистический анализ
st.header("Статистический анализ урожайности")

# Фильтруем агрономические данные
agronomic_filtered = agronomic_data[agronomic_data['Культура'] == selected_culture]
agronomic_filtered = agronomic_filtered[['Год', f'Урожайность ц/га, {selected_region}',
                                         f'Посевные площади, тыс га, {selected_region}',
                                         f'Внесено мин. удобрений, тыс. ц., {selected_region}',
                                         f'Удобрений на посевную площадь, ц/га, {selected_region}']]

# Фильтруем климатические данные
climate_filtered = climate_data[['Год', f'T, {selected_region}', f'Po, {selected_region}', f'U, {selected_region}',
                                 f'FF, {selected_region}', f'Tn, {selected_region}', f'Tx, {selected_region}',
                                 f'Td, {selected_region}', f'RRR, {selected_region}', f'Tg, {selected_region}']]

# Фильтруем почвенные данные для региона
soil_filtered = soil_data[[f'Доля почвы в %, {selected_region}']].copy()
soil_filtered.loc[:, 'Год'] = agronomic_filtered['Год']  # Изменяем с использованием loc

# Фильтруем данные о вегетационном периоде для культуры и региона
vegetation_filtered = vegetation_data[vegetation_data['Культура'] == selected_culture]
vegetation_filtered = vegetation_filtered[['Год', f'Количество дней, {selected_region}', f'СЭТ, {selected_region}']]

# Объединяем данные по году
merged_data = agronomic_filtered.merge(climate_filtered, on="Год", how="inner")
merged_data = merged_data.merge(soil_filtered, left_on="Год", right_on="Год", how="left")
merged_data = merged_data.merge(vegetation_filtered, on="Год", how="left")

# Проверка и замена NaN/inf значений в X
X = merged_data.drop(columns=[f'Урожайность ц/га, {selected_region}', 'Год'])
X = X.replace([float('inf'), float('-inf')], pd.NA)  # Заменяем inf на NaN
X = X.fillna(0)  # Заполняем NaN нулями, можно также использовать X.mean() или другой метод

y = merged_data[f'Урожайность ц/га, {selected_region}']

# Добавляем константу для регрессии
X = sm.add_constant(X)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель множественной линейной регрессии
model = sm.OLS(y_train, X_train).fit()

# Предсказываем урожайность на тестовых данных
y_pred = model.predict(X_test)

# Оцениваем модель по среднеквадратической ошибке и R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
st.write(f"Коэффициент детерминации (R^2): {r2:.2f}")

# Модель машинного обучения
st.header("Модель машинного обучения для анализа урожайности")
# Фильтруем агрономические данные
agronomic_filtered = agronomic_data[agronomic_data['Культура'] == selected_culture]
agronomic_filtered = agronomic_filtered[['Год', f'Урожайность ц/га, {selected_region}',
                                         f'Посевные площади, тыс га, {selected_region}',
                                         f'Внесено мин. удобрений, тыс. ц., {selected_region}',
                                         f'Удобрений на посевную площадь, ц/га, {selected_region}']]

# Фильтруем климатические данные
climate_filtered = climate_data[['Год', f'T, {selected_region}', f'Po, {selected_region}', f'U, {selected_region}',
                                 f'FF, {selected_region}', f'Tn, {selected_region}', f'Tx, {selected_region}',
                                 f'Td, {selected_region}', f'RRR, {selected_region}']]

# Фильтруем почвенные данные для региона
soil_filtered = soil_data[[f'Доля почвы в %, {selected_region}']].copy()
soil_filtered.loc[:, 'Год'] = agronomic_filtered['Год']  # Изменяем с использованием loc

# Фильтруем данные о вегетационном периоде для культуры и региона
vegetation_filtered = vegetation_data[vegetation_data['Культура'] == selected_culture]
vegetation_filtered = vegetation_filtered[['Год', f'Количество дней, {selected_region}', f'СЭТ, {selected_region}']]

# Объединяем данные по году
merged_data = agronomic_filtered.merge(climate_filtered, on="Год", how="inner")
merged_data = merged_data.merge(soil_filtered, on="Год", how="left")
merged_data = merged_data.merge(vegetation_filtered, on="Год", how="left")

# Удаляем строки с NaN значениями
merged_data = merged_data.dropna()

# Проверка и замена inf значений в X
X = merged_data.drop(columns=[f'Урожайность ц/га, {selected_region}', 'Год'])
X = X.replace([float('inf'), float('-inf')], pd.NA)  # Заменяем inf на NaN

# Удаляем NaN после замены
X = X.dropna()

# Убедимся, что y соответствует X
y = merged_data[f'Урожайность ц/га, {selected_region}'].loc[X.index]  # Обеспечиваем соответствие

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель Random Forest
model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)  # Увеличили количество деревьев
model.fit(X_train, y_train)

# Предсказываем урожайность на тестовых данных
y_pred = model.predict(X_test)

# Оцениваем модель по среднеквадратической ошибке, R^2, MAE и RMSE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE

# Кросс-валидация
cv_scores = cross_val_score(model, X, y, cv=5)  # Используем 5-кратную кросс-валидацию

st.write(f"Среднеквадратичная ошибка (MSE): {mse}")
st.write(f"Коэффициент детерминации (R^2): {r2_score(y_test, y_pred)}")