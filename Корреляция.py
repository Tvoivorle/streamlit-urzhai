import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Задаем пути к файлам
path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты"
agronomic_data_path = os.path.join(path, "Агрономические данные.xlsx")
vegetation_data_path = os.path.join(path, "Вегетационный период.xlsx")
climate_data_path = os.path.join(path, "Климат по 10дням.xlsx")
soil_data_path = os.path.join(path, "Почвенные данные.xlsx")

# Загружаем данные
agronomic_data = pd.read_excel(agronomic_data_path, decimal=',')
vegetation_data = pd.read_excel(vegetation_data_path, decimal=',')
climate_data = pd.read_excel(climate_data_path, decimal=',')
soil_data = pd.read_excel(soil_data_path, decimal=',')

# Пример фильтрации данных для картофеля в регионе Орловская область
culture = "Сахарная свекла"
region_prefix = "Тат"

# Фильтруем агрономические данные для культуры "Картофель"
agronomic_filtered = agronomic_data[agronomic_data['Культура'] == culture]

# Оставляем нужные столбцы и меняем их для удобства анализа
agronomic_filtered = agronomic_filtered[['Год', f'Урожайность ц/га, {region_prefix}',
                                         f'Посевные площади, тыс га, {region_prefix}',
                                         f'Внесено мин. удобрений, тыс. ц., {region_prefix}',
                                         f'Удобрений на посевную площадь, ц/га, {region_prefix}']]

# Фильтруем вегетационные данные
vegetation_filtered = vegetation_data[vegetation_data['Культура'] == culture][['Год',
                                                                                  f'Количество дней, {region_prefix}',
                                                                                  f'СЭТ, {region_prefix}']]

# Создаем пустой DataFrame для агрегированных климатических данных
climate_filtered = pd.DataFrame()

# Фильтруем климатические данные по вегетационному периоду
veg_period = vegetation_data[vegetation_data['Культура'] == culture]
for _, row in veg_period.iterrows():
    year = row['Год']
    start_date = row[f'Начало вегетации, {region_prefix}']
    end_date = row[f'Конец вегетации, {region_prefix}']

    # Фильтруем климатические данные по вегетационному периоду
    climate_year = climate_data[(climate_data['Год'] == year) &
                                (climate_data['Дата'] >= start_date) &
                                (climate_data['Дата'] <= end_date)]

    # Проверяем, что климатические данные не пустые
    if not climate_year.empty:
        # Усредняем показатели за вегетационный период
        climate_year_avg = climate_year.mean(numeric_only=True)
        climate_year_avg['Год'] = year  # Добавляем год к агрегированным данным
        climate_filtered = pd.concat([climate_filtered, climate_year_avg.to_frame().T], ignore_index=True)

# Убираем лишние столбцы, оставляя только нужные для анализа
climate_filtered = climate_filtered[[f'T, {region_prefix}', f'Po, {region_prefix}', f'U, {region_prefix}',
                                     f'FF, {region_prefix}', f'Tn, {region_prefix}', f'Tx, {region_prefix}',
                                     f'Td, {region_prefix}', f'RRR, {region_prefix}', f'Tg, {region_prefix}',
                                     'Год']]

# Объединяем по году
merged_data = agronomic_filtered.merge(vegetation_filtered, on="Год", how="inner")
merged_data = merged_data.merge(climate_filtered, on="Год", how="inner")

# Убираем столбец 'Год' перед расчетом корреляций
merged_data_no_year = merged_data.drop(columns=['Год'])

# Строим корреляционную матрицу для всех пересекающихся данных без столбца 'Год'
corr_matrix = merged_data_no_year.corr()

# Визуализируем корреляционную матрицу
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title(f'Корреляционная матрица для {culture} в регионе {region_prefix}')

# Сохраняем изображение
image_path = os.path.join(path, f"Корреляционная матрица_{culture}_{region_prefix}.png")
plt.savefig(image_path)

# Отображаем график
plt.show()

# Закрываем текущую фигуру
plt.close()

# Сохраняем корреляционную матрицу в Excel
output_path = os.path.join(path, f"Корреляционная матрица_{culture}_{region_prefix}.xlsx")
corr_matrix.to_excel(output_path)
