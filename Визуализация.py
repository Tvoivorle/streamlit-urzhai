import pandas as pd
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

# Фильтруем климатические данные, оставляя только необходимые столбцы
climate_filtered = climate_data[['Год', 'Дата', f'T, {region_prefix}', f'RRR, {region_prefix}']]
climate_filtered.loc[:, 'Дата'] = pd.to_datetime(climate_filtered['Дата'])  # Используем .loc для изменения

# Объединяем данные по году
merged_data = agronomic_filtered.merge(climate_filtered, on='Год', how='inner')

# Устанавливаем 'Дата' как индекс
merged_data.set_index('Дата', inplace=True)

# Плотим урожайность и климатические показатели
plt.figure(figsize=(14, 8))

# Подграфик 1: Урожайность
plt.subplot(2, 1, 1)
plt.plot(merged_data.index, merged_data[f'Урожайность ц/га, {region_prefix}'], marker='o', label='Урожайность', color='orange')
plt.title(f'Урожайность {culture} в регионе {region_prefix}')
plt.ylabel('Урожайность (ц/га)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Подграфик 2: Климатические показатели
plt.subplot(2, 1, 2)
plt.plot(merged_data.index, merged_data[f'T, {region_prefix}'], marker='o', label='Температура', color='blue')
plt.plot(merged_data.index, merged_data[f'RRR, {region_prefix}'], marker='o', label='Осадки', color='green')
plt.title(f'Климатические показатели для {culture} в регионе {region_prefix}')
plt.ylabel('Климатические показатели')
plt.xticks(rotation=45)
plt.legend()
plt.grid()

# Сохранение графиков
plt.tight_layout()
image_path = os.path.join(path, f'Графики_{culture}_{region_prefix}.png')
plt.savefig(image_path)
plt.show()
