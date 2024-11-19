import pandas as pd

# Загрузка данных
agro_data = pd.read_excel(r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты\Агрономические данные.xlsx")
vegetation_data = pd.read_excel(r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты\Вегетационный период.xlsx")
climate_data = pd.read_excel(r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты\Климат по 10дням.xlsx")
soil_data = pd.read_excel(r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты\Почвенные данные.xlsx")

# Объединение агрономических данных и данных по вегетации
merged_data = agro_data.merge(vegetation_data, on=['Культура', 'Год'], how='left')

# Извлечение значений почвенных данных для регионов
soil_orel = soil_data['Доля почвы в %, Орл'].values[0]
soil_stav = soil_data['Доля почвы в %, Став'].values[0]
soil_tat = soil_data['Доля почвы в %, Тат'].values[0]

# Добавление этих значений как постоянных столбцов
merged_data['Доля почвы в %, Орл'] = soil_orel
merged_data['Доля почвы в %, Став'] = soil_stav
merged_data['Доля почвы в %, Тат'] = soil_tat

# Объединение с климатическими данными по годам и регионам
merged_data = merged_data.merge(climate_data[['Год', 'T, Орл', 'RRR, Орл']], on='Год', how='left', suffixes=('', '_Орл'))
merged_data = merged_data.merge(climate_data[['Год', 'T, Став', 'RRR, Став']], on='Год', how='left', suffixes=('', '_Став'))
merged_data = merged_data.merge(climate_data[['Год', 'T, Тат', 'RRR, Тат']], on='Год', how='left', suffixes=('', '_Тат'))

# Сохранение объединенного датасета для дальнейшего анализа
merged_data.to_excel(r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Датасеты\Данные.xlsx", index=False)
