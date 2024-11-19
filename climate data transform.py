import pandas as pd

# Путь к файлу с данными
file_path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Орел 2005-2023.xlsx"

# Загрузка данных из Excel
df = pd.read_excel(file_path)

# Преобразуем столбец с датой в формат datetime (проверьте название столбца)
df['Местное время'] = pd.to_datetime(df['Местное время'], dayfirst=True)

# Извлекаем год, месяц и день для группировки
df['Год'] = df['Местное время'].dt.year
df['Месяц'] = df['Местное время'].dt.month
df['День'] = df['Местное время'].dt.day

# Определим декаду (1-10, 11-20, 21-31)
df['Декада'] = pd.cut(
    df['День'],
    bins=[0, 10, 20, 31],  # Три декады
    labels=[1, 2, 3],       # Метки для декад
    right=True              # Включаем правую границу
)

# Группировка по году, месяцу и декаде
dekada_avg = df.groupby(['Год', 'Месяц', 'Декада']).agg(
    T_avg=('T', 'mean'),           # Средняя температура
    Po_avg=('Po', 'mean'),         # Среднее давление
    U_avg=('U', 'mean'),           # Средняя влажность
    FF_avg=('Ff', 'mean'),         # Скорость ветра
    Tn_avg=('Tn', 'mean'),         # Минимальная температура
    Tx_max=('Tx', 'mean'),         # Максимальная температура
    Td_max=('Td', 'mean'),         # Точка росы
    RRR_mean=('RRR', 'mean'),      # Сумма осадков
    Tg_mean=('Tg', 'mean'),        # Мин. темп. пов. почвы
    sss_mean=('sss', 'mean')       # Высота снежного покрова
).reset_index()

# Сохраняем результат в новый Excel-файл
output_path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Светлоград 2006-2009 по декадам.xlsx"
dekada_avg.to_excel(output_path, index=False)

print("Усреднённые данные по декадам успешно сохранены в:", output_path)
