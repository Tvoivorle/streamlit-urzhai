import pandas as pd

# Путь к файлу с данными
file_path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Погода\Часы Светлоград.xlsx"

# Загрузка данных из Excel
df = pd.read_excel(file_path)

# Преобразуем столбец с датой и временем в формат datetime (убедитесь, что название столбца верное)
df['Местное время'] = pd.to_datetime(df['Местное время'], dayfirst=True)

# Устанавливаем 'Местное время' как индекс, чтобы использовать resample
df = df.set_index('Местное время')

# Агрегируем данные в средние значения за день
daily_avg = df.resample('D').agg(
    T_avg=('T', 'mean'),          # Средняя температура
    Po_avg=('Po', 'mean'),        # Среднее давление
    U_avg=('U', 'mean'),          # Средняя влажность
    FF_avg=('Ff', 'mean'),        # Средняя скорость ветра
    Tn_avg=('Tn', 'mean'),        # Минимальная температура
    Tx_max=('Tx', 'mean'),        # Максимальная температура
    Td_max=('Td', 'mean'),        # Средняя точка росы
    RRR_sum=('RRR', 'sum'),       # Сумма осадков за день
    Tg_mean=('Tg', 'mean'),       # Средняя мин. темп. почвы
    sss_mean=('sss', 'mean')      # Средняя высота снежного покрова
).reset_index()

# Сохраним результат в новый Excel-файл
output_path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Погода\Дни Светлоград.xlsx"
daily_avg.to_excel(output_path, index=False)

print("Среднесуточные данные успешно сохранены в:", output_path)
