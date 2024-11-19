import pandas as pd

# Задаем параметры
file_path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Погода\Дни Тат 2005-2023.xlsx"
output_path = r"C:\Users\Stepan\Documents\Машинное обучение для урожайности\Вегетационный_период_свеклы_Тат.xlsx"
temperature_threshold = 3  # Установите пороговую температуру

# Читаем данные из Excel
data = pd.read_excel(file_path)

# Преобразуем столбец "Местное время" в формат datetime
data['Местное время'] = pd.to_datetime(data['Местное время'])

# Сортируем данные по времени
data.sort_values(by='Местное время', inplace=True)

# Создаем пустой DataFrame для хранения результатов
results = pd.DataFrame(columns=["Год", "Начало вегетации", "Конец вегетации", "Количество дней", "СЭТ"])

# Группируем данные по годам
data['Год'] = data['Местное время'].dt.year

for year, group in data.groupby('Год'):
    # Считываем среднюю температуру
    temperatures = group['T_avg']

    count = 0
    start = None
    max_duration = 0
    best_start = None
    best_end = None
    total_sums = 0  # Сумма Эффективных Температур

    for index, temp in enumerate(temperatures):
        if temp > temperature_threshold:
            count += 1
            if count == 5:  # Если 5 дней подряд
                if start is None:  # Начало вегетации
                    start = group.iloc[index - 4]['Местное время']  # Начало с 5-го дня
        else:
            if count >= 5:  # Если предыдущие дни были выше порога
                end = group.iloc[index - 1]['Местное время']  # Конец вегетации
                duration = (end - start).days + 1  # Общее количество дней

                # Проверяем, является ли текущий период максимальным
                if duration > max_duration:
                    max_duration = duration
                    best_start = start
                    best_end = end

                    # Расчет СЭТ для текущего периода
                    total_sums = temperatures.iloc[index - duration:index].clip(
                        lower=temperature_threshold).sum() - temperature_threshold * duration

            count = 0  # Сброс счетчика

    # Проверка на случай, если вегетация закончилась в конце данных
    if count >= 5:
        end = group.iloc[-1]['Местное время']  # Конец вегетации
        duration = (end - start).days + 1  # Общее количество дней

        # Проверяем, является ли текущий период максимальным
        if duration > max_duration:
            max_duration = duration
            best_start = start
            best_end = end

            # Расчет СЭТ для текущего периода
            total_sums = temperatures.iloc[-duration:].clip(
                lower=temperature_threshold).sum() - temperature_threshold * duration

    # Если найден вегетационный период, добавляем его в результаты
    if best_start is not None and best_end is not None:
        results = pd.concat([results, pd.DataFrame([{"Год": year, "Начало вегетации": best_start,
                                                     "Конец вегетации": best_end,
                                                     "Количество дней": max_duration,
                                                     "СЭТ": total_sums}])],
                            ignore_index=True)

# Сохраняем результаты в новый Excel файл
results.to_excel(output_path, index=False)

print("Данные о вегетационном периоде и СЭТ успешно сохранены.")
