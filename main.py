import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('1.csv')
years = df['Рік']
schools_count = df['Кількість навчальних закладів']
plt.figure(figsize=(10, 6))
plt.plot(years, schools_count, marker='o', linestyle='-')
plt.title('Протягом наступних 10 років прогнозується зменшення кількості навчальних закладів')
plt.xlabel('Рік')
plt.ylabel('Кількість навчальних закладів')
plt.grid(True)
plt.xticks(years, rotation=45)  # Обернемо підписи по осі X на 45 градусів
plt.tight_layout()
plt.show()

df = pd.read_csv('2.csv')
categories = df['Категорія зарплат']
employee_count = df['Кількість співробітників']
plt.figure(figsize=(10, 6))
plt.bar(categories, employee_count, color='skyblue')
plt.xlabel('Категорія зарплат')
plt.ylabel('Кількість співробітників')
plt.title('Більшість співробітників отримує зарплату від 10 до 15 тисяч гривень')
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Додамо сітку тільки по вертикальній осі
plt.xticks(rotation=45)  # Обернемо підписи по осі X на 45 градусів
plt.tight_layout()
plt.show()

df = pd.read_csv('3.csv')
years = df['Рік'].unique()
fuels = df['Вид палива'].unique()
prices = [df[df['Вид палива'] == fuel]['Ціна'] for fuel in fuels]
quality = [df[df['Вид палива'] == fuel]['Якість'] for fuel in fuels]

# Побудова графіка цін на різних видах палива впродовж років
plt.figure(figsize=(10, 6))
for fuel, price in zip(fuels, prices):
    plt.plot(years, price, marker='o', label=f'Паливо {fuel}')
plt.xlabel('Рік')
plt.ylabel('Ціна')
plt.title('Динаміка цін на різних видах палива впродовж років')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for fuel, price in zip(fuels, quality):
    plt.plot(years, price, marker='o', label=f'Паливо {fuel}')
plt.xlabel('Рік')
plt.ylabel('Якість')
plt.title('Якість різних видів палива впродовж років')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

df = pd.read_csv('4.csv', delimiter=',', decimal=',')
months = df['Місяць']
data = df.drop(columns=['Місяць'])
data = data.astype(float)
plt.figure(figsize=(10, 6))
for column in data.columns:
    plt.plot(months, data[column], marker='o', label=f'Колонка {column}')

plt.xlabel('Місяць')
plt.ylabel('Значення')
plt.title('У вересні рівень плинності кадрів у шести підрозділах був приблизно однаковий')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

df = pd.read_csv('5.csv')
categories = df['Часовий проміжок']
percentages = df['Відсоток']
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90)
plt.title('Продавець магазину проводить з клієнтами лише 15% свого робочого часу')
plt.axis('equal')
plt.show()

df = pd.read_csv('6.csv')
work_experience = df['Трудовий стаж']
bonus_salary = df['Надбавка зарплати']
plt.figure(figsize=(10, 6))
plt.plot(work_experience, bonus_salary, marker='o', linestyle='-')
plt.title('Розмір надбавки зарплати не залежить від трудового стажу')
plt.xlabel('Трудовий стаж')
plt.ylabel('Надбавка зарплати')
plt.grid(True)
plt.xticks(work_experience)
plt.tight_layout()
plt.show()

df = pd.read_csv('7.csv')
years = df['Рік'].unique()
plt.figure(figsize=(10, 6))

for year in years:
    data_year = df[df['Рік'] == year]
    age_groups = data_year['Вікова група']
    turnover_percentage = data_year['Відсоток плинності кадрів']
    plt.plot(age_groups, turnover_percentage, marker='o', label=f'Рік {year}')

plt.title('Торік найбільш опинність кадрів спостерігалася у віковій групі від 30 до 35 років')
plt.xlabel('Вікова група')
plt.ylabel('Відсоток плинності кадрів')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

df = pd.read_csv('8.csv')
regions = df['Регіон']
births_count = df['Кількість народжень']
plt.figure(figsize=(10, 6))
plt.bar(regions, births_count, color='skyblue')
plt.xlabel('Регіон')
plt.ylabel('Кількість народжень')
plt.title('Центральний регіон займає останнє місце з народжуваності')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

df = pd.read_csv('9.csv')
months = df['Місяць']
profitability = df['Прибутковість акцій (%)']
plt.figure(figsize=(10, 6))
plt.plot(months, profitability, marker='o', color='orange', linestyle='-')
plt.xlabel('Місяць')
plt.ylabel('Прибутковість акцій (%)')
plt.title('Прибутковість акцій нашої компанії скорочуеться')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)  # Обертаємо позначки на осі X
plt.show()

df = pd.read_csv('10.csv')
categories = df['Категорія']
funds_share = df['Частка фондів (%)']
plt.figure(figsize=(8, 8))
plt.pie(funds_share, labels=categories, autopct='%1.1f%%', startangle=140)
plt.title('Найбільша частка фондів задіяна у виробництві')
plt.axis('equal')
plt.legend()
plt.show()

df = pd.read_csv('11.csv')
persons = df['Номер особи']
income = df['Дохід (у гривнях)']
salary = df['Зарплата (у гривнях)']
plt.figure(figsize=(10, 6))
plt.bar(persons, income, color='skyblue', label='Дохід')
plt.bar(persons, salary, color='orange', label='Зарплата')
plt.xlabel('Номер особи')
plt.ylabel('Сума (у гривнях)')
plt.title('Спостерігається зв\'язок між доходами і зарплатою')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

df = pd.read_csv('12.csv')
months = df['Місяць']
students_data = df.drop(columns=['Місяць'])

# Побудова графіка
plt.figure(figsize=(10, 6))

for student in students_data.columns:
    plt.plot(months, students_data[student], marker='o', label=student)

plt.title(' У серпні два студенти обігнали за успішністю шість інших')
plt.xlabel('Місяць')
plt.ylabel('Результат')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
