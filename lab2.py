import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import t, f
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

if __name__ == '__main__':

    # Reading the data back from CSV
    df_from_csv = pd.read_csv('Income_Loan_Data.csv')

    # Visualizing the data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_from_csv['Annual Income'], kde=True)
    plt.title('Distribution of Annual Income')

    plt.subplot(1, 2, 2)
    sns.histplot(df_from_csv['Max Loan Value'], kde=True)
    plt.title('Distribution of Max Loan Value')

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income', y='Max Loan Value', data=df_from_csv)
    plt.title('Annual Income vs. Max Loan Value')
    plt.xlabel('Annual Income ($)')
    plt.ylabel('Max Loan Value ($)')
    plt.show()

    # Fitting a linear regression model to the data
    model = LinearRegression()
    model.fit(df_from_csv[['Annual Income']], df_from_csv['Max Loan Value'])

    # Predicting max loan values
    predicted_loan_values = model.predict(df_from_csv[['Annual Income']])

    # Plotting the results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Annual Income', y='Max Loan Value', data=df_from_csv, label='Data from CSV')
    plt.plot(df_from_csv['Annual Income'], predicted_loan_values, color='red', label='Linear Regression')
    plt.title('Linear Regression on Max Loan Value vs. Annual Income')
    plt.xlabel('Annual Income ($)')
    plt.ylabel('Max Loan Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculating R-squared for the model
    r_squared = r2_score(df_from_csv['Max Loan Value'], predicted_loan_values)
    print(f'R-squared: {r_squared}')

    correlation_coefficient = np.corrcoef(df_from_csv['Annual Income'], df_from_csv['Max Loan Value'])[0, 1]
    print(f'Correlation Coefficient: {correlation_coefficient}')

    # Performing linear regression analysis to get the slope and intercept for significance testing
    slope, intercept, r_value, p_value, std_err = linregress(df_from_csv['Annual Income'],
                                                             df_from_csv['Max Loan Value'])

    # Displaying slope, intercept, and their p-values
    print(f'Slope: {slope}, Intercept: {intercept}')
    print(f'P-value for Slope: {p_value}')

    t_stat = correlation_coefficient * np.sqrt((len(df_from_csv) - 2) / (1 - correlation_coefficient ** 2))

    # F-тест для R^2
    n = len(df_from_csv)
    p = 1  # Кількість незалежних змінних у моделі
    dfn = p
    dfd = n - p - 1
    f_stat = (r_squared / dfn) / ((1 - r_squared) / dfd)
    p_value_r_squared = f.sf(f_stat, dfn, dfd)

    # Обчислення t-статистики
    t_stat = (correlation_coefficient * np.sqrt(n - 2)) / np.sqrt(1 - correlation_coefficient ** 2)

    # Обчислення p-значення для двостороннього тесту
    p_value = 2 * t.sf(np.abs(t_stat), df=n - 2)

    print(f"t-statistic: {t_stat}, p-value: {p_value}")

    print(f'F-statistic: {f_stat}')

    # Computing the 95% confidence interval for the regression line
    confidence_level = 0.95
    t_value = t.ppf((1 + confidence_level) / 2, len(df_from_csv) - 2)  # t-critical value for 95% CI
    confidence_interval_slope = (slope - t_value * std_err, slope + t_value * std_err)

    # Predicting a mean value for an example income
    example_income = 75000
    predicted_value = slope * example_income + intercept
    confidence_interval_prediction_lower_bound = predicted_value - t_value * std_err * example_income,
    confidence_interval_prediction_upper_bound = predicted_value + t_value * std_err * example_income


    # Displaying confidence intervals
    print(f'95% Confidence Interval for Slope: {confidence_interval_slope}')
    print(f'95% Confidence Interval for Predicted Loan Value at ${example_income}: {confidence_interval_prediction_lower_bound}, {confidence_interval_prediction_upper_bound}')

    # Assessing model adequacy
    if p_value < 0.05:
        print("The model is statistically significant.")
    else:
        print("The model is not statistically significant.")

    a = model.intercept_
    b = model.coef_[0]
    X_values = np.linspace(df_from_csv['Annual Income'].min(), df_from_csv['Annual Income'].max(), 100)
    mean_X = np.mean(df_from_csv['Annual Income'])
    MSE = np.mean((df_from_csv['Max Loan Value'] - (a + b * df_from_csv['Annual Income'])) ** 2)
    n = len(df_from_csv)

    # Обчислення прогнозованих значень та довірчих інтервалів
    Y_pred = a + b * X_values
    SE_Y_pred = np.sqrt(MSE * (1 / n + ((X_values - mean_X) ** 2) / np.sum((df_from_csv['Annual Income'] - mean_X) ** 2)))
    t_critical = t.ppf(0.975, df=n - 2)
    conf_interval_lower = Y_pred - t_critical * SE_Y_pred
    conf_interval_upper = Y_pred + t_critical * SE_Y_pred

    # Візуалізація
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_from_csv['Annual Income'], y=df_from_csv['Max Loan Value'], color='blue', label='Data')
    plt.plot(X_values, Y_pred, color='red', label='Regression Line')
    plt.fill_between(X_values, conf_interval_lower, conf_interval_upper, color='red', alpha=0.2,
                     label='95% Confidence Interval')
    plt.title('Linear Regression with 95% Confidence Interval')
    plt.xlabel('Annual Income ($)')
    plt.ylabel('Max Loan Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

