import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Experience': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [30,35,40,50,55,65,70,80,85,95]
}

df = pd.DataFrame(data)

X = df[['Experience']]
Y = df['Salary']

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($1000s)")
plt.title("Experience vs Salary")
plt.legend()
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

salary_12 = model.predict([[12]])
salary_15 = model.predict([[15]])

print("Predicted Salary for 12 Years of Experience:", round(salary_12[0], 2), "($1000s)")
print("Predicted Salary for 15 Years of Experience:", round(salary_15[0], 2), "($1000s)")
