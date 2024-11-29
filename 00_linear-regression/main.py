import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56, 30, 45, 60, 25, 40, 50, 10,
                         55, 37, 20, 48, 15, 60, 38, 42, 55, 30, 47, 30, 49, 51, 40, 22, 28, 43, 60, 52, 
                         25, 34, 53, 26]).reshape(-1, 1)

scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 8, 58, 72, 88, 55, 79, 90, 40,
                   89, 75, 53, 80, 45, 85, 70, 78, 91, 66, 82, 74, 87, 85, 73, 69, 75, 84, 92, 83, 
                   60, 72, 79, 67]).reshape(-1, 1)

time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.2)

model = LinearRegression()
model.fit(time_train, score_train)

print(model.score(time_test, score_test))

plt.scatter(time_studied, scores, color="blue", label="Data")
plt.plot(np.linspace(0, 70, 100).reshape(-1, 1), model.predict(np.linspace(0,70,100).reshape(-1, 1)), 'r')
plt.ylim(0, 100)
plt.savefig("plot.png")