import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import random

def studentregression(ages_train, net_worth_train):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(ages_train, net_worth_train)
    return reg
numpy.random.seed(42)
ages = []

for i in range(250):
    ages.append(random.randint(18, 75))

net_worth = [i * 6.25+numpy.random.normal(scale=40) for i in ages]

ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worth = numpy.reshape(numpy.array(net_worth), (len(net_worth), 1))

from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worth_train, net_worth_test = train_test_split(ages, net_worth)

reg1 = studentregression(ages_train, net_worth_train)

print("Coefficient : ", reg1.coef_)
print("Intercept : ", reg1.intercept_)

print("Training data : ", reg1.score(ages_train, net_worth_train))
print("Test data : ", reg1.score(ages_test, net_worth_test))

plt.figure(figsize=(12, 10))
sns.regplot(x=ages_train, y=net_worth_train, scatter=True, color='blue', marker="*")
plt.xlabel("Ages Train")
plt.ylabel("Net worth Train")
plt.title("tech sem")

plt.figure(figsize=(12, 10))
plt.scatter(ages_train, net_worth_train, color='b', label='train data', marker="*")
plt.scatter(ages_test, net_worth_test, color='r', label='test data', marker="*")
plt.plot(ages_test, reg1.predict(ages_test))
plt.xlabel("Ages")
plt.ylabel("Net worth")
plt.legend(loc=2)
plt.show()