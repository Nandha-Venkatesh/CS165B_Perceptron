import random
import numpy as np
import matplotlib.pyplot as plt

# Generate Random X-Coords. Calculate y-value based on y = x. Then adjust y above or below the line in an alternating fashion.
# Half the points will be above the line and half will be below it.

def generatePoints(size):
    list = [];
    a = 0;
    for i in range(0, size):
        x = random.randint(-10, 10);
        if(a % 2 == 0):
            y = x + random.uniform(0.5, 5);
        else:
            y = x - random.uniform(0.5, 5);
        a = a + 1;
        list.append((x, y));
    return list;


PointList = generatePoints(1000);
# for i in PointList:
    # print("X =", i[0], "Y =", i[1], "\n")

# Plot Initial Points and Target Function f(x) = x

x = list(range(-10, 10));
y = [i for i in x];
plt.plot(x, y, linestyle='solid', label="F(x) = x")
for i in PointList:
    plt.plot(i[0], i[1], marker='o', markersize=3, markerfacecolor="red", markeredgecolor="blue")
plt.title("Linearly Separable Points and Target Function")
plt.legend(loc='upper center')
plt.xlabel("X1 axis")
plt.ylabel("X2 axis")
plt.grid()
plt.show()

# Perceptron Algorithm goes here.

w = np.zeros(3);
timesteps = 0;
b = 0;
while (b != 1):
    a = 0;
    for j in range(len(PointList)):
        x = np.array([1, PointList[j][0], PointList[j][1]]);
        y_star = 1;
        if(x[1] > x[2]):
            y_star = -1;
        if(np.inner(x, w) * y_star) <= 0: # Misclassified Example
            # Update the weight.
            w = w + (y_star * x);
            break;
        else:
            a = a+1;
        if(a == 20): # Converged
            b = 1;
    timesteps = timesteps + 1;

print("Timesteps to Converge:", timesteps);

# Plot everything. Create perceptron line based on w.
a = (-1) * w[1] / w[2];
b = (-1) * w[0] / w[2];

x = [-10, 10];
c = a * -10 + b;
d = a * 10 + b;
y = [c, d];

message = "Linearly Separable Points and Perceptron Line: x_2 = {} * x_1 + {}".format(a, b);
print(message)

plt.plot(x, y, linestyle='solid', label="Perceptron Line")
for i in PointList:
    plt.plot(i[0], i[1], marker='o', markersize=3, markerfacecolor="red", markeredgecolor="blue")
plt.title(message)
plt.xlabel("X1 axis")
plt.ylabel("X2 axis")
plt.grid()
plt.show()
