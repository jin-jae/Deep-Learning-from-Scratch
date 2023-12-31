import numpy as np

# array
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# calculation
y = np.array([2.0, 4.0, 6.0])
print("x + y:", x + y)
print("x - y:", x - y)
print("x * y:", x * y)
print("x / y:", x / y)
print("x / 2.0", x / 2.0)

# multi dimensional array
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)
B = np.array([[3, 0], [0, 6]])
print("A + B:", A + B)
print("A * B:", A * B)

print(A * 10)

# broadcast
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print("A * B", A * B)

# indexing
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)

X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])
print(X > 15)
print(X[X > 15])
