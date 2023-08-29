import numpy as np
import matplotlib.pyplot as plt

def h(x, theta):
    x = np.array(x)
    # Reshape theta to be a column
    theta = np.array(theta).reshape(-1, 1)
    # Compute the dot product of x and theta
    y = np.dot(x, theta)
    return y


def mean_squared_error(y_predicted, y_label):
    # Compute the square differences
    squared_differences = (y_predicted - y_label) ** 2
    # Compute the mean of the squared differences
    mse = np.mean(squared_differences)


    return mse   

class LeastSquaresRegression():
    def __init__(self):
        self.theta_ = None  
        
    def fit(self, X, y):
        # Add bias term to X (column of ones)
        bias = np.ones((X.shape[0], 1))
        # Concatenate the bias column with the input matrix X
        X_bias = np.hstack([bias, X])
        
        # Closed-form solution
        self.theta_ = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
    
        
    def predict(self, X):
        bias = np.ones((X.shape[0], 1))
        X_bias = np.hstack([bias, X])
        
        # Predict
        y_pred = X_bias.dot(self.theta_)
        
        return y_pred
    
#generate some random points    
X = 4 * np.random.rand(100, 1)
y = 10 + 2 * X + np.random.randn(100, 1)

#Plot these points to get a feel of the distribution
plt.scatter(X, y, color='blue')
plt.grid(True)
plt.show()

def bias_column(X):
    #Write a function which adds one to each instance
    bias = np.ones((X.shape[0], 1))
    X_new = np.hstack([bias, X])
    return X_new

X_new = bias_column(X)

print(X[:5])
print(" ---- ")
print(X_new[:5])

model = LeastSquaresRegression()
model.fit(X, y)

print(model.theta_)

y_pred = model.predict(X)

# Plot the original data points 
plt.scatter(X, y, color='blue', label='Original Data')
# Plot the predicted data points
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Original Data and Predicted Line')
plt.legend()
plt.grid(True)
plt.show()

class GradientDescentOptimizer():
    def __init__(self, f, fprime, start, learning_rate = 0.1):
        # The function 
        self.f_ = f
        # The gradient of f
        self.fprime_ = fprime                  
        self.current_ = start                  
        self.learning_rate_ = learning_rate   
        # Save history as attributes
        self.history_ = [start]
        
    def step(self):
        #Compute the new value and update self.current_
        gradient = self.fprime_(self.current_)
        self.current_ = self.current_ - self.learning_rate_ * gradient
        #Append the new value to history
        self.history_.append(self.current_)
        
    def optimize(self, iterations = 100):
        # Use the gradient descent to get closer to the minimum
        for _ in range(iterations):
            self.step()
            
    def getCurrentValue(self) -> np.ndarray:
        # Getter for current_
        return np.array([self.current_[0][0], self.current_[1][0]])
        
    def print_result(self):
        print("Best theta found is " + str(self.current_))
        print("Value of f at this theta: f(theta) = " + str(self.f_(self.current_)))
        print("Value of f prime at this theta: f'(theta) = " + str(self.fprime_(self.current_))) 
        

def f(x):
    b = np.array([2, 6])
    diff = x - b
    return 3 + np.dot(diff, diff)

def fprime(x):
    b = np.array([2, 6])
    return 2 * (x - b)


grad = GradientDescentOptimizer(f, fprime, np.random.normal(size=(2,)), 0.1)
grad.optimize(10)
grad.print_result()

# Plot the function f in 3D
x = np.linspace(-5, 10, 400)
y = np.linspace(-5, 10, 400)
x, y = np.meshgrid(x, y)

# Vectorize your function for 2D inputs
f_vec = np.vectorize(lambda x, y: f(np.array([x, y])))
z = f_vec(x, y)

fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, alpha=0.7, cmap='viridis')

ax.set_title('3D plot of function f')

history = np.array(grad.history_)
ax.scatter(history[:, 0], history[:, 1], [f(val) for val in history], c='r', marker='o')
plt.show()
