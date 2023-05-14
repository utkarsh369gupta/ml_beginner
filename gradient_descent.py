import numpy as np

def grad(x, y):
    n = len(x)
    m_curr = b_curr = 0
    iterations = 100
    learning_rate = 0.08
    for i in range(iterations):
        y_pr = m_curr*x+b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_pr)])
        md = -(2/n)*sum(x*(y-y_pr))
        bd = -(2/n)*sum((y-y_pr))
        m_curr = m_curr-learning_rate*md
        b_curr = b_curr-learning_rate*bd
        print(f'm is {m_curr}, b is {b_curr}, cost is {cost}, iteration is {i} ')

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])
grad(x, y)


'''
to find gradient descent we have used mean square error mse:
therefore mse = (1/n) sumation([y-(mx+b)]^2,1,n)
'''