import numpy as np 
import matplotlib.pyplot as plt

 
plx =[]
ply =[]
def gradient_Decent(x,y,plx,ply):
    
    m_current = b_current = 0 #initial values of m and b, y=mx+b line formula 
    iterations = 10000 # learning iterations 
    n= len(x) #number of input data 
    learningRate = 0.08

    for i in range(iterations):

        y_predicted = m_current * x + b_current
        cost = (1/n) * sum((y-y_predicted)**2)

        m_derivative = -(2/n) * sum(x*(y-y_predicted))
        b_derevatice = -(2/n) * sum(y-y_predicted)

        m_current = m_current - learningRate * m_derivative
        b_current = b_current -learningRate* b_derevatice

        plx.append(m_current)
        ply.append(b_current)


        print("m {}, b {}, cost {}, iteration {}".format(m_current,b_current,cost, i))


x = np.array([1,2,3,4,5]) 
y= np.array([5,7,9,11,13])

gradient_Decent(x,y,plx,ply)

plt.plot(plx,ply)
plt.show()
