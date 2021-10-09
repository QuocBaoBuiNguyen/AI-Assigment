from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
#fit=plt.figure()
#ax=fit.add_subplot(111,projection='3d')

#----------------------------------------------------------------------------------------#
# Function
#Hàm bất kì có hơn 2 cực trị
def function(x1,x2):
    #return (x1**2+x2**2);
    return np.cos(x1)+2*np.cos(x2)
    #return np.cos(x1)*x1+np.sin(x2+2)
def partial_derivative(func, var, point):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return misc.derivative(wraps, point[var], dx = 1e-6)

#----------------------------------------------------------------------------------------#
# Plot Function

x1 = np.arange(-5, 5,0.1)
x2 = np.arange(-5, 5,0.1)
xx1,xx2 = np.meshgrid(x1,x2);

#z =xx1**2+xx2**2;
z= np.cos(xx1)+2*np.cos(xx2)
#z = np.cos(xx1)*xx1+np.sin(xx2+2)
h = plt.contourf(xx1,xx2,z)

#ax.plot_surface(xx1,xx2,z,color='yellow')
#plt.show()
#----------------------------------------------------------------------------------------#
# Gradient Descent Main

alpha = 0.1 # learning rate
nb_max_iter = 10 # Epoch
eps = 0.01 # early stop condition

x1_0 = np.array([ -4. ,0.  ,4.]);
x2_0 = np.array([ -4. ,0.  ,4.]);
z0 = []
k=0
for i in range(len(x1_0)):
    for j in range(len(x2_0)):
        z0.append(function(x1_0[i],x2_0[j]))
        plt.scatter(x1_0[i],x2_0[j])
        k+=1
##============================================================================
i, j,  k= 0,0, 0
plt.show()
for i in range(len(x1_0)):
    for j in range(len(x2_0)):
        x1_0 = np.array([ -4. ,0.  ,4.]);
        x2_0 = np.array([ -4. ,0.  ,4.]);
        cond = eps + 10.0 # cho điều kiện dừng lớn hơn nhiều so với eps
        nb_iter = 0 
        tmp_z0 = z0[k]   #biến tmpz0 dùng để cập nhập cond là điều kiện dừng < eps=0,0001(early point stop)
        while cond > eps and nb_iter < nb_max_iter:
                tmp_x1_0 = x1_0[i] - alpha * partial_derivative(function, 0, [x1_0[i],x2_0[j]])   #x' = x -lr * dz/dx
                tmp_x2_0 = x2_0[j] - alpha * partial_derivative(function, 1, [x1_0[i],x2_0[j]])     #y'=y- lr*dz/dy
                x1_0[i] = tmp_x1_0
                x2_0[j] = tmp_x2_0
                z0new = function(x1_0[i],x2_0[j])                # điểm mới được cập nhập (x0',y0',z0')
                nb_iter = nb_iter + 1                  # train nb_max_iter lần
                cond = abs( tmp_z0 - z0new )               # đánh giá sai số
                tmp_z0 = z0new
                plt.scatter(x1_0[i], x2_0[j])
        k+=1
plt.title("Gradient Descent Python (3d)")

plt.show()
input()
