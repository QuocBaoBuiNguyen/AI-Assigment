import numpy as np
import matplotlib.pyplot as plt
import  random

X1 = np.array(np.random.randn(50,3))
X_train =  np.array(X1[:,0])
Y_train =  np.array(X1[:,1])
Z_train =  np.array(X1[:,2])

X_train = np.transpose(X_train)
Y_train = np.transpose(Y_train)
Z_train = np.transpose(Z_train)
#T = np.concatenate((np.ones((1, 5)), -1*np.ones((1, 5))), axis = 1)
hs = [1]*50
X = np.array((X_train, Y_train, Z_train, hs))
x = X[0,:]
y = X[1,:]
z = X[2,:]

mp = np.array(np.random.rand(1,4))

w=np.array([0.1 ,1, 0.1, -0.5])
n=len(X_train)

T = []

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range (n):
	if mp[0,0]*X_train[i] + mp[0,1]*Y_train[i] + mp[0,2]*Z_train[i] + mp[0,3]>0:
		ax.scatter(X_train[i],Y_train[i],Z_train[i], color='r')
		T.append(1)
	else:
		ax.scatter(X_train[i],Y_train[i],Z_train[i], color='b')
		T.append(-1)
def func(k):
    if k > 0:
        return 1
    else:
        return 0
    #return 1/(1+np.exp(-k))

i = 0
lr = 0.01
while i < n:
        a = X[:, i]
        l = T[i]
        k = w @ a  # dot  bao gồm bias tương tự ouput a là cái  điểm x,y
        update = l - func(k)   #1 hoặc -1
        if update>0.01:
            a = np.transpose(a)
            w = w + lr*update * a
        else:
            i = i + 1
    # l = k * l
    # if l < 0 :
    #     a = np.transpose(a)
    #     w = w + l * a
    #     i = 1
    # else:
    #     i = i + 1

print(w)
[x, y] = np.meshgrid(x, y)
z = (-w[0] * x - w[1] * y - w[3]) / w[2]
ax.plot_surface(x, y, z, color='yellow')
plt.show()