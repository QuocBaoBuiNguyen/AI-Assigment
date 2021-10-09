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
        # truyền lần lượt x-dx, x, x+dx
    return misc.derivative(wraps, point[var], dx = 1e-6)
#Sử dụng hàm misc.derivative(wraps, point[var], dx = 1e-6) để tính đạo hàm tại 1 điểm,
#hàm này sẽ tính giá trị tại 3 điểm x-dx, x, x+dx, 
#sau đó tính trung bình dz/dx 
#và trả về giá trị là đạo hàm trung bình (Trong bài này sử dụng dx=10^-6)
#
#----------------------------------------------------------------------------------------#
# Plot Function
#Tạo mảng điểm để vẽ đồ thị, từ -5->5 với khoảng là 0.1
x1 = np.arange(-5, 5,0.1)
x2 = np.arange(-5, 5,0.1)
xx1,xx2 = np.meshgrid(x1,x2);

#Hàm có 2 cực trị
#z =xx1**2+xx2**2;
z= np.cos(xx1)+2*np.cos(xx2)
#z = np.cos(xx1)*xx1+np.sin(xx2+2)

#Vẽ đường mức
fig = plt.figure()
ax = fig.add_subplot(111)
h = plt.contourf(xx1,xx2,z)

#ax.plot_surface(xx1,xx2,z,color='yellow')
#plt.show()
#----------------------------------------------------------------------------------------#
# Gradient Descent
#Thiết lập thông số
alpha = 0.1 # learning rate
nb_max_iter = 10 # Lặp
eps = 0.01 # early stop condition

#Tạo mảng 3x3 điểm chạy, đây chính là mảng minh họa cho SGD
x1_0 = np.array([ -4. ,0.  ,4.]);
x2_0 = np.array([ -4. ,0.  ,4.]);
z0 = []
k=0     #số điểm của mảng chạy
for i in range(len(x1_0)):
    for j in range(len(x2_0)):
        z0.append(function(x1_0[i],x2_0[j]))
        plt.scatter(x1_0[i],x2_0[j])
        k+=1

i, j,  k= 0,0, 0

for i in range(len(x1_0)):
    for j in range(len(x2_0)):
        x1_0 = np.array([ -4. ,0.  ,4.]);
        x2_0 = np.array([ -4. ,0.  ,4.]);
        cond = eps + 10.0 
        #  giả sử điều kiện dừng sớm lúc đầu rất lớn nhằm chạy vòng lặp while
        nb_iter = 0 
        tmp_z0 = z0[k]   
        #biến tmpz0 dùng để cập nhập cond là điều kiện dừng < eps=0,0001(early point stop)
        while cond > eps and nb_iter < nb_max_iter:
                tmp_x1_0 = x1_0[i] - alpha * partial_derivative(function, 0, [x1_0[i],x2_0[j]])   
                #x' = x -lr * dz/dx
                tmp_x2_0 = x2_0[j] - alpha * partial_derivative(function, 1, [x1_0[i],x2_0[j]])    
                #y'=y- lr * dz/dy
                ax.plot([x1_0[i], tmp_x1_0], [x2_0[j], tmp_x2_0],  color = 'red')
                x1_0[i] = tmp_x1_0
                x2_0[j] = tmp_x2_0
                z0new = function(x1_0[i],x2_0[j])       
                # điểm θ mới được cập nhập (x0',y0',z0')
                nb_iter = nb_iter + 1                  
                # train nb_max_iter lần
                cond = abs( tmp_z0 - z0new )              
                # đánh giá sai số giữa 2 lần cập nhâp θ
                tmp_z0 = z0new
                plt.scatter(x1_0[i], x2_0[j])
        k+=1
plt.title("Stochastic Gradient Descent Python (3d)")

plt.show()
input()
