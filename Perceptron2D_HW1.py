import numpy as np
import matplotlib.pyplot as plt

##========            data          ==================================
X_train = np.array(np.random.rand(100,2))
#khởi tạo các điểm ngẫu nhiên trong không gian 2D
x = np.array(X_train[:,0])
#tọa độ x của các điểm
y = np.array(X_train[:,1])
#tọa độ y của các điểm
Y_train = []
#mảng chứa giá trị label
n =len(X_train)
weights = np.array([0.5 , 0.5])
#khởi tạo cặp trọng số ngẫu nhiên
bias  = 1.

##=========================================================================

def displayDot(x):
    #hàm chia các điểm khởi tạo thành 2 vùng linear seperable, đồng thời gắn nhãn tương ứng
    # Ví dụ đường thẳng được cho là x - 1 < -y
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i=0
    for i in range(n):
                if i<n:
                    if   X_train[i,0]  - 1 <  -X_train[i,1]:
                            plt.scatter(x[i], y[i], color="red")
                            Y_train.append(1)
                    else:
                            plt.scatter(x[i], y[i], color="blue")
                            Y_train.append(0)
                    i+=1
    return ax, fig, Y_train
##===========================================================================
def func(x):
    #unit step
    if x>0:
        return 1
    return 0
##============================================================================
def has_converged(X_train, weights, y, bias):
    #hàm kiểm tra xem thuật toán đã hội tụ chưa
    Out = []
    for i in range( len(X_train)):
        a = np.array(X_train[i])
        Out.append(func(np.dot(np.array(X_train[i]),weights)+bias))
        # Sau khi có bộ số w, bias đem thử lại với từng điểm X, lấy ra Output từng điểm cho vào mảng Out
        # trả về tất cả giá trị dự đoán tương ứng với các điểm X trong mảng Out để so sánh với Y_train
    print('Output',Out)
    print('Target',y)
    return np.array_equal(Out,y) #Trả về 1( True ) nếu Out == y còn trả về 0( False )
    ##============================================================================
def Perceptron(lr, weights, bias):
    #hàm thuật toán perceptron
    s = 0 #biếm đếm số lần cập nhật của thuật toán
    while True:
        for idx in range(len(X_train)):
                x_i=np.array(X_train[idx])                      # lấy từng điểm
                linear_output = np.dot(x_i, weights) + bias     
                #Tổng các tích input và trọng số cộng thêm bias
                y_predicted = func(linear_output)
                #Cho qua hàm unit step
                update = (Y_train[idx] - y_predicted)
                #update = 0, 1 hoặc -1, căn cứ để cập nhập w, bias hay không
                weights += lr * update * x_i
                bias    += lr * update
                #cập nhập
        if has_converged(X_train,weights,Y_train, bias ):
            break # Out == y thì thoát vòng lặp while, xong thuật toán, không bằng thì tiếp tục cập nhật
        s = s + 1
    print(s)
    return bias, s
##==============================================================================
def displayAxis(ax, fig, s):
    x0_1 =  np.min(x) # điểm x1
    x0_2 =  np.max(x)# điểm x2

    x1_1 = (-weights[0] * x0_1 - bias) / weights[1]
    # xác định tọa độ y ứng với điểm ngoài cùng bên trái trên đường thẳng
    x1_2 = (-weights[0] * x0_2 - bias) / weights[1]
    # xác định tọa độ y ứng với điểm ngoài cùng phải trên đường thẳng
    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")
    # vẽ đường thẳng từ 2 cặp (x, y) tìm được

    ax.set_title("Perceptron")
    ax.set_ylabel('y')
    label = 'x\nSố lần lặp để thuật toán hội tụ %d' %(s)
    ax.set_xlabel(label)
    print(weights,bias)
    plt.show()
##====      main      ============
lr = 0.01           # đối với ví dụ này của mình, 0.01 là điểm tốt nhất
ax, fig ,Y_train=displayDot(x)
bias,s=Perceptron(lr ,weights, bias)
displayAxis(ax, fig, s)
input()