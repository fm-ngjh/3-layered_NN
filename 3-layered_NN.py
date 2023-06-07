import numpy as np
import matplotlib.pyplot as plt
import random
PI = np.pi

# ハイパーパラメータ
n = 6  # 中間層のノード数
nbTeacherData = 100   # 教師データ数
epochs = 1000    #学習回数
alpha = 0.3 # 慣性項更新時の係数
beta = 1.0  # シグモイド関数の傾き
eta = 0.5   # 学習率
eta_dec = 0.9 # 学習が全体の1/10進むごとにかけて学習率を減衰させる割合

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-beta * x))

class NeuralNetwork:
    def __init__(self):
        # 重みパラメータの初期化
        self.weight_w = np.zeros((n, 3))
        randrange = 1
        for i in range(n):
            ww = [random.uniform(-randrange, randrange), 
                  random.uniform(-randrange, randrange),
                  random.uniform(-randrange, randrange)]
            self.weight_w[i] = ww
        
        self.weight_v = np.zeros((1, n))
        for i in range(n):
            self.weight_v[0][i] = random.uniform(-randrange, randrange)
        
        # 慣性項の初期化
        self.m_w = np.zeros((n, 3))
        self.m_v = np.zeros((1, n))

        # 教師データの作成
        self.x_teacher = np.zeros((nbTeacherData, 3))
        z_teacher = np.zeros(nbTeacherData)
        self.teacherSignal = np.zeros((nbTeacherData, 1))

        for i in range(nbTeacherData):
            # 入力ベクトル
            self.x_teacher[i][0] = 1            
            self.x_teacher[i][1] = random.uniform(0, 1)
            self.x_teacher[i][2] = random.uniform(0, 1)

            # x1 x2に対するz
            z_teacher[i] = (1 + np.sin(4 * PI * self.x_teacher[i][1])) * self.x_teacher[i][2] / 2
            
            # 教師信号化
            self.teacherSignal[i] = [sigmoid(z_teacher[i])]
        
    def forward_propagation(self, x):
        # 入力層から中間層
        self.hidden_layer_input = np.dot(self.weight_w, x.reshape(-1, 1))
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
                
        # 中間層から出力層
        self.output_layer_input = np.dot(self.weight_v, self.hidden_layer_output)
        self.output_layer_output = sigmoid(self.output_layer_input)
        
    def back_propagation(self, x, t):
        # 出力層から中間層
        self.output_error = t - self.output_layer_output
        output_delta = self.output_error * beta * self.output_layer_output * (1 - self.output_layer_output) 
        self.m_v = alpha * self.m_v + eta * np.dot(output_delta.reshape(-1, 1), self.hidden_layer_output.reshape(1, -1))
        self.weight_v += self.m_v
        
        # 中間層から入力層
        hidden_error = np.dot(output_delta, self.weight_v)
        hidden_delta = np.multiply(hidden_error, np.multiply(self.hidden_layer_output, 1- self.hidden_layer_output).T)
        self.m_w = alpha * self.m_w + eta * np.dot(hidden_delta.reshape(-1, 1), x.reshape(1, -1))
        self.weight_w += self.m_w
         
    def train(self):
        loss = []
        for epoch in range(epochs):
            for i in range(nbTeacherData):
                self.forward_propagation(self.x_teacher[i])
                self.back_propagation(self.x_teacher[i], self.teacherSignal[i])
                loss.append(pow(self.output_error[0], 2) / 2)
                
            print("epoch ", epoch+1)
            if((epoch+1)%(epochs/10) == 0):
                global eta
                eta = eta * eta_dec
            
        plt.plot(range(len(loss)), loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Lerning curve')
    
    def plot_origin(self):
        x = np.zeros(5000)
        y = np.zeros(5000)
        z = np.zeros(5000)
        for i in range(5000):
            x[i] = random.uniform(0, 1)
            y[i] = random.uniform(0, 1)
            z[i] = (1 + np.sin(4 * PI * x[i])) * y[i] / 2
            
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        mappable = ax.scatter(x, y, z, c=z, cmap='cool')
        plt.colorbar(mappable, ax=ax)    
    
    def plot(self):
        nbdata = 5000
        x1 = np.zeros(nbdata)
        x2 = np.zeros(nbdata)
        x = np.zeros((nbdata, 3))
        
        for i in range(nbdata):
            x1[i] = random.uniform(0, 1)
            x2[i] = random.uniform(0, 1)
            x[i] = [1, x1[i], x2[i]]
            
        z = []
        for i in range(nbdata):
            self.forward_propagation(x[i])
            z.append(self.output_layer_input[0])
            
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        mappable = ax.scatter(x1, x2, z, c=z, cmap='cool')
        plt.colorbar(mappable, ax=ax)

nn = NeuralNetwork()

# 学習
loss = nn.train()

# 理論値グラフの作成
nn.plot_origin()

# 学習データからグラフを作成
nn.plot()

# グラフを描画
plt.show()    
     