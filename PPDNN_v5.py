# import os
import torch
from torch import nn
# import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# from PIL import Image
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
import base64
import numpy as np
import time

# hyperparameter
BATCH_SIZE = 50
EPOCHS = 25
TRAIN_DATA_SIZE = 5000 # MAX 60000
TEST_DATA_SIZE = 10000 # MAX 10000
TEST_RATE = TRAIN_DATA_SIZE / BATCH_SIZE / 6
LR = 0.001
DOWNLOAD_MNIST = False
HIDDEN = 128

# RSA parameter
# RSA_PK = 19 # e
# N = 437 # p * q
# --------------
torch.set_printoptions(
    precision=4,
    threshold=1000,
    edgeitems=3,
    linewidth=150,
    profile=None,
    sci_mode=False,
)
# --------------

class SSDNN(nn.Module):
        def __init__(self):
            super(SSDNN, self).__init__()
            # self.flatten = nn.Flatten()
            self.hidden1 = nn.Linear(28*28, HIDDEN) # layer 1
            self.sigm = nn.Sigmoid()
            self.out = nn.Linear(HIDDEN, 10) # layer 2
            # self.soft = nn.Softmax()

        def forward(self, x):
            # x.size() = (n, 2, 28, 28)
            # x = torch.flatten(x, start_dim = 2)
            x = self.hidden1(x)
            x = self.sigm(x)
            x = self.out(x)
            # x = self.soft(x, dim=1)
            return x

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.transform = transform
        self.data = data
        self.targets = targets

    def __getitem__(self, index):  # 给下标，然后返回该下标对应的item
        img,target = self.data[index],self.targets[index]
        return img,target

    def __len__(self):  # 返回数据的长度
        return len(self.data)

# class MyDataset_enc(Dataset):
#     def __init__(self, Mat_A, data, targets, transform=None):
#         self.transform = transform
#         # self.Mat_A = Mat_A
#         self.data = {'Mat_A': Mat_A, 'enc_x': data}
#         self.targets = targets

#     def __getitem__(self, index):  # 给下标，然后返回该下标对应的item
#         img = {self.data['Mat_A'][index], self.data['enc_x'][index]}
#         target = self.targets[index]
#         return img,target

#     def __len__(self):  # 返回数据的长度
#         return len(self.data)

    # def _check_exists(self):
    #     return (os.path.exists(os.path.join(self.root, 'MNIST', 'processed', self.train_file)) and os.path.exists(
    #         os.path.join(self.root, 'MNIST', 'processed', self.test_file)))

def original(train_x, train_y, test_x, test_y):
    start = time.time()

    model_o = SSDNN()

    train_x = torch.flatten(train_x, start_dim=1)
    test_x = torch.flatten(test_x, start_dim=1)
    
    train_dataset_o = MyDataset(train_x, train_y)

    train_dataloader_o = DataLoader(train_dataset_o, batch_size=BATCH_SIZE)#, shuffle=True)

    loss_func_o = nn.CrossEntropyLoss()
    
    size = len(train_dataloader_o.dataset)
    for t in range(EPOCHS):
        batch_start = time.time()
        print(f"-------------------------------\noriginal Epoch {t+1}")
        for batch, (X, y) in enumerate(train_dataloader_o):
            # print('X:\n', X)

            # train...
            output = model_o(X)
            loss = loss_func_o(output, Variable(y))
            loss.backward()
            
            h_o = torch.zeros(X.size(dim=0), HIDDEN, 1)
            v_o = torch.zeros(X.size(dim=0), HIDDEN, 1)
            o_o = torch.zeros(X.size(dim=0), 10, 1)
            
            d1_o = torch.zeros(X.size(dim=0), 10, HIDDEN)
            d2_o = torch.zeros(X.size(dim=0), HIDDEN, 28*28)
            d3_o = torch.zeros(X.size(dim=0), 1, HIDDEN)
            
            for i in range(X.size(dim=0)):
                h_o[i] = torch.sigmoid(model_o.hidden1.weight @ X[i]).unsqueeze(1)
                v_o[i] = -1 * h_o[i] * (1 - h_o[i])

                o_o[i] = model_o.out.weight @ h_o[i]
                # print(o_o[i])

                d1_o[i] = (o_o[i] - y[i].unsqueeze(1)).to(torch.float32) @ h_o[i].t()
                d2_o[i] = (v_o[i] @ X[i].unsqueeze(0))
                # print('V * X',v_o[i].size(), X[i].size(), d2_o[i])
                # print(d2_o.size())

                d3_o[i] = (y[i].unsqueeze(1) - o_o[i]).to(torch.float32).t() @ model_o.out.weight
                # print((y[i].unsqueeze(1) - o_o[i]).to(torch.float32).t(), model_o.out.weight)
                # print('D3:', d3_o[i], y[i].unsqueeze(1), o_o[i], model_o.out.weight)
                # print('stop point')

            # model update...
            D_h = torch.zeros(X.size(dim=0), HIDDEN, 28*28)
            D_out = d1_o

            G_h = torch.zeros(HIDDEN, 28*28)
            G_out = torch.zeros(10, HIDDEN)

            for i in range(X.size(dim=0)):
                # print(d2_o[i].size(), d3_o[i].size())
                D_h[i] = (d2_o[i].t() * d3_o[i].squeeze(0)).t()
                # print('D',d2_o[i][0], d3_o[i][0], D_h[i][0])

                G_h = G_h + (D_h[i] / X.size(dim=0))
                G_out = G_out + (D_out[i] / X.size(dim=1))

            model_o.hidden1.weight = nn.Parameter(model_o.hidden1.weight - (LR * G_h))
            model_o.out.weight = nn.Parameter(model_o.out.weight - (LR * G_out))
            # print('weight',model_o.hidden1.weight, model_o.out.weight)

            # test...
            if batch % TEST_RATE == 0:

                test_output = model_o(test_x)
                test_pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((test_pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Batch: ', batch, '| loss: %.4f' % loss.data.numpy(), '| accuracy: %.2f' % accuracy, f'[{batch*X.size(dim=0):>5d}/{size:>5d}]')
        
        batch_end = time.time()
        print("執行時間：%f 秒" % (batch_end - batch_start))

        
        # test_output = model_o(test_x)
        # pred_y = torch.max(test_output, 1)[1].data.numpy()
        # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

        # print(f"Epoch {t+1}:", "test accuracy: %.2f" % accuracy)

    end = time.time()
    print("original Done! 執行時間：%f 秒" % (end - start))

# def RSA(num, key):
#     return (num ** key) % N

class Dataowner():
    def Dataprocessing(self): # O
        print('in dataowner O!')

        train_data = torchvision.datasets.MNIST(
            root='./pytorch/datasets',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            # download=DOWNLOAD_MNIST,
        )
        train_data.data = train_data.data.float()

        test_data = torchvision.datasets.MNIST(
            root='./pytorch/datasets',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            # download=True,
        )
        train_x = train_data.data.float()[:TRAIN_DATA_SIZE]
        train_y = train_data.targets[:TRAIN_DATA_SIZE]

        test_x = test_data.data.float()[:TEST_DATA_SIZE]
        test_y = test_data.targets[:TEST_DATA_SIZE]

        plt.imshow(train_x[0].numpy(), cmap = 'gray')
        plt.show()

        train_x = torch.flatten(train_x, start_dim=1)
        test_x = torch.flatten(test_x, start_dim=1)

        train_enc_x = torch.zeros(TRAIN_DATA_SIZE, 2, 28*28, 28*28)
        # Mat_A = torch.rand(TRAIN_DATA_SIZE, 28*28, 28*28) # 記憶體不夠，所以只生成1個
        # A_inv = torch.linalg.inv(Mat_A)

        for i in range(TRAIN_DATA_SIZE):
            train_enc_x[i][0] = torch.rand(28*28, 28*28)
            A_inv = torch.linalg.inv(train_enc_x[i][0])
            # train_enc_x[i][1] = A_inv @ train_x[i] 
            # for j in range(28*28):
            train_enc_x[i][1][0] = A_inv @ train_x[i] # 需要RSA加密
            # print(train_enc_x[i][1][0])
            for j in range(28*28):
                train_enc_x[i][1][0][j] = torch.round(train_enc_x[i][1][0][j], decimals=4)
            #     train_enc_x[i][1][0][j] = RSA(train_enc_x[i][1][0][j], RSA_PK) # RSA Enc
            # print(train_enc_x[i][1][0])

        # print('cat:\n', torch.cat((Mat_A, train_enc_x), 1).size())

        rec = train_enc_x[0][0] @ train_enc_x[0][1][0]
        rec = rec.round()

        plt.imshow(train_enc_x[0][1][0].reshape([28, 28]).numpy(), cmap = 'gray')
        plt.show()
        plt.imshow(rec.reshape([28, 28]).numpy(), cmap = 'gray')
        plt.show()

        for i in range(28 *28):
                if train_x[0][i] != rec[i]:
                    print('not same!')

        # AES Enc
        aes_x = torch.rand(1)
        for i in range(TRAIN_DATA_SIZE):
            train_enc_x[i][1][0] = train_enc_x[i][1][0] * aes_x
        aes_x = aes_x.numpy()
        aes_x = base64.b64encode(aes_x)

        # RSA enc AES
        pk = RSA.importKey(open("./pytorch/keypairs/public.pem").read())
        cipher = Cipher_pkcs1_v1_5.new(pk)     #建立用於執行pkcs1_v1_5加密或解密的密碼
        cipher_text_x = base64.b64encode(cipher.encrypt(aes_x))

        train_one_hot_y = torch.tensor(np.eye(10)[train_y])
        test_one_hot_y = torch.tensor(np.eye(10)[test_y])

        print('----------one hot----------')
        print('train:', train_one_hot_y) 
        print(type(train_one_hot_y))
        print(train_one_hot_y.size())
        print('test:', test_one_hot_y)
        print(type(test_one_hot_y))
        print(test_one_hot_y.size())
  

        #                                                  | original需要 |
        return train_enc_x, cipher_text_x, train_one_hot_y, test_x, test_y, train_x


class Trainer():
    Mat_A = torch.rand(BATCH_SIZE, 28*28, 28*28)

    def __init__(self) -> None:
        self.model = SSDNN()

    def Train(self, z, y):
        # print('in trainer T!')

        r1 = torch.randn(size = (BATCH_SIZE, 10))
        r2 = torch.randn(size = (10, HIDDEN))
        Mat_B = torch.rand(HIDDEN, HIDDEN)
        B_inv = torch.linalg.inv(Mat_B)

        y_share1 = y + 1 * r1
        y_share2 = y + 2 * r1

        W_o_share1 = self.model.out.weight + 1 * r2
        W_o_share2 = self.model.out.weight + 2 * r2
        
        t1_1 = torch.zeros(BATCH_SIZE, 28*28) # v3

        t1_2 = torch.zeros(BATCH_SIZE, HIDDEN, 28*28) # v3

        # print('z size', z.size(), t1_1.size())

        # AES Enc
        aes_w = torch.rand(1)
        for i in range(BATCH_SIZE):
            self.Mat_A[i] = z[i][0]
            t1_1[i] = z[i][1][0]
            t1_2[i] = self.model.hidden1.weight @ self.Mat_A[i]
            t1_2[i] = t1_2[i] * aes_w
        aes_w = aes_w.numpy()
        aes_w = base64.b64encode(aes_w)

        # RSA enc AES
        pk = RSA.importKey(open("./pytorch/keypairs/public.pem").read())
        cipher = Cipher_pkcs1_v1_5.new(pk)     #建立用於執行pkcs1_v1_5加密或解密的密碼
        cipher_text_w = base64.b64encode(cipher.encrypt(aes_w))

        #           需要RSA
        return t1_1, t1_2, cipher_text_w, y_share1, y_share2, W_o_share1, W_o_share2, self.model.out.weight @ B_inv, Mat_B

    def ModelUpdate(self, Mat_B, d1_share1, d1_share2, d2, d3_share1, d3_share2):
        # D_out = torch.zeros(BATCH_SIZE, 10, HIDDEN)
        D_h = torch.zeros(BATCH_SIZE, HIDDEN, 28*28)
        D_out = d1_share1 * 2 - d1_share2

        G_h = torch.zeros(HIDDEN, 28*28)
        G_out = torch.zeros(10, HIDDEN)


        for i in range(BATCH_SIZE):
            D_h[i] = ((d2[i] @ self.Mat_A[i].t()).t() * ((d3_share1[i] * 2 - d3_share2[i]) @ Mat_B).squeeze(0)).t()

            G_h = G_h + (D_h[i] / BATCH_SIZE)
            G_out = G_out + (D_out[i] / BATCH_SIZE)
            
            # self.model.hidden1.weight = nn.Parameter(self.model.hidden1.weight - LR * D_h[i] / BATCH_SIZE)
            # self.model.out.weight = nn.Parameter(self.model.out.weight - LR * D_out[i] / BATCH_SIZE)

        self.model.hidden1.weight = nn.Parameter(self.model.hidden1.weight - (LR * G_h))
        self.model.out.weight = nn.Parameter(self.model.out.weight - (LR * G_out))


class CloudServiceProvider1(): # have RSA sk
    def __init__(self):
        self.R = None
        # self.sk = 43 # d

    def RSAkeyGen():
        key = RSA.generate(1024)
        privateKey = key.export_key()
        publicKey = key.publickey().export_key()
        
        # RSA 私鑰
        privateKey = key.export_key()
        with open("./pytorch/keypairs/private.pem", "wb") as f:
            f.write(privateKey)

        # RSA 公鑰
        publicKey = key.publickey().export_key()
        with open("./pytorch/keypairs/public.pem", "wb") as f:
            f.write(publicKey)

    
    def train(self, cipher_text_x, cipher_text_w, t1_1, t1_2, t1_3_share1, t2_share1, t3):
        # print('in trainer CSP1!')
        # RSA Dec
        rsakey = RSA.importKey(open("./pytorch/keypairs/private.pem").read())
        cipher = Cipher_pkcs1_v1_5.new(rsakey)      #建立用於執行pkcs1_v1_5加密或解密的密碼
        aes_x = cipher.decrypt(base64.b64decode(cipher_text_x), "x解密失敗")
        aes_w = cipher.decrypt(base64.b64decode(cipher_text_w), "w解密失敗")

        aes_x = base64.decodebytes(aes_x)
        aes_x = np.frombuffer(aes_x, dtype=np.float32)
        aes_x = torch.tensor(aes_x)
        
        aes_w = base64.decodebytes(aes_w)
        aes_w = np.frombuffer(aes_w, dtype=np.float32)
        aes_w = torch.tensor(aes_w)

        h = torch.zeros(BATCH_SIZE, HIDDEN, 1)
        v = torch.zeros(BATCH_SIZE, HIDDEN, 1)
        o_share1 = torch.zeros(BATCH_SIZE, 10, 1)
        d1_share1 = torch.zeros(BATCH_SIZE, 10, HIDDEN)
        d2 = torch.zeros(BATCH_SIZE, HIDDEN, 28*28)
        d3_share1 = torch.zeros(BATCH_SIZE, 1, HIDDEN)

        # print(o_share1.size())
        # print(t1_3_share1.size())
        # print(t1_2.size(), t1_1.size())
        for i in range(BATCH_SIZE):
            t1_1[i] = t1_1[i] / aes_x # AES Dec
            t1_2[i] = t1_2[i] / aes_w # AES Dec
            h[i] = torch.sigmoid(t1_2[i] @ t1_1[i]).unsqueeze(1)
            v[i] = -1 * h[i] * (1 - h[i])
            o_share1[i] = t2_share1 @ h[i]

            # for j in range(10):
            #     for k in range(HIDDEN):
            #         d_share1[i][j][k] = (o_share1[i][j] - t1_3_share1[i][j]) * h[i][k]
            
            d1_share1[i] = (o_share1[i] - t1_3_share1[i].unsqueeze(1)).to(torch.float32) @ h[i].t()
            d2[i] = (v[i] @ t1_1[i].unsqueeze(0)) / self.R

            d3_share1[i] = (t1_3_share1[i].unsqueeze(1) - o_share1[i]).to(torch.float32).t() @ t3 * self.R

        return h, d1_share1, d2, d3_share1, o_share1


class CloudServiceProvider2():
    def __init__(self):
        self.R = None

    def train(self, h, t1_3_share2, t2_share2, t3):
        # print('in trainer CSP2!')

        o_share2 = torch.zeros(BATCH_SIZE, 10, 1)
        d1_share2 = torch.zeros(BATCH_SIZE, 10, HIDDEN)
        d3_share2 = torch.zeros(BATCH_SIZE, 1, HIDDEN)

        for i in range(BATCH_SIZE):
            o_share2[i] = t2_share2 @ h[i]
            d1_share2[i] = (o_share2[i] - t1_3_share2[i].unsqueeze(1)).to(torch.float32) @ h[i].t()
            d3_share2[i] = (t1_3_share2[i].unsqueeze(1) - o_share2[i]).to(torch.float32).t() @ t3 * self.R

        return d1_share2, d3_share2, o_share2


def keyexchange(CSP1, CSP2, seed):
    np.random.seed(seed)
    rd = np.random.random()
    CSP1.R = rd
    CSP2.R = rd
    # print(CSP1.R, CSP2.R)


def main():

    T = Trainer()
    O = Dataowner()
    CSP1 = CloudServiceProvider1()
    CSP2 = CloudServiceProvider2()

    # RSA_PK = CSP1.keyGen()
    # print('RSA:\n----------pk----------\n', RSA_PK, '\n----------sk----------\n', CSP1.sk)

    train_x, cipher_text_x, train_y, test_x, test_y, train_x_o = O.Dataprocessing()

    original(train_x_o, train_y, test_x, test_y)
    start = time.time()

    train_dataset = MyDataset(train_x, train_y)
    # test_dataset = MyDataset(test_x, test_y)

    # print('TX:',train_x[0])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)#, shuffle=True)
    print(train_dataloader.dataset)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # model = SSDNN()
    # optimizer = optim.Adam(T.model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    # print(T.model)
    # print(BATCH_SIZE)

    # print('h1 weight:',T.model.hidden1.weight.size())
    # print(T.model.hidden1.weight)

    # print('out weight:',T.model.out.weight.size())
    # print(T.model.out.weight)

    size = len(train_dataloader.dataset)
    for t in range(EPOCHS):
        batch_start = time.time()
        print(f"-------------------------------\nEpoch {t+1}")
        for batch, (X, y) in enumerate(train_dataloader):
            # train...
            X_re = torch.zeros(BATCH_SIZE, 28*28)
            for i in range(BATCH_SIZE):
                X_re[i] = X[i][0] @ X[i][1][0]
                X_re[i] = X_re[i].round()
            output = T.model(X_re)
            loss = loss_func(output, Variable(y))
            loss.backward()

            # enc_x, W_h*A,           y_1, y_2,             W_o_1, W_o_2    , W_o*B_inv, B
            t1_1, t1_2, cipher_text_w, t1_3_share1, t1_3_share2, t2_share1, t2_share2, t3 , Mat_B = T.Train(X, y)
            
            keyexchange(CSP1, CSP2, batch)
            h, d1_share1, d2, d3_share1, o_share1 = CSP1.train(cipher_text_x, cipher_text_w, t1_1, t1_2, t1_3_share1, t2_share1, t3)
            d1_share2, d3_share2, o_share2 = CSP2.train(h, t1_3_share2, t2_share2, t3)

            T.ModelUpdate(Mat_B, d1_share1, d1_share2, d2, d3_share1, d3_share2)

            # test...
            if batch % TEST_RATE == 0:
                
                test_output = T.model(test_x)
                test_pred_y = torch.max(test_output, 1)[1].data.numpy()

                accuracy = float((test_pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Batch: ', batch, '| loss: %.4f' % loss.data.numpy(), '| accuracy: %.2f' % accuracy, f'[{batch*BATCH_SIZE:>5d}/{size:>5d}]')
                
        batch_end = time.time()
        print("執行時間：%f 秒" % (batch_end - batch_start))

            
        # test_output = T.model(test_x)
        # pred_y = torch.max(test_output, 1)[1].data.numpy()
        # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

        # print(f"Epoch {t+1} test:", "accuracy:, %.2f" % accuracy)
    
    end = time.time()
    print("Done! 執行時間：%f 秒" % (end - start))
    

if __name__ == '__main__':
    main()