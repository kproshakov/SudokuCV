
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

BATCH_SIZE = 50
EPOCHS = 10

path = "data/digits"
count = 0
images = []
classNo = []
myList = os.listdir(path)
for x in range (0,10):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(32,32))
        curImg = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        curImg = cv2.equalizeHist(curImg)
        curImg = curImg/255
        images.append(curImg)
        classNo.append(x)

print("Data has been imported")

images = np.array(images)
classNo = np.array(classNo)


X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size=0.2)


X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1],X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1],X_test.shape[2])



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 60, kernel_size=5)
        self.conv2 = nn.Conv2d(60, 60, kernel_size=5)
        self.conv3 = nn.Conv2d(60, 30, kernel_size=3)
        self.conv4 = nn.Conv2d(30, 30, kernel_size=3)
        self.lin1 = nn.Linear(4*4*30, 500)
        self.lin2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.5, training = self.training)
        x = x.view(-1, 4*4*30)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)


torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)


train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)


train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = True)


def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()


            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()

            if batch_idx % 50 == 0:
                print("Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%".format(epoch, batch_idx*len(X_batch),                 len(train_loader.dataset), 100*batch_idx/len(train_loader), loss.data, float(correct*100)/ float(BATCH_SIZE* (batch_idx + 1))))


model = Model()


fit(model, train_loader)


def evaluate(model):
    correct = 0
    for test_img, test_lbl in test_loader:
        test_img = Variable(test_img).float()
        
        output = model(test_img)
        predicted = torch.max(output, 1)[1]
        correct += (predicted == test_lbl).sum()
    
    print("Test accuracy:{:.3f}%".format(float(100 * correct)/(len(test_loader) * BATCH_SIZE)))

evaluate(model)

torch.save(model.state_dict(), "./model.pt")



