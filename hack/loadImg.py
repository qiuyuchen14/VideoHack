import torch
import numpy as np
import torch.utils.data as tor
import time
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import glob

size = 32
user = 'zoey'
t0 = time.time()

data_tensor = np.load('/home/zoey/data/Youtube/keypoints.npy')
print(data_tensor.shape)
#print(time.time() - t0)
target_tensor = np.load('/home/zoey/data/Youtube/labels.npy')
n = data_tensor.shape[0]
#data1_tensor = np.load('/home/zoey/data/Youtube/WholeImage/matrix.npy') # read original RGB image
data2_tensor = np.load('/home/zoey/data/Youtube/ReLMovement.npy') # read keypoints relationships
#print(data_tensor.shape)

idx = np.arange(len(data_tensor))
np.random.shuffle(idx)
data_tensor = data_tensor[idx]
target_tensor = target_tensor[idx]

#data1_tensor = data1_tensor[idx]
data2_tensor = data2_tensor[idx]

data_tensor = data_tensor.reshape(1600, -1)
#data1_tensor = data1_tensor.reshape(1600, -1)
data2_tensor = data2_tensor.reshape(1600, -1)


#Data = np.hstack([data_tensor, data1_tensor, data2_tensor])
Data = data_tensor
val = Data[int(n*0.8):]
train = Data[:int(n*0.8)]
print(data_tensor.shape)
m = target_tensor.shape[0]
#need to be randomized before spliting
target_val = target_tensor[int(m*0.8):]
target_train = target_tensor[:int(m*0.8)]


for i in range(0, len(data_tensor), 11):
    if len(train) > 0.8*len(data_tensor):
        break
    train


Data_tensor = torch.from_numpy(train)
Target_tensor = torch.from_numpy(target_train)
DS = tor.TensorDataset(Data_tensor, Target_tensor)

Data_Validation = torch.from_numpy(val)
Target_Validation = torch.from_numpy(target_val)
VS = tor.TensorDataset(Data_Validation, Target_Validation)

dl = tor.DataLoader(DS, 512)

vl = tor.DataLoader(VS, 512)

class ForwardModel(torch.nn.Module):
    # __init__(self):
    #    super(ForwardModel, self).__init__()
    #    self.w1 = torch.nn.Linear(18*2*1*536, 512)
    #    self.w3 = torch.nn.Linear(512, 512)
    #    self.w2 = torch.nn.Linear(512, 11)
    #    self.dropout = torch.nn.Dropout(0.6)
    #    self.loss = torch.nn.CrossEntropyLoss()
    #    self.input_drop = torch.nn.Dropout(0.6)

    #def forward(self, videos):
    #    videos = videos.view(videos.size(0), -1)
    #    videos = self.input_drop(videos)
    #    x = self.w1(videos)
    #    x = F.relu(x)
    #    x = self.dropout(x)
    #    x = self.w3(x)
    #    x = F.relu(x)
    #    x = self.dropout(x)
    #    predictions = self.w2(x)
    #    return predictions

    def __init__(self):
        super(ForwardModel, self).__init__()

        self.inp = nn.Linear(18*2*1*536, 512)
        self.rnn = nn.LSTM(512, 512, 2, dropout=0.05)
        self.out = nn.Linear(512, 11)

    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden

    def forward(self, inputs, hidden=None, force=True, steps=0):
        if force or steps == 0: steps = len(inputs)
        outputs = Variable(torch.zeros(steps, 1, 1))
        for i in range(steps):
            if force or i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden



model = ForwardModel().cuda()
opt = Adam(model.parameters(), lr=0.001)

best_acc = 0.0
for epoch in range(1000):
    accuracy = []
    model.train(True)
    print('')
    print('Epoch: {0}'.format(epoch))
    for batch, label in dl:
        batch = Variable(batch).cuda()
        label = Variable(label).cuda()
        opt.zero_grad()
        pred = model.forward(batch)
        loss = model.loss(pred, label)
        loss.backward()
        opt.step()

        maxvalue, argmax = torch.topk(pred, 1)
        correct = torch.sum(argmax.data == label.view(-1, 1).data)
        accuracy.append(correct/batch.size(0))
    print('train', np.mean(accuracy))


    accuracy = []
    model.train(False)
    for batch, label_val in vl:
        batch = Variable(batch).cuda()
        label_val = Variable(label_val).cuda()
        pred_val = model.forward(batch)
        loss = model.loss(pred_val, label_val)

        maxvalue, argmax = torch.topk(pred_val, 1)
        correct = torch.sum(argmax.data == label_val.view(-1, 1).data)
        accuracy.append(correct/batch.size(0))
    acc = np.mean(accuracy)
    if acc > best_acc:
        print('New best model found!')
        best_acc = acc
        #torch.save(model, open('/home/zoey/data/model/model.pkl', 'wb'))
    print('validation', np.mean(accuracy))

# load model
# make predictions on test set



