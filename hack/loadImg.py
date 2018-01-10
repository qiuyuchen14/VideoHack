import torch
import numpy as np
import torch.utils.data as tor
import time
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import glob


user = 'zoey'
t0 = time.time()
for i, path in enumerate(glob.iglob('/home/{0}/ucf_sports_actions/ucfaction/**/**/*.avi'.format(user), recursive=True)):
    print(i,path)



data_tensor = np.load('/home/zoey/data/matrix.npy')
print(time.time() - t0)
target_tensor=np.load('/home/zoey/data/labels.npy')
n = data_tensor.shape[0]
print("video:", len(data_tensor))

#val = data_tensor[int(n*0.8):]
#train = data_tensor[:int(n*0.8)]
#print(len(val))

#m = target_tensor.shape[0]
#target_val = target_tensor[int(m*0.8):]
#target_train = target_tensor[:int(m*0.8)]
#print(len(target_val))

train = []
val = []
target_val = []
target_train = []

for i in range(0, len(data_tensor), 13):
    if len(train) > 0.8*len(data_tensor):
        break
    train




Data_tensor = torch.from_numpy(train)
Target_tensor = torch.from_numpy(target_train)
DS = tor.TensorDataset(Data_tensor, Target_tensor)

Data_Validation = torch.from_numpy(val)
Target_Validation = torch.from_numpy(target_val)
VS = tor.TensorDataset(Data_Validation,Target_Validation)

dl = tor.DataLoader(DS,128)

vl = tor.DataLoader(VS,128)

class ForwardModel(torch.nn.Module):
    def __init__(self):
        super(ForwardModel, self).__init__()
        self.w1 = torch.nn.Linear(18*2*1*144, 128)
        self.w2 = torch.nn.Linear(128, 13)
        self.dropout = torch.nn.Dropout(0.0)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, videos):
        videos = videos.view(videos.size(0), -1)
        x = self.w1(videos)
        x = F.relu(x)
        x=self.dropout(x)
        predictions = self.w2(x)

        return predictions

model = ForwardModel()
opt = Adam(model.parameters(), lr=0.001)

best_acc = 0.0
for epoch in range(40):
    accuracy = []
    model.train(True)
    print('')
    print('Epoch: {0}'.format(epoch))
    for batch, label in dl:
        batch = Variable(batch)
        label = Variable(label)
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
        batch = Variable(batch)
        label_val = Variable(label_val)
        pred_val = model.forward(batch)
        loss = model.loss(pred_val, label_val)

        maxvalue, argmax = torch.topk(pred_val, 1)
        correct = torch.sum(argmax.data == label_val.view(-1, 1).data)
        accuracy.append(correct/batch.size(0))
    acc = np.mean(accuracy)
    if acc > best_acc:
        print('New best model found!')
        best_acc = acc
        #torch.save(model, open('/home/zoey/data/model/newmodel.pkl', 'wb'))
    print('validation', np.mean(accuracy))

# load model
# make predictions on test set



