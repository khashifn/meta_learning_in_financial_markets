import numpy as np
import torch
from torch import nn, autograd as ag
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm.notebook import tqdm
import pandas as pd
import os
import pickle
import csv
from sklearn.preprocessing import StandardScaler
from os.path import join
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae

seed = 0
plot = False
innerstepsize = 0.001 # stepsize in inner SGD
innerepochs = 200 # number of epochs of each inner SGD
outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
niterations = 150 # number of outer updates; each iteration we sample one task and update on it

seed = 0
rng = np.random.RandomState(seed)
torch.manual_seed(seed)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def totorch(x):
    return ag.Variable(torch.Tensor(x))

def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()

def train_on_batch(x, y):
    x = totorch(x)
    x = to_cuda(x)
    y = totorch(y)
    y = to_cuda(y)
    model.zero_grad()
    ypred = model(x)
    loss = criterion(ypred,y)
    #loss = (ypred - y).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    #optimizer.step()
    return loss.cpu().detach().numpy()

def val_cost(x, y):
    x = totorch(x)
    x = to_cuda(x)
    y = totorch(y)
    y = to_cuda(y)
    ypred = model(x)
    loss = criterion(ypred,y)
    return loss.cpu().detach().numpy()

def predict(x):
    x = totorch(x)
    x = to_cuda(x)
    return model(x).data.cpu().numpy()

# Choose a fixed task and minibatch for visualization
def measure_f1_val(x,y):
    y_hat = predict(x)
    batch_size = y.shape[0]
    predict_label = y_hat>0.5
    try:
        tp=np.sum(np.logical_and((y_hat>0.5),(y>0.5)))
        tn=np.sum(np.logical_and((y_hat<=0.5),(y<=0.5)))
        fp=np.sum(np.logical_and((y_hat>0.5),(y<=0.5)))
        fn=np.sum(np.logical_and((y_hat<=0.5),(y>0.5)))
    except:
        tp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()>0.5)))
        tn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()<=0.5)))
        fp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()<=0.5)))
        fn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()>0.5)))
    f1_p=tp/(tp+0.5*(fn+fp))
    f1_n=tn/(tn+0.5*(fn+fp))
    return (f1_p+f1_n)/2, f1_p, f1_n


def measure_f1(x,y):
    y_hat = predict(x)
    batch_size = y.shape[0]
    #predict_label = np.argmax(y_hat, axis=1)
    predict_label = y_hat>0.5
    try:
        tp=np.sum(np.logical_and((y_hat>0.5),(y>0.5)))
        tn=np.sum(np.logical_and((y_hat<=0.5),(y<=0.5)))
        fp=np.sum(np.logical_and((y_hat>0.5),(y<=0.5)))
        fn=np.sum(np.logical_and((y_hat<=0.5),(y>0.5)))
    except:
        tp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()>0.5)))
        tn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()<=0.5)))
        fp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()<=0.5)))
        fn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()>0.5)))
    f1_p=tp/(tp+0.5*(fn+fp))
    f1_n=tn/(tn+0.5*(fn+fp))
    return (f1_p+f1_n)/2

def init_fc():
    for layer in list_fc:
        try:
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.41)
            #print(f'layer {layer} initialized')
        except:
            pass
            #print("layer not detected for layer:", layer)
        try:
            torch.nn.init.uniform_(layer.bias, a=0.0, b=0.05)
            #torch.nn.init.normal_(layer.bias)
            #print(f'layer {layer} bias initialized')
        except:
            #print("Bias layer not detected for layer:", layer)
            pass
def freeze_cnn(freeze=True):
        try:
            if freeze:
                for layer in list_cnn:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for layer in list_cnn:
                    for param in layer.parameters():
                        param.requires_grad = True
        except:
            if freeze:
                for layer in [self.model.conv1, self.model.conv2, self.model.conv3]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for layer in [self.model.conv1, self.model.conv2, self.model.conv3]:
                    for param in layer.parameters():
                        param.requires_grad = True
        return None

class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0
        
    def log_kv(self, key, value):
        # logs the (key, value) pair
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path,save_name=None):
        if save_name==None:
            save_name='log'
        pickle.dump(self.log, open(save_path+f'/{save_name}.pickle', 'wb'))
        with open(save_path+f'/{save_name}.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
class cts:
    def __init__(self, num_tasks, innerstepsize, outerstepsize, optim='adam', drop=False, droprate=0.5, loss_func='bce'):
        self.innerstepsize=innerstepsize
        self.outerstepsize=outerstepsize
        self.num_tasks=num_tasks
        
        self.slow_learner=[]
        self.fast_learner=[]
        
        self.init_model(True, outerstepsize, optim=optim, drop=drop, droprate=0.5, loss_func=loss_func)
        for i in range(num_tasks):
            self.init_model(False,innerstepsize,optim=optim,drop=drop,droprate=0.5, loss_func=loss_func)
            self.set_params(self.get_params(),i)
        ###########some random input for loss to initialize gradients of slow learner
        loss = self.slow_learner[0][1](self.slow_learner[0][0](to_cuda(totorch(np.ones([1,1,60,82])))),
                                      to_cuda(totorch(np.array([0]).reshape([1,1]))))
        loss.backward()
        self.slow_learner[0][0].zero_grad()
        self.slow_learner[0][2].zero_grad()
        
        
    def init_model(self, slow=False, stepsize=None, optim='adam',drop=False,droprate=0.5, loss_func='bce'):
        model = nn.Sequential()
        num_filter=8
        num_features=82

        model.add_module('conv1', nn.Conv2d(1, num_filter, kernel_size=(1, num_features)))
        model.add_module('relu1', nn.ReLU())
        model.add_module('conv2', nn.Conv2d(num_filter, num_filter, kernel_size=(3, 1)))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=(2, 1)))
        model.add_module('conv3', nn.Conv2d(num_filter, num_filter, kernel_size=(3, 1)))
        model.add_module('relu3', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=(2, 1)))
        model.add_module('flatten', Flatten())
        if drop:
            model.add_module('drop1', nn.Dropout(droprate))
        #model.add_module('fc1', nn.Linear(104, 1))
        #model.add_module('sig1', nn.Sigmoid())

        model.add_module('fc1', nn.Linear(104, 50))
        model.add_module('sig1', nn.Sigmoid())
        #model.add_module('sig1', nn.Tanh())

        model.add_module('fc2', nn.Linear(50, 25))
        model.add_module('sig2', nn.Sigmoid())
        #model.add_module('sig2', nn.Tanh())

        model.add_module('fc3', nn.Linear(25, 1))
        model.add_module('sig3', nn.Sigmoid())

        list_cnn=[model.conv1, model.conv2, model.conv3]
        #list_fc=[model.fc1]
        list_fc=[model.fc1, model.fc2, model.fc3]
        for layer in list_cnn + list_fc:
            try:
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.41)
                #print(f'layer {layer} initialized')
            except:
                #print("layer not detected for layer:", layer)
                pass

            try:
                torch.nn.init.uniform_(layer.bias, a=0.0, b=0.05)
                #torch.nn.init.normal_(layer.bias)
                #print(f'layer {layer} bias initialized')
            except:
                #print("Bias layer not detected for layer:", layer)
                pass
        model.cuda()
        if loss_func=='bce':
            criterion = nn.BCELoss()
        elif loss_func=='mae':
            criterion = nn.L1Loss()
        if optim=='adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=stepsize)
        elif optim=='sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=stepsize, momentum=0.9)
        
        if slow:
            self.slow_learner.append([model, criterion, optimizer, list_fc, list_cnn])
        else:
            self.fast_learner.append([model, criterion, optimizer, list_fc, list_cnn])



    def init_fc(self,tid):
        for layer in self.fast_learner[tid][3]:
            try:
                torch.nn.init.xavier_uniform_(layer.weight, gain=1.41)
                #print(f'layer {layer} initialized')
            except:
                pass
                #print("layer not detected for layer:", layer)
            try:
                torch.nn.init.uniform_(layer.bias, a=0.0, b=0.05)
                #torch.nn.init.normal_(layer.bias)
                #print(f'layer {layer} bias initialized')
            except:
                #print("Bias layer not detected for layer:", layer)
                pass
    
    def learn_on_data(self,x,y,tid):
        x = totorch(x)
        x = to_cuda(x)
        y = totorch(y)
        y = to_cuda(y)
        self.fast_learner[tid][0].zero_grad()
        ypred = self.fast_learner[tid][0](x)
        loss = self.fast_learner[tid][1](ypred,y)
        #loss = (ypred - y).pow(2).mean()
        self.fast_learner[tid][2].zero_grad()
        loss.backward()
        self.fast_learner[tid][2].step()
        
        return loss.cpu().detach().numpy()

    def get_loss(self,x,y,tid,numpy=False):
        x = totorch(x)
        x = to_cuda(x)
        y = totorch(y)
        y = to_cuda(y)
        ypred = self.fast_learner[tid][0](x)
        loss = self.fast_learner[tid][1](ypred,y)
        if numpy:
            return loss.cpu().detach().numpy()
        return loss


    def freeze_cnn(self,tid,freeze=True):
        for layer in self.fast_learner[tid][4]:
            for param in layer.parameters():
                if freeze:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def get_params(self,slow=True,tid=0):
        if slow:
            return torch.cat([param.data.view(-1) for param in self.slow_learner[0][0].parameters()],0).clone()
        else:
            return torch.cat([param.data.view(-1) for param in self.fast_learner[tid][0].parameters()],0).clone()
        
    
    def set_params(self, param_vals, tid):
        offset = 0
        for param in self.fast_learner[tid][0].parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
    
    def set_all_params(self, param_vals,init_fc=False):
        for i in range(self.num_tasks):
            self.set_params(param_vals,i)
            if init_fc:
                self.init_fc(i)
    
    def get_meta_grad(self,x=None,y=None,method='FOMAML',flat=False):
        flat_grads=[]
        if method=='FOMAML':
            for i in range(self.num_tasks):
                tloss = self.get_loss(x[i,:], y[i,:], i)
                grad_ft = torch.autograd.grad(tloss, self.fast_learner[i][0].parameters(),retain_graph=True)
                tloss.backward()
                self.fast_learner[i][0].zero_grad()
                flat_grads.append(torch.cat([g.contiguous().view(-1) for g in grad_ft]))
            grad=torch.zeros_like(flat_grads[0])
            for flat_grad in flat_grads:
                grad+=flat_grad/self.num_tasks        
        elif method=='Reptile':
            slow_param=self.get_params()
            grad=torch.zeros_like(slow_param)
            
            for i in range(self.num_tasks):
                flat_grads.append(slow_param-self.get_params(slow=False,tid=i))
                grad+=(slow_param-self.get_params(slow=False,tid=i))/self.num_tasks
        if flat:
            return flat_grads
        return grad
    
    def set_all_grads(self,grads):
        self.set_grads(grads)
        for i in range(self.num_tasks):
            self.set_grads(grads,False,i)
    
    def step(self):
        for learner in self.slow_learner+self.fast_learner:
            learner[2].step()
        
    def set_eval(self):
        for learner in self.slow_learner+self.fast_learner:
            learner[0].eval()
        
    def set_train(self):
        for learner in self.slow_learner+self.fast_learner:
            learner[0].train()
            
    def set_grads(self, grads, slow=True,tid=None):
        offset=0
        if slow:   
            for param in self.slow_learner[0][0].parameters():
                param.grad.data.copy_(grads[offset:offset + param.grad.nelement()].view(param.grad.size()))
                offset += param.grad.nelement()
        else:
            for param in self.fast_learner[tid][0].parameters():
                param.grad.data.copy_(grads[offset:offset + param.grad.nelement()].view(param.grad.size()))
                offset += param.grad.nelement()
                
    def predict(self,x,tid):
        x = totorch(x)
        x = to_cuda(x)
        return self.fast_learner[tid][0](x).data.cpu().numpy()

    def measure_f1(self,x,y,tid):
        y_hat = self.predict(x,tid)
        #batch_size = y.shape[0]
        #predict_label = np.argmax(y_hat, axis=1)
        predict_label = y_hat>0.5
        try:
            tp=np.sum(np.logical_and((y_hat>0.5),(y>0.5)))
            tn=np.sum(np.logical_and((y_hat<=0.5),(y<=0.5)))
            fp=np.sum(np.logical_and((y_hat>0.5),(y<=0.5)))
            fn=np.sum(np.logical_and((y_hat<=0.5),(y>0.5)))
        except:
            tp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()>0.5)))
            tn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()<=0.5)))
            fp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()<=0.5)))
            fn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()>0.5)))
        #recall=tp/(tp+fn)
        #precision=tp/(tp+fp)
        #recall_n=tn/(tn+fp)
        #precision_n=tn/(tn+fn)
        #f1__p=2*recall*precision/(recall+precision)
        #f1__n=2*recall_n*precision_n/(recall_n+precision_n)
        f1_p=tp/(tp+0.5*(fn+fp))
        #print('f1_p',f1_p)
        f1_n=tn/(tn+0.5*(fn+fp))
        #print('f1_n',f1_n)
        return (f1_p+f1_n)/2
    def measure_f1_val(self,x,y,tid):
        y_hat = self.predict(x,tid)
        #batch_size = y.shape[0]
        #predict_label = np.argmax(y_hat, axis=1)
        predict_label = y_hat>0.5
        try:
            tp=np.sum(np.logical_and((y_hat>0.5),(y>0.5)))
            tn=np.sum(np.logical_and((y_hat<=0.5),(y<=0.5)))
            fp=np.sum(np.logical_and((y_hat>0.5),(y<=0.5)))
            fn=np.sum(np.logical_and((y_hat<=0.5),(y>0.5)))
        except:
            tp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()>0.5)))
            tn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()<=0.5)))
            fp=np.sum(np.logical_and((y_hat>.5),(y.data.numpy()<=0.5)))
            fn=np.sum(np.logical_and((y_hat<=.5),(y.data.numpy()>0.5)))
        f1_p=tp/(tp+0.5*(fn+fp))
        f1_n=tn/(tn+0.5*(fn+fp))
        return f1_p,f1_n

    """
    if flat_grad:
            offset = 0
            grad = utils.to_device(grad, self.use_gpu)
            for p in self.model.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for i, p in enumerate(self.model.parameters()):
                p.grad = grad[i]



    tloss = self.get_loss(xt, yt)
    grad_ft = torch.autograd.grad(tloss, self.model.parameters(), create_graph=True)        
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])


    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()],0).clone()

    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()    
    """
    