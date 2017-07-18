import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from visdom import Visdom
viz = Visdom()

num_data=1000
num_epoch=400

x = init.uniform(torch.Tensor(num_data,1),-10,10)
y = init.uniform(torch.Tensor(num_data,1),-10,10)
z = x**2 + y**2

x_noise = x + init.normal(torch.FloatTensor(num_data,1),std=0.5)
y_noise = y + init.normal(torch.FloatTensor(num_data,1),std=0.5)
z_noise = x_noise**2 + y_noise**2
data_noise = torch.cat([x_noise,y_noise,z_noise],1)

# visualize data

win_1=viz.scatter(
		X=data_noise,
		opts=dict(
			markersize=5,
			markercolor=np.ndarray(shape=[num_data,3],dtype=float,buffer=[51,153,255]*np.ones(shape=[num_data,3]))
			)
		)

model = nn.Sequential(
            nn.Linear(2,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,5),
            nn.ReLU(),
            nn.Linear(5,1),
        ).cuda()

loss_func = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

input_data = torch.cat([x,y],1).cuda()
label = z_noise.cuda()
for i in range(num_epoch):
	optimizer.zero_grad()
	output = model(Variable(input_data))
	loss = loss_func(output,Variable(label))
	loss.backward()
	optimizer.step()
	print(loss)

	data = torch.cat([input_data.cpu(),output.cpu().data],1)

	if i % (num_epoch/10) == 0:
		win_2 =viz.scatter(
					X=data,
					opts=dict(
					markersize=5,
					markercolor=np.ndarray(shape=[num_data,3],dtype=float,buffer=128*np.ones(shape=[num_data,3]))
				)
			)

# np.ndarray(shape=[num_data,3],dtype=float,buffer=[0,0,255]*np.ones(shape=[num_data,3]))