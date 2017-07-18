import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from visdom import Visdom
viz = Visdom()

# data generation

num_data = 1000
num_epoch = 5000

noise = init.normal(torch.FloatTensor(num_data,1),std=0.5)
x = init.uniform(torch.Tensor(num_data,1),-10,10)
y = (x**3) - 3*(x**2) - 9*x - 1
x_noise = x + noise
y_noise = (x_noise**3) -3*(x_noise**2) - 9*x_noise - 1

input_data = torch.cat([x,y_noise],1)

win=viz.scatter(
	X = input_data,
	opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=0,
        ytickmax=500,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=5,
    ),
)

viz.updateTrace(
	X = x,
	Y = y,
	win=win,
)

# model & optimizer

model = nn.Sequential(
            nn.Linear(1,20),
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
optimizer = optim.Adam(model.parameters(),lr=0.003)

# train
loss_arr =[]
label = Variable(y_noise.cuda())
for i in range(num_epoch):
	output = model(Variable(x.cuda()))
	optimizer.zero_grad()

	loss = loss_func(output,label)
	loss.backward()
	optimizer.step()
	#print(loss)
	loss_arr.append(loss.cpu().data.numpy()[0])

# visualize 

win2=viz.scatter(
    X = input_data,
    opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=0,
        ytickmax=500,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
        markersize=5,
    ),
)


viz.updateTrace(
	X = x,
	Y = output.cpu().data,
	win = win2,
	opts=dict(
        xtickmin=-15,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-0,
        ytickmax=500,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, (num_data, 3,)),
    ),
)

# loss 

x = np.reshape([i for i in range(num_epoch)],newshape=[num_epoch,1])
loss_data = np.reshape(loss_arr,newshape=[num_epoch,1])

win3=viz.line(
	X = x,
	Y = loss_data,
	opts=dict(
        xtickmin=0,
        xtickmax=num_epoch,
        xtickstep=1,
        ytickmin=0,
        ytickmax=20,
        ytickstep=1,
        markercolor=np.random.randint(0, 255, num_epoch),
    ),
)

param_list = list(model.parameters())
print(param_list[0].data,param_list[1].data)