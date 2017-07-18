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

noise = init.normal(torch.FloatTensor(num_data,1),std=0.2)
x = init.uniform(torch.Tensor(num_data,1),-10,10)
y = 2*x+3
y_noise = 2*(x+noise)+3

input_data = torch.cat([x,y_noise],1)

win=viz.scatter(
	X = input_data,
	opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-20,
        ytickmax=20,
        ytickstep=1,
        markersymbol='dot',
        markercolor=np.random.randint(0, 255, num_data),
    ),
)

viz.updateTrace(
	X = x,
	Y = y,
	win=win,
	)

# model & optimizer

model = nn.Linear(1,1)
output = model(Variable(x))

loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(),lr=1)

# train
loss_arr =[]
label = Variable(y_noise)
for i in range(1000):
	output = model(Variable(x))
	optimizer.zero_grad()

	loss = loss_func(output,label)
	loss.backward()
	optimizer.step()
	#print(loss)
	loss_arr.append(loss.data.numpy()[0])


param_list = list(model.parameters())
print(param_list[0].data,param_list[1].data)

# visualize 

win2=viz.line(
	X = x,
	Y = output.data,
	opts=dict(
        xtickmin=-10,
        xtickmax=10,
        xtickstep=1,
        ytickmin=-20,
        ytickmax=20,
        ytickstep=1,
        markersymbol='dot',
    ),
)

# loss 

x = np.reshape([i for i in range(1000)],newshape=[1000,1])
loss_data = np.reshape(loss_arr,newshape=[1000,1])

win3=viz.line(
	X = x,
	Y = loss_data,
	opts=dict(
        xtickmin=0,
        xtickmax=1000,
        xtickstep=1,
        ytickmin=0,
        ytickmax=20,
        ytickstep=1,
        markercolor=np.random.randint(0, 255, 1000),
    ),
)

