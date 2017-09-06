import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from PIL import ImageFile

batch_size = 16
learning_rate = 0.0002
num_epoch = 10

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(),target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Linear(28*28,20)
        self.decoder = nn.Linear(20,28*28)   
                
    def forward(self,x):
        x = x.view(batch_size,-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size,1,28,28)
                
        return encoded,out
    
model = Autoencoder().cuda()

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("Training Start")

for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = Variable(image).cuda()
        
        optimizer.zero_grad()
        _,output = model.forward(x)
        loss = loss_func(output,x)
        loss.backward()
        optimizer.step()
        
    if j % 1000 == 0:
        print(loss)

print("Test data encoding")

total_arr = []
for i in range(1):
    for j,[image,label] in enumerate(test_loader):
        x = Variable(image).cuda()
        
        optimizer.zero_grad()
        encoded,output = model.forward(x)
        for k in range(batch_size):
            total_arr.append(encoded[k].view(-1).cpu().data.numpy())
        
        if j >125:
            break
            
print(len(total_arr))

print("\n------Starting TSNE------\n")

tsne_model = TSNE(n_components=2, init='pca',random_state=0)
result = tsne_model.fit_transform(total_arr)

print("\n------TSNE Done------\n")

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = image
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

print("\n------Starting to plot------\n")

mnist_test = dset.MNIST("./", train=False,target_transform=None, download=True)

for i in range(len(result)):
    print("{}/{}".format(i,len(result)))
    image = mnist_test[i][0]
    imscatter(result[i,0],result[i,1], image=image ,zoom=0.2)

plt.show()