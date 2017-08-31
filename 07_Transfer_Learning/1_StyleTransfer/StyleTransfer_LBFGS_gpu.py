# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
# https://github.com/leongatys/PytorchNeuralStyleTransfer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torch.utils.data as data
import torchvision.models as models
import torchvision.utils as v_utils
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image

# hyperparameters

content_dir = "./image/content/Neckarfront_origin.jpg"
style_dir = "./image/style/monet.jpg"

content_layer_num = 4
image_size = 512
epoch = 10000

# import pretrained resnet50 model
# check what layers are in the model

resnet = models.resnet50(pretrained=True)
for name,module in resnet.named_children():
	print(name)

# resnet without fully connected layers
# return activations in each layers 

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.layer0 = nn.Sequential(*list(resnet.children())[0:1])
        self.layer1 = nn.Sequential(*list(resnet.children())[1:4])
        self.layer2 = nn.Sequential(*list(resnet.children())[4:5])
        self.layer3 = nn.Sequential(*list(resnet.children())[5:6])
        self.layer4 = nn.Sequential(*list(resnet.children())[6:7])
        self.layer5 = nn.Sequential(*list(resnet.children())[7:8])

    def forward(self,x):
        out_0 = self.layer0(x)
        out_1 = self.layer1(out_0)
        out_2 = self.layer2(out_1)
        out_3 = self.layer3(out_2)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)

        return out_0, out_1, out_2, out_3, out_4, out_5
# read & preprocess image

def image_preprocess(img_dir):
	img = Image.open(img_dir)
	transform = transforms.Compose([
					transforms.Scale(image_size),
					transforms.CenterCrop(image_size),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], 
                                         std=[1,1,1]),
				])
	img = transform(img).view((-1,3,image_size,image_size))
	return img

# image post process

def image_postprocess(tensor):
	transform = transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], 
                                     std=[1,1,1])
	img = transform(tensor.clone())
	img[img>1] = 1    
	img[img<0] = 0
	return 2*img-1

# show image given a tensor as input

def imshow(tensor):
    image = tensor.clone().cpu()
    image = image.view(3, image_size, image_size)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.show()

# gram matrix

class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        return G

# gram matrix mean squared error

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)

# initialize resnet and put on gpu
# model is not updated so .requires_grad = False

resnet = Resnet().cuda()

for param in resnet.parameters():
    param.requires_grad = False

# get content and style image & image to be generated

content = Variable(image_preprocess(content_dir), requires_grad=False).cuda()
style = Variable(image_preprocess(style_dir), requires_grad=False).cuda()
generated = Variable(content.data.clone(),requires_grad=True)

v_utils.save_image(image_postprocess(image_preprocess(content_dir)),"./content.png")
v_utils.save_image(image_postprocess(image_preprocess(style_dir)),"./style.png")

# set targets and style weights

style_target = list(GramMatrix().cuda()(i) for i in resnet(style))
content_target = resnet(content)[content_layer_num]
style_weight = [1/n**2 for n in [64,64,256,512,1024,2048]]

# set LBFGS optimizer
# change the image through training

optimizer = optim.LBFGS([generated])

iteration = [0]
while iteration[0] < epoch:

	def closure():
		optimizer.zero_grad()
		out = resnet(generated)
		style_loss = [GramMSELoss().cuda()(out[i],style_target[i])*style_weight[i] for i in range(len(style_target))]
		content_loss = nn.MSELoss().cuda()(out[content_layer_num],content_target)
		total_loss = 1000 * sum(style_loss) + sum(content_loss)
		total_loss.backward()

		if iteration[0] % 100 == 0:
			print(total_loss)
			v_utils.save_image(image_postprocess(generated.data),"./gen_{}.png".format(iteration[0]))
		iteration[0] += 1

		return total_loss
	
	optimizer.step(closure)