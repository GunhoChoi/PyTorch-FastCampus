from PIL import Image
import os
import tarfile
import shutil
cwd = os.getcwd()

img_size = 256

folder_dir = "/image"

folder_list = os.listdir("."+folder_dir)

for i in folder_list:
	img_list = os.listdir("./image/"+i)
	for j in img_list:
		print(j)
		image = Image.open("."+folder_dir+'/'+i+'/'+j)
		image = image.resize([img_size,img_size])
		try:
			image.save("."+folder_dir+"/"+i+"/"+j)
		except:
			pass
