import os
import torch
import numpy as np

DATA_PATH = '/home/jiarui/git/FaderNetworks/data'

images_filename = 'images_%i_%i_200.pth'
images_filename = images_filename % (256, 256)
images = torch.load(os.path.join(DATA_PATH, images_filename))

test_image = images[:1]
test_image = test_image.cuda().float().mul_(255.0).cpu()
npimg = test_image.view(3, 256, 256).numpy()
import matplotlib.pyplot as plt
plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
plt.show()