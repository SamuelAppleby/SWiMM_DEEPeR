import os

import numpy as np
from PIL import Image
from numpy.linalg import norm

imgs = ([], [])
dir = "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SiM_DEEPeR\\Logs\\run_0"

for file in os.listdir(dir):
    if file.endswith('.jpg'):
        img = np.array(Image.open(os.path.join(dir, file))).flatten() / 255
        imgs[0].append(img)

dir = "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SiM_DEEPeR\\Logs\\run_1"

for file in os.listdir(dir):
    if file.endswith('.jpg'):
        img = np.array(Image.open(os.path.join(dir, file))).flatten() / 255
        imgs[1].append(img)

for img_1, img_2 in zip(*imgs):
    result = np.dot(img_1, img_2) / (norm(img_1) * norm(img_2))
    print(result)
    assert result == 1

