import os

import numpy as np

from cmvae_utils.dataset_utils import load_img_from_file_or_array_and_resize_cv2

# NB CV2 Reads images in the form height x width
res = (64, 64, 3)

dir = "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\data\\image_similarity\\1920x1080\\64x64"
imgs = np.zeros(np.concatenate(([len(os.listdir(dir))], res))).astype(np.int8)

for idx, filename in enumerate(os.listdir(dir)):
    imgs[idx, :] = load_img_from_file_or_array_and_resize_cv2(file=os.path.join(dir, filename), res=res, normalise=False)


dir = "C:\\Users\\sambu\\Documents\\Repositories\\CodeBases\\SWiMM_DEEPeR\\data\\test\\1920x1080\\64x64"
imgs_1 = np.zeros(np.concatenate(([len(os.listdir(dir))], res))).astype(np.int8)

for idx, filename in enumerate(os.listdir(dir)):
    imgs_1[idx, :] = load_img_from_file_or_array_and_resize_cv2(file=os.path.join(dir, filename), res=res, normalise=False)


assert np.array_equal(imgs, imgs_1)
