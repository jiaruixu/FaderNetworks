#!/usr/bin/env python
import os
import matplotlib.image as mpimg
import cv2
import numpy as np
import torch
import glob


N_IMAGES = 3904
IMG_SIZE = 256
IMG_PATH = 'images_%i_%i.pth' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes.pth'
DATASET_PATH = '/mnt/fcav/3D-lighting/dataset/data_attributes2'
# DATASET_PATH = '/Users/jiarui/git/FaderNetworks/data/all_rgb'
# IMAGE_PATHS = glob.glob(DATASET_PATH + '/all_rgb/*.png')
# N_IMAGES = len(IMAGE_PATHS)


def preprocess_images():

    if os.path.isfile(IMG_PATH):
        print("%s exists, nothing to do." % IMG_PATH)
        return

    print("Reading images from all_rgb/ ...")

    all_images = []
    for i in range(N_IMAGES):
        # all_images.append(mpimg.imread(IMAGE_PATHS[i]))
        image_file = glob.glob('%s/all_rgb/*_%d.png' % (DATASET_PATH, i))
        all_images.append(mpimg.imread(image_file[0]))

    data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
    data = torch.from_numpy(data)
    assert data.size() == (N_IMAGES, 3, IMG_SIZE, IMG_SIZE)

    print("Saving images to %s ..." % IMG_PATH)
    torch.save(data[:200].clone(), 'images_%i_%i_200.pth' % (IMG_SIZE, IMG_SIZE))
    torch.save(data, IMG_PATH)


def preprocess_attributes():

    if os.path.isfile(ATTR_PATH):
        print("%s exists, nothing to do." % ATTR_PATH)
        return

    attr_lines = [line.rstrip() for line in open('%s/cornellbox_attribute_list_rgb.txt' % DATASET_PATH, 'r')]
    # attr_lines = [line.rstrip() for line in open('/Users/jiarui/git/FaderNetworks/data/cornellbox_attribute_list_rgb_header.txt', 'r')]

    attr_keys = attr_lines[0].split()
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[1:]):
        split = line.split()
        assert all(x in ['0', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = value == '1'

    print("Saving attributes to %s ..." % ATTR_PATH)
    torch.save(attributes, ATTR_PATH)


preprocess_images()
preprocess_attributes()
