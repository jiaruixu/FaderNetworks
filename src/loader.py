# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from torch.autograd import Variable
from logging import getLogger


logger = getLogger()


# AVAILABLE_ATTR = [
#     "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
#     "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
#     "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
#     "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
#     "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
#     "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
#     "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
#     "Wearing_Necklace", "Wearing_Necktie", "Young"
# ]

AVAILABLE_ATTR = [
    "-10_0_-10", "-10_0_-5", "-10_0_0", "-10_0_10", "-10_0_15", "-10_0_20", "-10_0_5",
    "-10_10_-10", "-10_10_-5", "-10_10_0", "-10_10_10", "-10_10_15", "-10_10_20", "-10_10_5",
    "-10_15_-10", "-10_15_-5", "-10_15_0", "-10_15_10", "-10_15_15", "-10_15_20", "-10_15_5",
    "-10_20_-10", "-10_20_-5", "-10_20_0", "-10_20_10", "-10_20_15", "-10_20_20", "-10_20_5",
    "-10_5_-10", "-10_5_-5", "-10_5_0", "-10_5_10", "-10_5_15", "-10_5_20", "-10_5_5", "-5_0_-10",
    "-5_0_-5", "-5_0_0", "-5_0_10", "-5_0_15", "-5_0_20", "-5_0_5", "-5_10_-10", "-5_10_-5",
    "-5_10_0", "-5_10_10", "-5_10_15", "-5_10_20", "-5_10_5", "-5_15_-10", "-5_15_-5", "-5_15_0",
    "-5_15_10", "-5_15_15", "-5_15_20", "-5_15_5", "-5_20_-10", "-5_20_-5", "-5_20_0", "-5_20_10",
    "-5_20_15", "-5_20_20", "-5_20_5", "-5_5_-10", "-5_5_-5", "-5_5_0", "-5_5_10", "-5_5_15",
    "-5_5_20", "-5_5_5", "0_0_-10", "0_0_-5", "0_0_10", "0_0_15", "0_0_20", "0_0_5", "0_10_-10",
    "0_10_-5", "0_10_0", "0_10_10", "0_10_15", "0_10_20", "0_10_5", "0_15_-10", "0_15_-5", "0_15_0",
    "0_15_10", "0_15_15", "0_15_20", "0_15_5", "0_20_-10", "0_20_-5", "0_20_0", "0_20_10", "0_20_15",
    "0_20_20", "0_20_5", "0_5_-10", "0_5_-5", "0_5_0", "0_5_10", "0_5_15", "0_5_20", "0_5_5", "10_0_-10",
    "10_0_-5", "10_0_0", "10_0_10", "10_0_15", "10_0_20", "10_0_5", "10_10_-10", "10_10_-5", "10_10_0",
    "10_10_10", "10_10_15", "10_10_20", "10_10_5", "10_15_-10", "10_15_-5", "10_15_0", "10_15_10",
    "10_15_15", "10_15_20", "10_15_5", "10_20_-10", "10_20_-5", "10_20_0", "10_20_10", "10_20_15",
    "10_20_20", "10_20_5", "10_5_-10", "10_5_-5", "10_5_0", "10_5_10", "10_5_15", "10_5_20", "10_5_5",
    "15_0_-10", "15_0_-5", "15_0_0", "15_0_10", "15_0_15", "15_0_20", "15_0_5", "15_10_-10", "15_10_-5",
    "15_10_0", "15_10_10", "15_10_15", "15_10_20", "15_10_5", "15_15_-10", "15_15_-5", "15_15_0", "15_15_10",
    "15_15_15", "15_15_20", "15_15_5", "15_20_-10", "15_20_-5", "15_20_0", "15_20_10", "15_20_15", "15_20_20",
    "15_20_5", "15_5_-10", "15_5_-5", "15_5_0", "15_5_10", "15_5_15", "15_5_20", "15_5_5", "20_0_-10", "20_0_-5",
    "20_0_0", "20_0_10", "20_0_15", "20_0_20", "20_0_5", "20_10_-10", "20_10_-5", "20_10_0", "20_10_10", "20_10_15",
    "20_10_20", "20_10_5", "20_15_-10", "20_15_-5", "20_15_0", "20_15_10", "20_15_15", "20_15_20", "20_15_5",
    "20_20_-10", "20_20_-5", "20_20_0", "20_20_10", "20_20_15", "20_20_20", "20_20_5", "20_5_-10", "20_5_-5",
    "20_5_0", "20_5_10", "20_5_15", "20_5_20", "20_5_5", "5_0_-10", "5_0_-5", "5_0_0", "5_0_10", "5_0_15",
    "5_0_20", "5_0_5", "5_10_-10", "5_10_-5", "5_10_0", "5_10_10", "5_10_15", "5_10_20", "5_10_5", "5_15_-10",
    "5_15_-5", "5_15_0", "5_15_10", "5_15_15", "5_15_20", "5_15_5", "5_20_-10", "5_20_-5", "5_20_0", "5_20_10",
    "5_20_15", "5_20_20", "5_20_5", "5_5_-10", "5_5_-5", "5_5_0", "5_5_10", "5_5_15", "5_5_20", "5_5_5"
]

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def log_attributes_stats(train_attributes, valid_attributes, test_attributes, params):
    """
    Log attributes distributions.
    """
    k = 0
    for (attr_name, n_cat) in params.attr:
        logger.debug('Train %s: %s' % (attr_name, ' / '.join(['%.5f' % train_attributes[:, k + i].mean() for i in range(n_cat)])))
        logger.debug('Valid %s: %s' % (attr_name, ' / '.join(['%.5f' % valid_attributes[:, k + i].mean() for i in range(n_cat)])))
        logger.debug('Test  %s: %s' % (attr_name, ' / '.join(['%.5f' % test_attributes[:, k + i].mean() for i in range(n_cat)])))
        assert train_attributes[:, k:k + n_cat].sum() == train_attributes.size(0)
        assert valid_attributes[:, k:k + n_cat].sum() == valid_attributes.size(0)
        assert test_attributes[:, k:k + n_cat].sum() == test_attributes.size(0)
        k += n_cat
    assert k == params.n_attr


def load_images(params):
    """
    Load celebA dataset.
    """
    # load data
    # images_filename = 'images_%i_%i_20000.pth' if params.debug else 'images_%i_%i.pth'
    images_filename = 'images_%i_%i_200.pth' if params.debug else 'images_%i_%i.pth'
    images_filename = images_filename % (params.img_sz, params.img_sz)
    images = torch.load(os.path.join(DATA_PATH, images_filename))
    attributes = torch.load(os.path.join(DATA_PATH, 'attributes.pth'))

    # parse attributes
    attrs = []
    for name, n_cat in params.attr:
        for i in range(n_cat):
            attrs.append(torch.FloatTensor((attributes[name] == i).astype(np.float32)))
    attributes = torch.cat([x.unsqueeze(1) for x in attrs], 1)
    # split train / valid / test
    # if params.debug:
    #     train_index = 10000
    #     valid_index = 15000
    #     test_index = 20000
    # else:
    #     train_index = 162770
    #     valid_index = 162770 + 19867
    #     test_index = len(images)
    if params.debug:
        train_index = 100
        valid_index = 150
        test_index = 200
    else:
        train_index = 3000
        valid_index = 3000 + 500
        test_index = len(images)
    train_images = images[:train_index]
    valid_images = images[train_index:valid_index]
    test_images = images[valid_index:test_index]
    train_attributes = attributes[:train_index]
    valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]
    # log dataset statistics / return dataset
    logger.info('%i / %i / %i images with attributes for train / valid / test sets'
                % (len(train_images), len(valid_images), len(test_images)))
    log_attributes_stats(train_attributes, valid_attributes, test_attributes, params)
    images = train_images, valid_images, test_images
    attributes = train_attributes, valid_attributes, test_attributes
    return images, attributes


def normalize_images(images):
    """
    Normalize image values.
    """
    return images.float().div_(255.0).mul_(2.0).add_(-1)


class DataSampler(object):

    def __init__(self, images, attributes, params):
        """
        Initialize the data sampler with training data.
        """
        assert images.size(0) == attributes.size(0), (images.size(), attributes.size())
        self.images = images
        self.attributes = attributes
        self.batch_size = params.batch_size
        self.v_flip = params.v_flip
        self.h_flip = params.h_flip

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.images.size(0)

    def train_batch(self, bs):
        """
        Get a batch of random images with their attributes.
        """
        # image IDs
        idx = torch.LongTensor(bs).random_(len(self.images))

        # select images / attributes
        batch_x = normalize_images(self.images.index_select(0, idx).cuda())
        batch_y = self.attributes.index_select(0, idx).cuda()

        # data augmentation
        if self.v_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(2, torch.arange(batch_x.size(2) - 1, -1, -1).long().cuda())
        if self.h_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(3, torch.arange(batch_x.size(3) - 1, -1, -1).long().cuda())

        return Variable(batch_x, volatile=False), Variable(batch_y, volatile=False)

    def eval_batch(self, i, j):
        """
        Get a batch of images in a range with their attributes.
        """
        assert i < j
        batch_x = normalize_images(self.images[i:j].cuda())
        batch_y = self.attributes[i:j].cuda()
        return Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
