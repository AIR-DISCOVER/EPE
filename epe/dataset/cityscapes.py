import IPython
import imageio
import numpy as np
import torch
from .robust_labels import RobustlyLabeledDataset
from .batch_types import EPEBatch
from .utils import mat2tensor


def transform_labels(original_label_map):
    # if name in ['unlabeled', 'water', 'other', 'dynamic']: return 11
    # if name in ['building', 'fence', 'bridge']: return 10
    # if name in ['pole']: return 6
    # if name in ['wall', 'rail_track', 'guard_rail', 'road_line']: return 10
    # if name in ['person']: return 5
    # if name in ['sky']: return 0
    # if name in ['road', 'static', 'sidewalk', 'ground']: return 1
    # if name in ['car']: return 2
    # if name in ['vegetation']: return 4
    # if name in ['traffic_light']: return 7
    # if name in ['traffic_sign']: return 8
    # if name in ['terrain']: return 3

    label_map = np.zeros((original_label_map.shape[0], original_label_map.shape[1], 12), dtype=np.long)
    label_map[:,:,0] = (original_label_map == 23).astype(np.long) # sky
    label_map[:,:,1] = (np.isin(original_label_map, [4, 6, 7, 8, 9, 10])).astype(np.long) # road / static / sidewalk
    label_map[:,:,2] = (np.isin(original_label_map, [1,26,27,28,29,30,31,32,33])).astype(np.long) # vehicle
    label_map[:,:,3] = (original_label_map == 22).astype(np.long) # terrain
    label_map[:,:,4] = (original_label_map == 21).astype(np.long) # vegetation
    label_map[:,:,5] = (np.isin(original_label_map, [24, 25])).astype(np.long) # person
    label_map[:,:,6] = (np.isin(original_label_map, [17, 18])).astype(np.long) # pole
    label_map[:,:,7] = (np.isin(original_label_map, [19])).astype(np.long) # traffic light
    label_map[:,:,8] = (np.isin(original_label_map, [20])).astype(np.long) # traffic sign
    label_map[:,:,10] = (np.isin(original_label_map, [11, 12, 13, 14, 15, 16])).astype(np.long) # building
    # label_map[:,:,11] = (np.isin(original_label_map, [0, 2, 3, 5, 255])).astype(np.long) # other

    for k in range(12):
        label_map[:,:,k] = label_map[:,:,k] * k

    label_map = np.concatenate([label_map[:, :, k][np.newaxis, :, :] for k in range(12)], axis=0)\
                    .max(axis=0)[np.newaxis, :, :]
    return label_map


class Cityscapes(RobustlyLabeledDataset):
    def __init__(self, name, img_and_robust_label_paths, img_transform=None, label_transform=None):
        super().__init__(name, img_and_robust_label_paths, img_transform, label_transform)

    def __getitem__(self, index):

        idx = index % self.__len__()
        img_path = self.paths[idx]
        img = self._load_img(img_path)

        if self.transform is not None:
            img = self.transform(img)
            pass

        img = mat2tensor(img)

        label_path = self._img2label[img_path]
        robust_labels = imageio.imread(label_path)

        robust_labels = torch.LongTensor(transform_labels(robust_labels))

        return EPEBatch(img, path=img_path, robust_labels=robust_labels)