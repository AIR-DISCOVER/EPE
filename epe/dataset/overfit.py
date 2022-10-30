import logging
from pathlib import Path

import IPython
import imageio
import numpy
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch

from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim
from Carla.generate_dataset_file import Carla


def center(x, m, s):
	x[0,:,:] = (x[0,:,:] - m[0]) / s[0]
	x[1,:,:] = (x[1,:,:] - m[1]) / s[1]
	x[2,:,:] = (x[2,:,:] - m[2]) / s[2]
	return x


class OverfitDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='overfit'):
		"""
		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(OverfitDataset, self).__init__('OverfitDataset')

		self.transform = transform
		self.gbuffers  = gbuffers

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass


		self._gbuf_mean = None
		self._gbuf_std  = None

		self.data = None


		self._log.info(f'Found {len(self._paths)} samples.')
		pass


	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		# return 2
		return 26


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return 12


	@property
	def cls2gbuf(self):
		return {}


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, gbuffer_path, gt_label_path = self._paths[index]

		if not gbuffer_path.exists():
			self._log.error(f'Gbuffers at {gbuffer_path} do not exist.')
			raise FileNotFoundError
			pass

		if self.data is None:
			print("===> Start Loading data...")
			import time; t_start = time.time()
			self.data = np.load(gbuffer_path)['item']  # To accelerate loading process
			print(f"===> Loading data finished in {time.time() - t_start}...")

		data = self.data

		img = mat2tensor(data[:, :, 0:3])
		gbuffers = mat2tensor(data[:, :, 3:29])

		shader_map = np.zeros(shape=(526, 957, 12))
		shader_map[:, :, 0] = data[:, :, 29]  # sky
		shader_map[:, :, 1] = data[:, :, 30]  # road / static
		shader_map[:, :, 2] = data[:, :, 34]  # vehicle
		shader_map[:, :, 3] = data[:, :, 36]  # terrain  # empty :(
		shader_map[:, :, 4] = data[:, :, 33]  # vegetation
		shader_map[:, :, 5] = data[:, :, 41]  # person
		shader_map[:, :, 6] = data[:, :, 35]  # infrastructure
		shader_map[:, :, 7] = data[:, :, 36]  # traffic light  # empty :(
		shader_map[:, :, 8] = data[:, :, 36]  # traffic sign  # empty :(
		shader_map[:, :, 9] = data[:, :, 36]  # ego vehicle  # empty :(
		shader_map[:, :, 10] = data[:, :,39]  # building
		shader_map[:, :, 11] = data[:, :,42]  # unlabelled

		gt_labels = mat2tensor(shader_map.astype(np.float32).round())

		if not robust_label_path.exists():
			self._log.error(f'Robust labels at {robust_label_path} do not exist.')
			raise FileNotFoundError
			pass

		label_map = [gt_labels[k][np.newaxis, :, :] * k for k in range(12)]
		robust_labels = np.concatenate(label_map, axis=0).max(axis=0)[np.newaxis, :, :]
		robust_labels =	torch.Tensor(robust_labels).long()

		# img =  mat2tensor(np.array(imageio.imread(img_path))).float() / 255.0
		# gbuffers = np.load(gbuffer_path)['data'].transpose((2, 0, 1))[0:1, :, :]
		# gtlabel_map = np.array(imageio.imread(gt_label_path))
		# mask = np.zeros(shape=(gtlabel_map.shape[0], gtlabel_map.shape[1], 12))
		# for idx, color in enumerate(Carla.color2id.keys()):
		# 	mask[:, :, Carla.color2id[tuple(color)]] += (gtlabel_map == color).all(axis=2)
		# gt_labels = mask.transpose((2, 0, 1))
		# robust_labels = np.concatenate([mask[:, :, k][np.newaxis, :, :] * k for k in range(12)], axis=0)\
		# 				.max(axis=0)[np.newaxis, :, :]
		#
		# # Convert to tensor
		# gbuffers = torch.Tensor(gbuffers).float()
		# gt_labels = torch.Tensor(gt_labels).float()
		# robust_labels = torch.Tensor(robust_labels).long()


		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)
