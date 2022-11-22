import logging
from pathlib import Path

import os
import sys
DS_DIR = os.path.dirname(os.path.abspath(__file__))
EPE_DIR = os.path.dirname(DS_DIR)
CODE_DIR = os.path.dirname(EPE_DIR)

sys.path.append(DS_DIR)
sys.path.append(EPE_DIR)
sys.path.append(CODE_DIR)

import IPython
import imageio
from PIL import Image
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch

from batch_types import EPEBatch
from synthetic import SyntheticDataset
from dataset.utils import mat2tensor, normalize_dim, read_filelist
from Carla.generate_dataset_file import Carla


def center(x, m, s):
	x[0,:,:] = (x[0,:,:] - m[0]) / s[0]
	x[1,:,:] = (x[1,:,:] - m[1]) / s[1]
	x[2,:,:] = (x[2,:,:] - m[2]) / s[2]
	return x


class CarlaDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='carla'):
		"""
		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(CarlaDataset, self).__init__('CarlaDataset')

		assert gbuffers in ['carla']

		self.carla = Carla()

		self.transform = transform
		self.gbuffers  = gbuffers
		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass

		try:
			data = np.load(Path(__file__).parent.parent / 'stats/carla_stats.npz')
			# self._img_mean  = data['i_m']
			# self._img_std   = data['i_s']
			self._gbuf_mean = data['g_m']
			self._gbuf_std  = data['g_s']
			self._log.info(f'Loaded dataset stats.')
		except:
			# self._img_mean  = None
			# self._img_std   = None
			self._gbuf_mean = None
			self._gbuf_std  = None
			pass

		self._log.info(f'Found {len(self._paths)} samples.')
		pass


	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		# return 2
		return 16


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return 11


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, _, gt_label_path = self._paths[index]

		filename = img_path.__str__().split("/")[-1][:-4]
		gbuffer_path = os.path.join('/'.join(img_path.__str__().split("/")[:-2]), "gbuffer_v")
		
		g_buffer_list = ["SceneDepth", "SceneColor", "SceneGBufferA", "SceneGBufferB", "SceneGBufferC", "SceneGBufferD"]
		gbuffer_filenames = [f"{filename}-{gbuffer_name}.png" for gbuffer_name in g_buffer_list]

		data = {}
		data['img'] = np.array(imageio.imread(img_path))

		# Read from pngs
		base_path = gbuffer_path

		img_scene_depth = Image.open(os.path.join(base_path, gbuffer_filenames[0]))
		data['depth'] = np.array(img_scene_depth)[:,:,0].astype(np.float32) / 255.0

		img_albedo = Image.open(os.path.join(base_path, gbuffer_filenames[1]))
		data['albedo'] = np.array(img_albedo)[:, :, 0:3].astype(np.float32) / 255.0

		img_normal = Image.open(os.path.join(base_path, gbuffer_filenames[2]))
		data['normal'] = np.array(img_normal)[:, :, 0:3].astype(np.float32) / 255.0

		img_gbufferB = Image.open(os.path.join(base_path, gbuffer_filenames[3]))
		data['gbufferB'] = np.array(img_gbufferB)[:, :, 0:3].astype(np.float32) / 255.0

		img_gbufferC = Image.open(os.path.join(base_path, gbuffer_filenames[4]))
		data['gbufferC'] = np.array(img_gbufferC)[:, :, 0:3].astype(np.float32) / 255.0

		img_gbufferD = Image.open(os.path.join(base_path, gbuffer_filenames[5]))
		data['gbufferD'] = np.array(img_gbufferD)[:, :, 0:3].astype(np.float32) / 255.0

		data['gbuffers'] = np.concatenate(
			[
				data['depth'][:, :, np.newaxis],
				data['albedo'],
				data['normal'],
				data['gbufferB'],
				data['gbufferC'],
				data['gbufferD']
			], axis=2
		)

		robust_label_color = np.array(imageio.imread(robust_label_path))
		if len(robust_label_color.shape) > 2 and robust_label_color.shape[2] == 3: # Robust label png contains RGB values
			robust_label_map = np.ndarray(robust_label_color.shape[:2])
			for idx, color in enumerate(Carla.color2id.keys()):
				robust_label_map[(robust_label_color == color).all(axis=2)] = Carla.color2id[tuple(color)]
		else: # Robust label png contains label ID
			robust_label_map = robust_label_color

		gt_label_map = np.array(imageio.imread(gt_label_path))
		mask = np.zeros(shape=(gt_label_map.shape[0], gt_label_map.shape[1], 12))
		for idx, color in enumerate(Carla.color2id.keys()):
			mask[:, :, Carla.color2id[tuple(color)] ] += (gt_label_map == color).all(axis=2)
		data['shader'] = mask

		img       = mat2tensor(data['img'].astype(np.float32) / 255.0)
		robust_labels = mat2tensor(robust_label_map).long()
		gbuffers  = mat2tensor(data['gbuffers'].astype(np.float32))
		gt_labels = mat2tensor(data['shader'].astype(np.float32))


		if self._gbuf_mean is not None:
			gbuffers = center(gbuffers, self._gbuf_mean, self._gbuf_std)
			pass

		if not robust_label_path.exists():
			self._log.error(f'Robust labels at {robust_label_path} do not exist.')
			raise FileNotFoundError
			pass

		# label_map = [gt_labels[k][np.newaxis, :, :] * k for k in range(12)]
		# label_map = label_map[0:9] + label_map[10:12]  # Exclude 9
		# robust_labels = np.concatenate(label_map, axis=0).max(axis=0)[np.newaxis, :, :]
		# robust_labels =	torch.Tensor(robust_labels).long()

		gt_labels = torch.concat([gt_labels[0:9, :, :], gt_labels[10:12, :, :]], dim=0)


		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)

if __name__ == "__main__":
    data_path = "data/file_lists/fake_dataset/Carla/file.txt"
    dset = CarlaDataset((read_filelist(data_path, 4, True)))
    from IPython import embed
    from p_tqdm import p_map
    embed()

    def check(i):
        try:
            dset.__getitem__(i)
            return True
        except:
            return False
    
    L = dset.__len__()
    result = p_map(check, range(L), num_cpus=16)
    
    with open("broken.txt", 'w') as f:
        f.write(str(result))
