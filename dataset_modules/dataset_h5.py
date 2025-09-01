import numpy as np
import pandas as pd
from skimage.color import rgb2hed
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None,
		HED=False,
		reform_4k=False):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms
		if img_transforms:
			self.roi_transforms = transforms.Compose([img_transforms, transforms.ToTensor()])
		else:
			self.roi_transforms = transforms.ToTensor()

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
		self.HED = HED
		self.reform_4k = reform_4k
		# self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		# check self.wsi has read_region method
		if not hasattr(self.wsi, 'read_region'):
			img = self.wsi.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		else:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		if self.HED:
			img = rgb2hed(np.array(img))
		img = self.roi_transforms(img)
		# if self.reform_4k:
		# 	c, h, w = img.size()
		# 	w_256 = w//256
		# 	h_256 = h//256
		# 	img = img.view(c, w_256, h_256, 256, 256)
		# 	img = img.view(w_256*h_256, c, 256, 256)
			# print('reforming 4k ', img.size())
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




