import os
import random
import h5py
import numpy as np
import openslide
import torch
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import Dataset
from utils.util import image_read
import torchvision.transforms as transforms
import pandas as pd
import cv2


class singlestainData(Dataset):
    """
    :param:
        source_dataframe
        target_dataframe
    :return:
        input: A=target, B=source, #A_paths, B_paths#
    """

    def __init__(self, opt, stage):
        self.opt = opt
        self.df = pd.read_csv(opt['test_dataframe']) if stage == 'test' else pd.read_csv(opt['train_dataframe'])
        if self.opt['name'] == 'breakhis':
            self.label_map = {'B': 0.0, 'M': 1.0}
        elif self.opt['name'] == 'tcga':
            self.label_map = {'MU': 0.0, 'WT': 1.0}
        elif self.opt['name'] in ['cam16', 'cam17']:
            self.label_map = {'NORMAL': 0.0, 'TUMOR': 1.0}
        # else:
        #     raise Exception('%s dataset not used' % self.opt['name'])
        self.label_map = {'NORMAL': 0, 'TUMOR': 1}

    @staticmethod
    def name():
        return 'single dataset for test'

    def __len__(self):
        return int(len(self.df))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[[idx % len(self.df)]]
        label_str = row.iloc[0, self.opt['label_index']]
        # label = label_str
        transform, rgb = image_read(self.opt, row, augment_fn='None', img_index=self.opt['image_index'])
        # print(label_str, self.label_map, self.label_map[label], torch.FloatTensor(self.label_map[label]), torch.tensor(2))
        return transform, rgb, torch.tensor(self.label_map[label_str])


class alignedstainData(Dataset):
    """
    :param:
        source_dataframe
        target_dataframe
    :return:
        input: A=target, B=source, #A_paths, B_paths#
    """

    def __init__(self, opt):
        self.opt = opt
        self.source_df = pd.read_csv(opt['source_dataframe'])
        self.target_df = pd.read_csv(opt['target_dataframe'])
        print('source vs target ', len(self.source_df), len(self.target_df))

    @staticmethod
    def name():
        return 'aligned dataset for Pix2pix-based'

    def __len__(self):
        target_length = int(len(self.target_df))
        source_length = int(len(self.source_df))
        return target_length if target_length > source_length else source_length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target_row = self.target_df.iloc[[idx % len(self.target_df)]]
        source_row = self.source_df.iloc[[idx % len(self.source_df)]]
        target_transform, target_rgb = image_read(self.opt, target_row, augment_fn=self.opt['augment_fn'],
                                                  img_index=self.opt['image_index'])
        source_transform, source_rgb = image_read(self.opt, source_row, augment_fn=self.opt['augment_fn'],
                                                  img_index=self.opt['image_index'])
        return {'target': (target_transform, target_rgb), 'source': (source_transform, source_rgb)}

class Camelyon16Raw(Dataset):
    def __init__(
        self, X_dtype=torch.float32,
            y_dtype=torch.float32,
            debug=False,
            data_path=None,
            logger=None,
            top_k=-1,
    ):
        self.data_path = data_path
        self.labels = pd.read_csv('/scratch/iq24/cc0395/FedDDHist/data/cam16/labels.csv')
        self.metadata = pd.read_csv('/scratch/iq24/cc0395/FedDDHist/data/cam16/metadata.csv')
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug

        self.slide_paths = []
        self.slide_coord_paths = []
        self.slide_labels = []
        self.slide_centers = []
        self.slide_sets = []
        self.patch_list = []
        self.perms = {}
        self.top_k = top_k
        self.trsforms = None
        npys_list = [
            e
            for e in sorted(self.labels['filenames'].tolist())
            if e.lower() not in ("normal_086.tif.npy", "test_049.tif.npy")
        ]
        random.seed(0)
        random.shuffle(npys_list)
        for slide in npys_list:
            slide_name = os.path.basename(slide).split(".")[0].lower()
            slide_id = int(slide_name.split("_")[1])
            slide_h5 = slide_name + ".h5"
            label_from_metadata = int(
                self.metadata.loc[
                    [
                        e.split(".")[0] == slide_name
                        for e in self.metadata["slide_name"].tolist()
                    ],
                    "label",
                ].item()
            )
            center_from_metadata = int(
                self.metadata.loc[
                    [
                        e.split(".")[0] == slide_name
                        for e in self.metadata["slide_name"].tolist()
                    ],
                    "hospital_corrected",
                ].item()
            )
            label_from_data = int(self.labels.loc[self.labels['filenames'] == slide.lower(), 'tumor'].iloc[0])
            if label_from_data == 0:
                label_str = "normal"
            else:
                label_str = "tumor"

            if "test" not in str(slide).lower():
                if slide_name.startswith("normal"):
                    # Normal slide
                    if slide_id > 100:
                        center_label = 1
                    else:
                        center_label = 0
                    label_from_slide_name = 0  # Normal slide
                elif slide_name.startswith("tumor"):
                    # Tumor slide
                    if slide_id > 70:
                        center_label = 1
                    else:
                        center_label = 0
                    label_from_slide_name = 1  # Tumor slide
                assert label_from_slide_name == label_from_data, "This shouldn't happen"
                assert center_label == center_from_metadata, "This shouldn't happen"
                stage = "train"
            else:
                stage = "test"

            slide_h5_pth = f'{self.data_path}/{stage}/{label_str}/patches/{slide_h5}'
            slide_root = self.data_path.replace('_patches', '')
            slide_pth = f'{slide_root}/{stage}/{label_str}/{slide}'
            if not os.path.exists(slide_h5_pth):
                print(f"Warning: {slide_h5_pth} does not exist")
                if logger is not None:
                    logger.warning(f"Warning: {slide_h5_pth} does not exist")
                continue

            assert label_from_metadata == label_from_data
            self.slide_paths.append(slide_pth)
            self.slide_coord_paths.append(slide_h5_pth)
            self.slide_labels.append(label_from_data)
            self.slide_centers.append(center_from_metadata)
            self.slide_sets.append(stage)
        if len(self.slide_paths) < len(self.labels.index):
            if logger is not None:
                logger.warning(
                    "Warning you are operating on a reduced dataset in  DEBUG mode with"
                    f" in total {len(self.slide_paths)}/{len(self.labels.index)}"
                    " features."
                )

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx, path=False):
        slide_pth, coord = self.patch_list[idx]  # Get WSI file path and patch coordinate
        wsi = openslide.open_slide(slide_pth)
        img = wsi.read_region(coord, 0, (256, 256)).convert('RGB')
        img = self.trsforms(img) if self.trsforms else img
        del wsi
        if path:
            return img, slide_pth
        return img


class WSIPatchDatasetCAM16(Camelyon16Raw):
    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        debug: bool = False,
        data_path: str = None,
        logger=None,
        image_size: int = 256,
        top_k=-1,
    ):
        """
        Cf class docstring
        """
        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            debug=debug,
            data_path=data_path,
            logger=logger,
            top_k=top_k,
        )
        assert center in [0, 1]
        self.centers = [center]
        if pooled:
            self.centers = [0, 1]
        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]

        to_select = [
            (self.slide_sets[idx] in self.sets)
            and (self.slide_centers[idx] in self.centers)
            for idx, _ in enumerate(self.slide_centers)
        ]
        self.slide_paths = [
            fp for idx, fp in enumerate(self.slide_paths) if to_select[idx]
        ]
        self.slide_coord_paths = [
            fp for idx, fp in enumerate(self.slide_coord_paths) if to_select[idx]
        ]
        self.slide_sets = [
            fp for idx, fp in enumerate(self.slide_sets) if to_select[idx]
        ]
        self.slide_labels = [
            fp for idx, fp in enumerate(self.slide_labels) if to_select[idx]
        ]
        self.slide_centers = [
            fp for idx, fp in enumerate(self.slide_centers) if to_select[idx]
        ]
        self.trsforms = []
        self.trsforms.append(transforms.Resize(image_size))
        self.trsforms.append(transforms.ToTensor())
        # self.trsforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.trsforms = transforms.Compose(self.trsforms)

        # Scan each WSI file and collect patch coordinate indices
        for wsi_idx, wsi_file in enumerate(self.slide_coord_paths):
            slide_pth = self.slide_paths[wsi_idx]
            if '.npy' in slide_pth:
                slide_pth = slide_pth.replace('.npy', '')
            with h5py.File(wsi_file, "r") as h5f:
                coords = h5f["coords"][:]  # Read all patch coordinates
                self.patch_list.extend([(slide_pth, coord) for coord in coords])
                if self.top_k > 0 and len(self.patch_list) > self.top_k:
                    break

class WSIPatchDatasetMQ(Dataset):
    def __init__(
        self,
        center: str = 'source',
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        debug: bool = False,
        data_path: str = None,
        logger=None,
        image_size: int = 256,
        top_k=-1,
    ):
        """
        Cf class docstring
        """
        super().__init__()
        assert center in ['source', 'target']
        self.centers = [center]
        if pooled:
            self.centers = ['source', 'target']
        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]
        self.slide_paths = []
        self.slide_coord_paths = []
        self.slide_centers = []
        self.patch_list = []
        self.top_k = top_k

        slide_path_root = f'{data_path}/{center}/raw_slide'
        for slide in os.listdir(slide_path_root):
            self.slide_paths.append(f'{slide_path_root}/{slide}')
            self.slide_centers.append(center)

        slide_coord_root = f'{data_path}/{center}/processed_slide/patches/'
        for slide in os.listdir(slide_coord_root):
            self.slide_coord_paths.append(f'{slide_coord_root}/{slide}')

        if self.sets[0] == 'test':
            # random split 20% for test
            self.slide_paths = self.slide_paths[:int(len(self.slide_paths) * 0.2)]
            self.slide_coord_paths = self.slide_coord_paths[:int(len(self.slide_coord_paths) * 0.2)]

        self.trsforms = []
        self.trsforms.append(transforms.Resize(image_size))
        self.trsforms.append(transforms.ToTensor())
        # self.trsforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.trsforms = transforms.Compose(self.trsforms)

        # Scan each WSI file and collect patch coordinate indices
        for wsi_idx, wsi_file in enumerate(self.slide_coord_paths):
            slide_pth = self.slide_paths[wsi_idx]
            if '.npy' in slide_pth:
                slide_pth = slide_pth.replace('.npy', '')
            with h5py.File(wsi_file, "r") as h5f:
                coords = h5f["coords"]  # Read all patch coordinates
                self.patch_level = h5f['coords'].attrs['patch_level']
                self.patch_size = h5f['coords'].attrs['patch_size']
                self.patch_list.extend([(slide_pth, coord) for coord in coords])
                if self.top_k > 0 and len(self.patch_list) > self.top_k:
                    break
        print('total patches: ', len(self.patch_list), self.patch_level, self.patch_size)

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx, path=False):
        slide_pth, coord = self.patch_list[idx]  # Get WSI file path and patch coordinate
        wsi = openslide.open_slide(slide_pth)
        img = wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        img = self.trsforms(img) if self.trsforms else img
        del wsi
        if path:
            return img, slide_pth
        return img

class AlignedPatchDataset(Dataset):
    def __init__(self, dataset1, dataset2, opt):
        """
        Args:
            dataset1 (WSIPatchDataset): Dataset for Center 1
            dataset2 (WSIPatchDataset): Dataset for Center 2
        """
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)
        self.opt = opt

    def __len__(self):
        # Return the minimum length to avoid index mismatch
        return min(self.len1, self.len2)

    def __getitem__(self, idx):
        """
        Randomly sample one patch from each dataset (center).
        """
        idx1 = random.randint(0, self.len1 - 1)  # Random index for dataset1
        idx2 = random.randint(0, self.len2 - 1)  # Random index for dataset2

        patch1 = self.dataset1[idx1]['img']  # Patch from center1
        patch2 = self.dataset2[idx2]['img']  # Patch from center2

        target_transform, target_rgb = image_read(self.opt, patch1, augment_fn=self.opt['augment_fn'],
                                                  img_index=self.opt['image_index'], raw=False)
        source_transform, source_rgb = image_read(self.opt, patch2, augment_fn=self.opt['augment_fn'],
                                                  img_index=self.opt['image_index'], raw=False)

        return {'target': (target_transform, patch1), 'source': (source_transform, patch2)}

class SinglePatchDataset(Dataset):
    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.len = len(self.dataset)
        self.opt = opt

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        patch = self.dataset[idx]['img']
        transform, rgb = image_read(self.opt, patch, augment_fn=self.opt['augment_fn'],
                                    img_index=self.opt['image_index'], raw=False)
        return transform, patch

class WSISlideDatasetMQ(Dataset):
    def __init__(
        self,
        split: str = 'source',
        center: list = 'HB',
        train: bool = True,
        pooled: bool = False,
        data_path: str = None,
    ):
        """
        Cf class docstring
        """
        super().__init__()
        assert split in ['source', 'target']
        self.split = [split]
        if pooled:
            self.split = ['source', 'target']
        if train:
            self.sets = ["train"]
        else:
            self.sets = ["test"]
        self.slide_paths = []
        self.slide_coord_paths = []
        slide_root = f'{data_path}/{center}/raw_slide/'
        slide_coord_root = f'{data_path}/{center}/processed_slide/patches/'
        for slide in os.listdir(slide_root):
            slide_name = slide.split('.')[0]
            self.slide_paths.append(f'{slide_root}/{slide}')
            self.slide_coord_paths.append(f'{slide_coord_root}/{slide_name}.h5')
        assert len(self.slide_paths) == len(self.slide_coord_paths)
        if self.sets[0] == 'test':
            # random split 20% for test
            self.slide_paths = self.slide_paths[:int(len(self.slide_paths) * 0.2)]
            self.slide_coord_paths = self.slide_coord_paths[:int(len(self.slide_coord_paths) * 0.2)]

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        return self.slide_paths[idx], self.slide_coord_paths[idx]

# class WSISlideDatasetTT(Dataset):
#     def __init__(
#         self,
#         split: str = 'source',
#         center: str = 'HB',
#         data_path: str = None,
#     ):
#         """
#         Cf class docstring
#         """
#         super().__init__()
#         assert split in ['source', 'target']

#         self.slide_paths = []
#         self.slide_coord_paths = []
#         if ',' in center:
#             center = center.split(',')
#             print(f'Gather from {len(center)} centers: {center}')
#             for c in center:
#                 slide_root = f'{data_path}/{split}/raw_slide/{c}/'
#                 slide_coord_root = f'{data_path}/{split}/processed_slide/{c}/patches/'
#                 for slide in os.listdir(slide_root):
#                     slide_name = slide.split('.')[0]
#                     if 'svs' in slide:
#                         slide_name = slide_name+'.svs'
#                     if not os.path.exists(f'{slide_root}/{slide}') or not os.path.exists(f'{slide_coord_root}/{slide_name}.h5'):
#                         print(f'Warning: {slide_root}/{slide} or {slide_coord_root}/{slide_name}.h5 does not exist')
#                         continue
#                     self.slide_paths.append(f'{slide_root}/{slide}')
#                     self.slide_coord_paths.append(f'{slide_coord_root}/{slide_name}.h5')
#         else:
#             print(f'Gather from {center}')
#             slide_root = f'{data_path}/{split}/raw_slide/{center}/'
#             slide_coord_root = f'{data_path}/{split}/processed_slide/{center}/patches/'
#             for slide in os.listdir(slide_root):
#                 slide_name = slide.split('.')[0]
#                 if 'svs' in slide:
#                     slide_name = slide_name + '.svs'
#                 if not os.path.exists(f'{slide_root}/{slide}') or not os.path.exists(
#                         f'{slide_coord_root}/{slide_name}.h5'):
#                     print(f'Warning: {slide_root}/{slide} or {slide_coord_root}/{slide_name}.h5 does not exist')
#                     continue
#                 self.slide_paths.append(f'{slide_root}/{slide}')
#                 self.slide_coord_paths.append(f'{slide_coord_root}/{slide_name}.h5')
#         assert len(self.slide_paths) == len(self.slide_coord_paths)

#     def __len__(self):
#         return len(self.slide_paths)

#     def __getitem__(self, idx):
#         return self.slide_paths[idx], self.slide_coord_paths[idx]

class WSISlideDatasetTT(Dataset):
    def __init__(
        self,
        split: str = 'source',
        data_path: str = None,
        use_all=False,
        test=False,
    ):
        """
        Cf class docstring
        """
        super().__init__()
        assert split in ['source', 'target']

        self.slide_paths = []
        self.slide_coord_paths = []
        slide_root = f'{data_path}/{split}/'
        slide_coord_root = f'{data_path}/processed_slide/{split}/patches/'
        for slide in os.listdir(slide_root):
            slide_name = slide.split('.')[0]
            # if 'svs' in slide:
            #     slide_name = slide_name + '.svs'
            if not os.path.exists(f'{slide_root}/{slide}') or not os.path.exists(
                    f'{slide_coord_root}/{slide_name}.h5'):
                print(f'Warning: {slide_root}/{slide} or {slide_coord_root}/{slide_name}.h5 does not exist')
                continue
            self.slide_paths.append(f'{slide_root}/{slide}')
            self.slide_coord_paths.append(f'{slide_coord_root}/{slide_name}.h5')
        assert len(self.slide_paths) == len(self.slide_coord_paths)
        if split=='source' and use_all:
            self.slide_paths = self.slide_paths
            self.slide_coord_paths = self.slide_coord_paths
        elif split=='source' and test:
            self.slide_paths = self.slide_paths[:int(len(self.slide_paths) * 0.2)]
            self.slide_coord_paths = self.slide_coord_paths[:int(len(self.slide_coord_paths) * 0.2)]
        elif split=='source' and not test:
            self.slide_paths = self.slide_paths[int(len(self.slide_paths) * 0.2):]
            self.slide_coord_paths = self.slide_coord_paths[int(len(self.slide_coord_paths) * 0.2):]

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        return self.slide_paths[idx], self.slide_coord_paths[idx]
