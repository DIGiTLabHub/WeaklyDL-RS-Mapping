import os
from torchvision import transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import json
import csv
from osgeo import gdal
import cv2
import transforms


def load_img_name_list(dataset_path):
    img_name_list = []
    with open(dataset_path, 'r') as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            img_name_list.append(row[0].strip())   
    return img_name_list

def load_image_label_list_from_json(label_file_path=None):
    with open(label_file_path, 'rb') as f:
        label_json = json.load(f)
    label_dict = label_json['BigEarthNet-19_labels_name2vale']
    return label_dict

class BigEarthNetMCTDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}.csv')
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.label_dict = load_image_label_list_from_json(label_file_path)
        self.dataset_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn
        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        i=0
        img = np.zeros((12,120,120))
        for band_name in self.band_names:
            band_path = os.path.join(self.dataset_root, name + '/' + name + '_' + band_name + '.tif')
            band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
            img[i] = cv2.resize(band_data, dsize=(120,120))
            i = i+1
        label = []
        patch_json_path = os.path.join(self.dataset_root, name + '/' + name + '_labels_metadata.json')
        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)
        original_labels = patch_json['labels']
        for label_name in original_labels:
            label.append(self.label_dict.get(label_name))
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.img_name_list)

def build_dataset(is_train, args, gen_attn=False):
    transform = build_transform(is_train, args)
    dataset = None
    nb_classes = None
    dataset = BigEarthNetClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, scales=tuple(args.scales), label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
    nb_classes = 15
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 120
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im and not args.gen_attention_maps:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class BigEarthNetDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        label_file_path = None,
        split='train',
        stage='train',
    ):
        super().__init__()
        self.root_dir = root_dir
        self.band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        self.stage = stage
        self.img_dir = root_dir
        self.label_dict = load_image_label_list_from_json(label_file_path)
        self.name_list_dir = os.path.join(root_dir, split + '.csv')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        i = 0
        img = np.zeros((12,120,120))
        for band_name in self.band_names:
            band_path = os.path.join(self.root_dir, _img_name + '/' + _img_name + '_' + band_name + '.tif')
            band_ds = gdal.Open(band_path,  gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
            img[i] = cv2.resize(band_data, dsize=(120,120))
            i = i+1
        label = []
        patch_json_path = os.path.join(self.root_dir, _img_name + '/' + _img_name + '_labels_metadata.json')
        with open(patch_json_path, 'rb') as f:
            patch_json = json.load(f)
        original_labels = patch_json['labels']
        for label_name in original_labels:
            label.append(self.label_dict.get(label_name))
        seg_label = img[:,:,0]

        return _img_name, img, label, seg_label


class BigEarthNetClsDataset(BigEarthNetDataset):
    def __init__(self,
                 root_dir=None,
                 label_file_path=None,
                 split='train',
                 stage='train',
                 resize_range=[60, 240],
                 rescale_range=[0.5, 2.0],
                 crop_size=60,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=15,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, label_file_path, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 30
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        # self.color_jittor = transforms.PhotoMetricDistortion()

        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)

        self.label_list = load_image_label_list_from_json(label_file_path)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])
        
        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            # self.gaussian_blur(p=1.0),
            # self.normalize,
        ])
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            # self.flip_and_color_jitter,
            # self.gaussian_blur(p=0.1),
            # self.solarization,
            self.normalize,
        ])
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            # self.flip_and_color_jitter,
            # self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        img_box = None
        local_image = None
        if self.aug:

            if self.rescale_range:
                image = transforms.random_scaling(image, scale_range=self.rescale_range)
            if self.img_fliplr:
                image = transforms.random_fliplr(image)
            if self.crop_size:
                image, img_box = transforms.random_crop(image, crop_size=self.crop_size, mean_rgb=[0,0,0], ignore_index=self.ignore_index)
            
            local_image = self.local_view(Image.fromarray(image))
            # image = self.global_view1(Image.fromarray(image))
        
        image = self.normalize(image)
        
        return image, local_image, img_box

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)
        
        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        img_name, image, cls_label, _ = super().__getitem__(idx)
        
        pil_image = Image.fromarray(image)

        image, local_image, img_box = self.__transforms(image=image)

        if self.aug:

            crops = []
            crops.append(image)
            crops.append(self.global_view2(pil_image))
            crops.append(local_image)
            # for _ in range(8):
            #     crops.append(self.local_view(pil_image))

            return img_name, image, cls_label, img_box, crops
        else:
            return img_name, image, cls_label


class BigEarthNetSegDataset(BigEarthNetDataset):
    def __init__(self,
                 root_dir=None,
                 label_file_path=None,
                 split='train',
                 stage='train',
                 resize_range=[60, 240],
                 rescale_range=[0.5, 2.0],
                 crop_size=30,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, label_file_path, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_image_label_list_from_json(label_file_path)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            if self.img_fliplr:
                image, label = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = transforms.random_crop(image, label, crop_size=self.crop_size, mean_rgb=[123.675, 116.28, 103.53], ignore_index=self.ignore_index)
        '''
        if self.stage != "train":
            image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        '''
        image = transforms.normalize_img(image)
        ## to chw
        image = np.transpose(image, (2, 0, 1))

        return image, label

    def __getitem__(self, idx):
        img_name, image, cls_label, label = super().__getitem__(idx)

        image, label = self.__transforms(image=image, label=label)

        return img_name, image, label, cls_label
