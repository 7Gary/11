import os
from bisect import bisect_right
from enum import Enum

import PIL
import torch
from torchvision import transforms

# 自定义数据集类名（根据您的实际数据集修改）
_CUSTOM_CLASSNAMES = [
    "bupi",
    # 添加您的自定义类别
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for custom datasets without ground truth masks.
    """

    def __init__(
            self,
            source,
            classname,
            resize=None,
            imagesize=None,
            split=DatasetSplit.TRAIN,
            train_val_split=1.0,
            tile_size=512,
            tile_overlap=0,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the custom data folder.
            classname: [str or None]. Name of class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available classes.
            resize: [int or None]. Optional (square) size to which the loaded
                    image tile is resized before further processing.
            imagesize: [int or None]. Optional (square) size the resized loaded
                       image tile gets (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used.
            tile_size: [int]. Size of the (square) tiles extracted from the
                        original image.
            tile_overlap: [int]. Overlap between neighbouring tiles measured
                           in pixels. Defaults to 0 (no overlap).
        """
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CUSTOM_CLASSNAMES
        self.train_val_split = train_val_split

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = []
        self.transform_mask_ops = []

        if resize is not None:
            self.transform_img.append(transforms.Resize(resize))
            self.transform_mask_ops.append(
                transforms.Resize(resize, interpolation=PIL.Image.NEAREST)
            )
        if imagesize is not None:
            self.transform_img.append(transforms.CenterCrop(imagesize))
            self.transform_mask_ops.append(transforms.CenterCrop(imagesize))

        self.transform_img.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform_mask_ops.append(transforms.ToTensor())

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask_ops)

        if imagesize is not None:
            self.imagesize = (3, imagesize, imagesize)
        elif resize is not None:
            self.imagesize = (3, resize, resize)
        else:
            self.imagesize = (3, self.tile_size, self.tile_size)

        self.tile_bboxes_cache = {}
        self.tiles_per_image = []
        self.cumulative_tiles = []
        self.total_tiles = 0

        self._prepare_tiles_metadata()

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_tiles:
            raise IndexError("Index out of range for dataset tiles.")

        image_idx = bisect_right(self.cumulative_tiles, idx)
        prev_cumulative = self.cumulative_tiles[image_idx - 1] if image_idx > 0 else 0
        tile_idx = idx - prev_cumulative

        classname, anomaly, image_path, mask_path = self.data_to_iterate[image_idx]

        bboxes = self.tile_bboxes_cache[image_path]
        bbox = bboxes[tile_idx]

        with PIL.Image.open(image_path) as pil_image:
            tile = pil_image.crop(bbox).convert("RGB")
        image = self.transform_img(tile)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            with PIL.Image.open(mask_path) as pil_mask:
                mask = pil_mask.crop(bbox)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        slice_suffix = f"_tile_{tile_idx}_x{bbox[0]}_y{bbox[1]}"
        image_name = "/".join(image_path.split("/")[-4:]) + slice_suffix

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_name,
            "image_path": image_path,
        }

        # return {
        #     "image": image,
        #     "mask": mask,
        #     "classname": classname,
        #     "anomaly": anomaly,
        #     "is_anomaly": int(anomaly != "good"),
        #     "image_name": "/".join(image_path.split("/")[-4:]),
        #     "image_path": image_path,
        # }

    def __len__(self):
        return self.total_tiles

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)

            # 检查路径是否存在
            if not os.path.exists(classpath):
                print(f"警告: 路径不存在 {classpath}")
                continue

            anomaly_types = os.listdir(classpath)
            imgpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)

                # 跳过非目录的文件
                if not os.path.isdir(anomaly_path):
                    continue

                anomaly_files = sorted([
                    f for f in os.listdir(anomaly_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])

                # 跳过空目录
                if not anomaly_files:
                    print(f"警告: 目录为空 {anomaly_path}")
                    continue

                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, f) for f in anomaly_files
                ]

                # 训练/验证分割
                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                                                                     classname
                                                                 ][anomaly][train_val_split_idx:]

                # 为每个图像创建数据项 (classname, anomaly, image_path, None)
                for image_path in imgpaths_per_class[classname][anomaly]:
                    data_to_iterate.append([classname, anomaly, image_path, None])

        return imgpaths_per_class, data_to_iterate

    def _prepare_tiles_metadata(self):
        total = 0
        cumulative = []

        for data in self.data_to_iterate:
            image_path = data[2]
            try:
                with PIL.Image.open(image_path) as image:
                    width, height = image.size
            except FileNotFoundError:
                width, height = self.tile_size, self.tile_size

            bboxes = self._compute_tile_bboxes(width, height)
            self.tile_bboxes_cache[image_path] = bboxes

            tile_count = len(bboxes)
            self.tiles_per_image.append(tile_count)
            total += tile_count
            cumulative.append(total)

        self.total_tiles = total
        self.cumulative_tiles = cumulative

    def _compute_tile_bboxes(self, width, height):
        if width <= 0 or height <= 0:
            return [(0, 0, 0, 0)]

        step = max(1, self.tile_size - self.tile_overlap)

        x_starts = self._compute_axis_starts(width, step)
        y_starts = self._compute_axis_starts(height, step)

        bboxes = []
        for y in y_starts:
            for x in x_starts:
                right = min(x + self.tile_size, width)
                lower = min(y + self.tile_size, height)
                left = max(0, right - self.tile_size)
                upper = max(0, lower - self.tile_size)
                bboxes.append((left, upper, right, lower))

        return bboxes

    def _compute_axis_starts(self, length, step):
        starts = []
        pos = 0
        while pos + self.tile_size <= length:
            starts.append(pos)
            pos += step

        if not starts:
            starts.append(0)
        elif starts[-1] + self.tile_size < length:
            starts.append(max(0, length - self.tile_size))

        return sorted(set(starts))
