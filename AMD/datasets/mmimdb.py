import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

def build_clip_transforms(image_size=224, train=False):
    if train:
        # 轻度增广：不破坏版面/文字
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),  # shorter side=256
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
            # 不建议强 ColorJitter 和水平翻转（海报文字会镜像）
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        # 与 CLIP eval 兼容：Resize→CenterCrop，不拉伸
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),
        ])

class MMIMDbCLIPDataset(Dataset):
 
    def __init__(self, hdf5_path='raw_datasets/mmimdb/mmimdb.hdf5',
                 image_size=224, split='train'):
        assert split in ('train', 'val', 'test')
        self.hdf5_path = hdf5_path
        self._h5 = None     
        self.split = split
        self.transform = build_clip_transforms(image_size=image_size, train=(split=='train'))


    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
            self.images = self._h5["images"]
            self.texts  = self._h5["texts"]
            self.labels = self._h5["genres"]
            self.idx = np.arange(len(self.images), dtype=np.int64)

    def __len__(self):
        self._ensure_open()
        return len(self.idx)

    def __getitem__(self, i):
        self._ensure_open()
        idx = int(self.idx[i])

        arr = self.images[idx].astype(np.uint8) 
        if arr.ndim == 3 and arr.shape[0] in (1,3):  
            arr = np.transpose(arr, (1, 2, 0))
        img = Image.fromarray(arr).convert("RGB")
        img = self.transform(img)
       
        raw = self.texts[idx]
        if isinstance(raw, (bytes, np.bytes_)):
            text = raw.decode("utf-8", errors="ignore")
        else:
            text = str(raw)

       
        lab = self.labels[idx]
        label = torch.tensor(lab, dtype=torch.float32)
       
        if label.ndim != 1:
            label = label.view(-1).float()

        sample = {"image": img, "text": text, "label": label}
        return sample

    def __del__(self):
        try:
            if getattr(self, "_h5", None) is not None:
                self._h5.close()
        except Exception:
            pass

    def compute_class_freq(self):
        self._ensure_open()
        y = self.labels[:]            # (N, C)
        pos = y.sum(axis=0).astype(np.float64)
        neg = y.shape[0] - pos
        return pos, neg
