import json
import os
from typing import Iterable, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class COCORetrievalDataset(Dataset):
    def __init__(self, 
                 json_path="raw_datasets/ms-coco/dataset_coco.json", 
                 image_root="raw_datasets/ms-coco", 
                 split: Union[str, Iterable[str]] = "train", 
                 image_size=224, 
                 transform=None):
      
        with open(json_path, 'r') as f:
            data = json.load(f)

        if isinstance(split, str):
            target_splits = {split}
        else:
            target_splits = set(split)

        self.samples = []
        for entry in data["images"]:
            if entry.get("split", "train") not in target_splits:
                continue

            filepath = entry["filepath"]   
            filename = entry["filename"]  
            image_id = entry["imgid"]

            for sent in entry["sentences"]:
                self.samples.append({
                    "image": os.path.join(filepath, filename),
                    "text": sent["raw"].strip(),
                    "image_id": image_id,
                    "caption_id": sent["sentid"]
                })

        self.image_root = image_root
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_root, sample["image"]) 
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "text": sample["text"],
            "image_id": sample["image_id"],
            "caption_id": sample["caption_id"]
        }
