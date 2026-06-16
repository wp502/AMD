import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VQAv2Dataset(Dataset):
    def __init__(self,
                 json_path="raw_datasets/vqav2/vqa_merged_3129class.json",
                 image_root="raw_datasets/vqav2",
                 num_classes=3129,
                 image_size=224):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        img_rel = entry["image_path"]
        img_path = img_rel if os.path.isabs(img_rel) else os.path.join(self.image_root, img_rel)
        if not os.path.exists(img_path):
           
            alt = os.path.join(self.image_root, os.path.basename(img_rel))
            if os.path.exists(alt):
                img_path = alt
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        text = entry["question"]

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        soft_label = torch.zeros(self.num_classes, dtype=torch.float32)

        for cid in entry.get("labels", []):
            cid = int(cid)
            if 0 <= cid < self.num_classes:
                label[cid] = 1.0
                soft_label[cid] = 1.0  

        return {
            "image": image,           
            "text": text,             
            "label": label,           
            "soft_label": soft_label  
        }
