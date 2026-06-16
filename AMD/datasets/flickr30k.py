import json
import os
from functools import lru_cache
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

class Flickr30kRetrievalDataset(Dataset):
    """
    Flickr30k retrieval dataset with:
      - Proper CLIP-like preprocessing (train vs eval)
      - Optional one-caption-per-image sampling for training
      - Lightweight image cache to reduce IO
    """
    def __init__(
        self,
        json_path="raw_datasets/flickr-30k/dataset_flickr30k.json",
        image_root="raw_datasets/flickr-30k/images",
        split="train",
        image_size=224,
        mode="train",  # "train" or "eval"
        one_caption_per_image=False,  # recommended True for training
        transform=None,
        tokenize_fn=None,  # e.g., CLIP tokenizer; if None, return raw text
        max_text_len=77,   # CLIP default context length
    ):
        assert mode in ["train", "eval"]
        self.mode = mode
        self.one_caption_per_image = one_caption_per_image and (mode == "train")
        self.tokenize_fn = tokenize_fn
        self.max_text_len = max_text_len

        with open(json_path, "r") as f:
            data = json.load(f)

        # Build per-image records
        images = []
        for entry in data["images"]:
            if entry["split"] != split:
                continue
            img_path = os.path.join(image_root, entry["filename"])
            img_id = entry["imgid"]
            caps = [s["raw"] for s in entry["sentences"]]
            cap_ids = [s["sentid"] for s in entry["sentences"]]
            images.append({
                "image": img_path,
                "image_id": img_id,
                "captions": caps,
                "caption_ids": cap_ids
            })

        self.image_records = images

        # Flatten samples according to mode
        self.samples = []
        if self.one_caption_per_image:
            # One caption per image (chosen lazily in __getitem__ via random)
            # We still keep per-image records; length equals number of images
            self.samples = list(range(len(self.image_records)))  # indices into image_records
        else:
            # Use all (image, caption) pairs (good for eval; OK for train but weaker negatives)
            for i, rec in enumerate(self.image_records):
                for cap, cid in zip(rec["captions"], rec["caption_ids"]):
                    self.samples.append((i, cap, cid))

        # Transforms
        if transform is not None:
            self.transform = transform
        else:
            if mode == "train":
                # CLIP-like train pipeline: random resized crop + small jitter (可按需去掉 jitter)
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.9, 1.0),
                        ratio=(3/4, 4/3),
                        interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711)),
                ])
            else:
                # Eval pipeline: keep aspect ratio, center crop
                self.transform = transforms.Compose([
                    transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711)),
                ])

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=8192)
    def _load_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        return img

    def _sample_caption_for_image(self, rec):
        # Randomly pick one caption for training
        idx = torch.randint(0, len(rec["captions"]), (1,)).item()
        return rec["captions"][idx], rec["caption_ids"][idx]

    def _maybe_tokenize(self, text):
        if self.tokenize_fn is None:
            return text
        # Expect tokenize_fn to handle truncation/padding to max_text_len if needed
        return self.tokenize_fn(text, max_len=self.max_text_len)

    def __getitem__(self, idx):
        if self.one_caption_per_image:
            rec = self.image_records[self.samples[idx]]
            text, cap_id = self._sample_caption_for_image(rec) if self.mode == "train" else (rec["captions"][0], rec["caption_ids"][0])
        else:
            img_index, text, cap_id = self.samples[idx]
            rec = self.image_records[img_index]

        image = self._load_image(rec["image"])
        image = self.transform(image)

        item = {
            "image": image,
            "text": self._maybe_tokenize(text),
            "image_id": rec["image_id"],
            "caption_id": cap_id,
            
        }
        return item
