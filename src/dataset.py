import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial

class MedicalDataset(Dataset):
    def __init__(self,samples,transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return(len(self.samples))

    def __getitem__(self,index):
        sample = self.samples[index]
        return {
        "image": self.transform(sample["image"].convert("RGB")),
        "caption": sample["caption"],
        "image_id": sample["image_id"]
        }




def collate_fn(batch,tokenizer):
    images = [sample["image"] for sample in batch]
    captions = [sample["caption"] for sample in batch]
    image_ids = [sample["image_id"] for sample in batch]
    image_batch = torch.stack(images,dim = 0)
    encoded = tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
            )
    return{ "pixel_values": image_batch,
    "input_ids": encoded["input_ids"],
    "attention_mask": encoded["attention_mask"],
    "image_ids": image_ids}

def create_dataloader(dataset, tokenizer, batch_size=4, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
        )