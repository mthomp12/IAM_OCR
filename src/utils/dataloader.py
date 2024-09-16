import json
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import torch.nn.functional as F

class OCRData:
    def __init__(self, root="IAM_OCR/data", eval=False):
        self.root = root
        meta_filepath=f"{self.root}/gt_test.txt"
        self.df = self.get_data(meta_filepath, eval)
        self.eval=eval
    
    @staticmethod
    def get_data(meta_filepath, eval):
        with open(meta_filepath, "r") as f:
            x = f.read()
        df = pd.DataFrame([xx.split("\t") for xx in x.splitlines()], columns=['file_path','text'])
        train_records = int(len(df) * 0.8)
        eval_records = len(df) - train_records
        if eval:
            df = df.tail(eval_records).reset_index(drop=True)
        else:
            df  = df.head(train_records)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        file_path = f"{self.root}/image/{row['file_path']}"
        img = cv2.imread(file_path)
        img = cv2.resize(img, (1024,128), interpolation=cv2.INTER_CUBIC)
        img = img / 255
        img = torch.tensor(img).to(torch.float32)
        label = row['text']
        self.df.loc[idx, ['s0','s1','s2']] = list(img.shape)
        self.df.loc[idx, ['dtype']] = img.dtype

        return {"pixel_values": img, "labels": label}

def collate_fn(x, tokenizer):
    # pad pixel_values to max width
    pixel_values = [xx.pop('pixel_values') for xx in x]
    pixel_values = [img.permute(2,0,1) for img in pixel_values] #h,w,c -> c,h,w
    pixel_values = torch.stack(pixel_values)

    x = torch.utils.data.default_collate(x)
    x['pixel_values'] = pixel_values
    labels = x.pop('labels')
    tokens = tokenizer(labels, return_tensors='pt', padding=True, truncation=True)
    x['labels'] = tokens['input_ids']
    x['decoder_attention_mask'] = tokens['attention_mask']

    return x

if __name__ == '__main__':
    import functools
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    ds = OCRData()
    tokenizer = AutoTokenizer.from_pretrained("FacebookAi/roberta-base")
    collator = functools.partial(collate_fn, tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=32, collate_fn=collator)
    for batch in tqdm(dl):
        pass

    df = dl.dataset.df
