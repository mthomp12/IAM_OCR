import datetime
import functools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTConfig, RobertaConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTModel, Trainer, TrainingArguments, AutoTokenizer, AutoModel
import wandb

from utils.dataloader import OCRData, collate_fn

wandb.init(project="IAM_OCR", entity="mthomp12", name=str(datetime.datetime.now()).partition(".")[0])

config_encoder = ViTConfig(image_size=(128,500), num_hidden_layers=1)
decoder = AutoModel.from_pretrained("smallbenchnlp/roberta-small")
config_decoder = decoder.config
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)
model.config.decoder_start_token_id = 0
model.config.pad_token_id = 1

pretrained_state_dict = decoder.state_dict()
decoder_state_dict = model.decoder.state_dict()

ds = OCRData()

for k,v in pretrained_state_dict.items():
    key = f"roberta.{k}"
    if "pooler" not in key:
        decoder_state_dict[key] = v

model.decoder.load_state_dict(decoder_state_dict)

training_arguments = TrainingArguments(
                                        output_dir="outputs",
                                        per_device_train_batch_size=1,
                                        logging_steps=10,
                                        bf16=False
                                        )
tokenizer = AutoTokenizer.from_pretrained("FacebookAi/roberta-base")
collator = functools.partial(collate_fn, tokenizer=tokenizer)

wandb.watch(model, log='all', log_freq=400)
trainer = Trainer( 
                    model = model, 
                    args = training_arguments,
                    train_dataset = ds,
                    data_collator = collator,
                    eval_dataset = None,
                    tokenizer = tokenizer
                    )

trainer.train()