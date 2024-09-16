import datetime
import functools
from transformers import ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, Trainer, TrainingArguments, AutoTokenizer, AutoModel, TrainerCallback
import wandb

from utils.dataloader import OCRData, collate_fn

wandb.init(project="IAM_OCR", entity="mthomp12", name=str(datetime.datetime.now()).partition(".")[0])

config_encoder = ViTConfig(image_size=(128,1024))
decoder = AutoModel.from_pretrained("FacebookAi/roberta-base")
config_decoder = decoder.config
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)
model.config.decoder_start_token_id = 0
model.config.pad_token_id = 1

pretrained_state_dict = decoder.state_dict()
decoder_state_dict = model.decoder.state_dict()

train_ds = OCRData()
eval_ds = OCRData(eval=True)

for k,v in pretrained_state_dict.items():
    key = f"roberta.{k}"
    if "pooler" not in key:
        decoder_state_dict[key] = v

model.decoder.load_state_dict(decoder_state_dict)

training_arguments = TrainingArguments(
                                        output_dir="outputs",
                                        per_device_train_batch_size=16,
                                        per_device_eval_batch_size=5,
                                        logging_steps=10,
                                        bf16=True,
                                        eval_strategy='epoch',
                                        num_train_epochs = 10,
                                        save_strategy = 'epoch',
                                        save_total_limit = 3,
                                        load_best_model_at_end  = True,
                                        gradient_accumulation_steps = 1
                                        )
tokenizer = AutoTokenizer.from_pretrained("FacebookAi/roberta-base")
collator = functools.partial(collate_fn, tokenizer=tokenizer)

wandb.watch(model, log='all', log_freq=40)

class SampleCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs['model']
        batch = next(iter(kwargs['eval_dataloader']))
        tokens = model.generate(**batch)
        labels = batch['labels']
        tokenizer = kwargs['tokenizer']
        pred = tokenizer.batch_decode(tokens)
        actual = tokenizer.batch_decode(labels)
        delim = "-" * 50
        for p, a in zip(pred, actual):
            print(f"{delim}\npredicted: {p}\nactual: {a}\n{delim}\n\n")


sample_callback = [SampleCallback()]
trainer = Trainer( 
                    model = model, 
                    args = training_arguments,
                    train_dataset = train_ds,
                    eval_dataset= eval_ds,
                    data_collator = collator,
                    tokenizer = tokenizer,
                    callbacks = sample_callback
                    )

trainer.train()
