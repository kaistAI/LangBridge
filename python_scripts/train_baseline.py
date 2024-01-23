
import os
from dataclasses import asdict, dataclass, field
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchinfo import summary

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from deepspeed.ops.adam import FusedAdam

from transformers import HfArgumentParser, AutoTokenizer, AutoModelForCausalLM
from transformers import get_constant_schedule_with_warmup
from transformers.utils import logging as hf_logging

from dataset import Data

torch.set_float32_matmul_precision('medium')
logger = logging.getLogger(__name__)
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
hf_logging.set_verbosity_error()


class PretrainBaselineModule(LightningModule):
    def __init__(self, model, lm_tokenizer, args):
        super().__init__()
        self.model = model
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})

        self.lm_tokenizer = lm_tokenizer
        self.args = args
        self.save_hyperparameters(asdict(args))
        self.sync_dist = True if self.args.n_gpu > 1 else False

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels']
                       )

        loss = outputs[0]
        self.log('train_loss', loss, on_step=True,
                 prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('validation_loss', loss,
                 prog_bar=True, logger=True, sync_dist=self.sync_dist)

    def configure_optimizers(self):
        params = self.model.parameters()
        if 'deepspeed' in self.args.strategy:
            optimizer = FusedAdam(
                params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            # using gradient clipping is not compatible with PyTorch fused adamw
            optimizer = AdamW(
                params, weight_decay=self.args.weight_decay, lr=self.args.learning_rate, fused=True)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, self.args.warmup_steps)
        return [optimizer], [scheduler]

    def on_fit_end(self):
        if self.global_rank == 0 and self.global_step > 0:
            save_name = f'epoch={self.current_epoch}-step={self.global_step}'
            self.model.save_pretrained(
                f'{self.args.output_dir}/{save_name}', safe_serialization=False)

    def collate_fn(self, batch):
        inputs = [d['input'] for d in batch]

        tokens = self.lm_tokenizer(
            inputs, return_tensors='pt', padding=True, truncation=True, max_length=self.args.max_length)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        labels = tokens['input_ids'].clone().detach()
        labels[labels == self.lm_tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def collate_fn_output_exists(self, batch):
        inputs = [d['input'] for d in batch]
        outputs = [d['output'] for d in batch]

        inputs_ids = [self.lm_tokenizer(d, return_tensors='pt')[
            'input_ids'][0] for d in inputs]
        target_ids = [self.lm_tokenizer(d, return_tensors='pt')[
            'input_ids'][0] for d in outputs]

        # no loss on the input ids
        input_labels = [torch.ones_like(d) * -100 for d in inputs_ids]

        # for target copy the target ids
        target_labels = [d.clone().detach() for d in target_ids]

        ids, labels, masks = [], [], []
        # zip inputs and targets and pad to max length
        for i in range(len(inputs_ids)):
            ids.append(torch.cat((inputs_ids[i], target_ids[i]), dim=0))
            labels.append(
                torch.cat((input_labels[i], target_labels[i]), dim=0))
            masks.append(torch.ones_like(ids[i]))

            assert ids[i].shape == labels[i].shape == masks[i].shape

        # pad to max length
        ids = torch.nn.utils.rnn.pad_sequence(
            ids, batch_first=True, padding_value=self.lm_tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100)
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0)

        # truncate to max length
        ids = ids[:, :self.args.max_length]
        labels = labels[:, :self.args.max_length]
        masks = masks[:, :self.args.max_length]

        assert ids.shape == labels.shape == masks.shape

        return {
            'input_ids': ids,
            'attention_mask': masks,
            'labels': labels,
        }

    def train_dataloader(self):
        train_dataset = Data(
            self.args.train_set_path, split='train')

        if self.args.output_exists:  # labeled finetuning data
            def collate_fn(batch): return self.collate_fn_output_exists(
                batch)
        else:  # unlabeled pretraining data
            def collate_fn(batch): return self.collate_fn(batch)
        dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size,
                                shuffle=True, num_workers=self.args.dataloader_num_workers, collate_fn=collate_fn)

        return dataloader


@dataclass
class TrainingArguments:
    """ custom arguments """
    n_gpu: int = field(default=1)
    strategy: str = field(default='auto')
    debug: bool = field(default=False)
    eval_only: bool = field(default=False)
    hf_checkpoint_path: str = field(default=None)

    lm_name_or_path: str = field(default='bigscience/bloom-560m')

    train_set_path: str = field(
        default='../data/SlimPajama-6B/train-256.jsonl')

    val_set_path: str = field(
        default=None)
    limit_val_samples: int = field(default=None)
    output_exists: bool = field(default=False)

    max_length: int = field(default=256)

    # redefine some HF arguments
    seed: int = field(default=42)
    run_name: str = field(default='finetune')
    output_dir: str = field(default='./checkpoints')
    save_total_limit: int = field(default=5)
    save_interval: float = field(default=1.0)
    check_val_every_n_epoch: int = field(default=1)
    logging_steps: int = field(default=10)
    dataloader_num_workers: int = field(default=8)

    num_train_epochs: int = field(default=1)
    weight_decay: float = field(default=0.1)
    learning_rate: float = field(default=5e-5)
    warmup_steps: int = field(default=0)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1)
    bf16: bool = field(default=True)


if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    training_args: TrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    if os.path.isdir(training_args.output_dir):
        raise OSError(
            f"Directory '{training_args.output_dir}' already exists.")

    # reseed if loading from checkpoint
    seed = training_args.seed
    seed_everything(training_args.seed)

    # add latent token length to max_length

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s',
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ]
    )

    lm_tokenizer = AutoTokenizer.from_pretrained(
        training_args.lm_name_or_path, use_fast=True)

    logger.info('loading model...')

    lm_path = training_args.hf_checkpoint_path if training_args.hf_checkpoint_path else training_args.lm_name_or_path

    model = AutoModelForCausalLM.from_pretrained(lm_path)

    summary(model)

    pl_model = PretrainBaselineModule(model, lm_tokenizer, training_args)

    wandb_logger = WandbLogger(
        project='langbridge',
        name=training_args.run_name)

    strategy = training_args.strategy

    trainer = Trainer(
        accelerator='gpu',
        strategy=strategy,
        enable_checkpointing=False,
        devices=training_args.n_gpu,
        max_epochs=int(training_args.num_train_epochs),
        precision='bf16-mixed' if training_args.bf16 else 32,
        num_sanity_val_steps=0,
        log_every_n_steps=training_args.logging_steps,
        logger=wandb_logger,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        limit_val_batches=0.0
    )

    if not training_args.eval_only:
        trainer.validate(pl_model)
        trainer.fit(pl_model)
    else:
        trainer.validate(pl_model)
