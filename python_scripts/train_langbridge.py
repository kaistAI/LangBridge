import os
from dataclasses import asdict, dataclass, field
import logging
import string
import random


import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchinfo import summary

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from deepspeed.ops.adam import FusedAdam

from transformers import HfArgumentParser, AutoTokenizer
from transformers.utils import logging as hf_logging

from langbridge import LangBridgeModel, LangBridgeConfig
from dataset import Data

torch.set_float32_matmul_precision('medium')
logger = logging.getLogger(__name__)
logging.getLogger("lightning.pytorch").setLevel(logging.INFO)
hf_logging.set_verbosity_error()


class AlignLBModule(LightningModule):
    def __init__(self, model, enc_tokenizer, lm_tokenizer, args):
        super().__init__()
        self.model: LangBridgeModel = model
        self.enc_tokenizer = enc_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.args = args
        self.save_hyperparameters(asdict(args))
        self.sync_dist = True if self.args.n_gpu > 1 else False

        self.total_max_length = self.args.max_length + self.args.max_length_enc

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True,
                 prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs[0]
        self.log(f'validation_loss_{dataloader_idx}', loss,
                 prog_bar=True, logger=True, sync_dist=self.sync_dist, add_dataloader_idx=False)

    def configure_optimizers(self):
        alignment, enc, lm = [], [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'alignment' in name:
                    alignment.append(param)
                elif 'enc' in name:
                    enc.append(param)
                elif 'lm' in name:
                    lm.append(param)
                else:
                    raise ValueError('unknown parameter')
        params = [
            {'params': alignment,
                'lr': self.args.learning_rate_alignment, 'weight_decay': self.args.w_decay_alignment},
            {'params': enc, 'lr': self.args.learning_rate_enc,
                'weight_decay': self.args.w_decay_enc},
            {'params': lm, 'lr': self.args.learning_rate_lm,
                'weight_decay': self.args.w_decay_lm},
        ]
        if 'deepspeed' in self.args.strategy:
            optimizer = FusedAdam(params)
        else:
            # using gradient clipping is not compatible with PyTorch fused adamw
            optimizer = AdamW(params,
                              fused=False if self.args.gradient_clip_val > 0 else True)
        return [optimizer]

    def on_fit_end(self):
        if self.global_rank == 0 and self.global_step > 0:
            save_name = f'epoch={self.current_epoch}-step={self.global_step}'
            self.model.save_pretrained(
                f'{self.args.output_dir}/{save_name}', safe_serialization=False)

    def collate_fn(self, batch, use_dynamic=False):
        if use_dynamic:
            enc_length, max_length = self.adjust_enc_length()
        else:
            enc_length, max_length = self.args.max_length_enc, self.args.max_length

        no_output_data = [d['input']
                          for d in batch if not d['output']]  # unlabeled data

        suffix = []
        enc_tokens = self.enc_tokenizer(
            no_output_data, padding=True, add_special_tokens=False, return_offsets_mapping=True, return_tensors='pt')
        split_index = enc_length - 1

        enc_input_ids = enc_tokens['input_ids'][:, :split_index]
        enc_attention_mask = enc_tokens['attention_mask'][:, :split_index]

        # cat eos token
        enc_input_ids = torch.cat(
            (enc_input_ids, torch.tensor([[self.enc_tokenizer.eos_token_id]] * len(no_output_data), dtype=torch.long)), dim=1)
        enc_attention_mask = torch.cat(
            (enc_attention_mask, torch.ones(len(no_output_data), 1, dtype=torch.long)), dim=1)

        # get offsets for each split_index
        offsets = enc_tokens['offset_mapping'][:, split_index][:, 0]
        # reconstruct suffix i.e. the labels
        try:
            suffix = suffix + [no_output_data[i][offsets[i]:]
                               for i in range(len(no_output_data))]
        except IndexError:
            print(len(no_output_data), offsets.shape)
            raise IndexError

        lm_tokens = self.lm_tokenizer(
            suffix, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        labels = lm_tokens['input_ids'].clone().detach()
        labels[labels == self.lm_tokenizer.pad_token_id] = -100

        return {
            'enc_ids': enc_input_ids,
            'enc_mask': enc_attention_mask,
            'input_ids': lm_tokens['input_ids'],
            'attention_mask': lm_tokens['attention_mask'],
            'labels': labels,
        }

    def collate_fn_output_exists(self, batch):
        inputs = [d['input'] for d in batch]
        outputs = [d['output'] for d in batch]

        enc_tokens = self.enc_tokenizer(
            inputs, padding=True, truncation=True, max_length=self.args.max_length_enc, return_tensors='pt')
        actual_enc_length = enc_tokens['input_ids'].shape[1]
        lm_max_length = self.total_max_length - actual_enc_length

        lm_tokens = self.lm_tokenizer(
            outputs, padding=True, truncation=True, max_length=lm_max_length, return_tensors='pt')
        labels = lm_tokens['input_ids'].clone().detach()
        labels[labels == self.lm_tokenizer.pad_token_id] = -100

        return {
            'enc_ids': enc_tokens['input_ids'],
            'enc_mask': enc_tokens['attention_mask'],
            'input_ids': lm_tokens['input_ids'],
            'attention_mask': lm_tokens['attention_mask'],
            'labels': labels,
        }

    def adjust_enc_length(self):
        # randomness is consistent across devices
        enc_length = random.randint(32, self.args.max_length_enc)

        possible_max_length = self.total_max_length - enc_length
        if possible_max_length > self.args.max_length:
            max_length = random.randint(
                self.args.max_length, possible_max_length)
        else:
            max_length = self.args.max_length

        return enc_length, max_length

    def train_dataloader(self):
        train_dataset = Data(
            self.args.train_set_path, split='train')

        if self.args.output_exists:  # labeled finetuning data
            def collate_fn(batch): return self.collate_fn_output_exists(
                batch)
        else:  # unlabeled pretraining data
            def collate_fn(batch): return self.collate_fn(
                batch, use_dynamic=self.args.use_dynamic_enc_length)
        dataloader = DataLoader(train_dataset, batch_size=self.args.per_device_train_batch_size,
                                shuffle=True, num_workers=self.args.dataloader_num_workers, collate_fn=collate_fn)

        return dataloader

    def val_dataloader(self):
        val_sets = self.args.val_set_path.split(',')
        dataloaders = []
        def collate_fn(batch): return self.collate_fn(batch, use_dynamic=False)

        for val_set in val_sets:
            dataset = Data(
                val_set, split='validation')
            dataloader = DataLoader(dataset, batch_size=self.args.per_device_eval_batch_size,
                                    shuffle=False, num_workers=self.args.dataloader_num_workers, collate_fn=collate_fn)
            dataloaders.append(dataloader)
        return dataloaders


@dataclass
class LBTrainingArguments:
    """ custom arguments """
    n_gpu: int = field(default=1)
    strategy: str = field(default='auto')
    eval_only: bool = field(default=False)
    hf_checkpoint_path: str = field(default=None)

    enc_name_or_path: str = field(default='DKYoon/mt5-small-lm-adapt')
    lm_name_or_path: str = field(default='facebook/opt-125m')
    alignments: str = field(default='linear')
    add_new_lines_to_enc: bool = field(default=True)

    enc_hidden_size: int = field(default=512)
    lm_hidden_size: int = field(default=768)

    train_set_path: str = field(
        default='DKYoon/metamath-200k')
    val_set_path: str = field(
        default=None)
    limit_val_samples: int = field(default=None)
    output_exists: bool = field(default=False)

    max_length: int = field(default=256)
    max_length_enc: int = field(default=128)
    use_dynamic_enc_length: bool = field(default=True)

    freeze_language_model: bool = field(default=True)
    freeze_encoder: bool = field(default=True)

    # redefine some HF arguments
    seed: int = field(default=42)
    run_name: str = field(default='debug')
    output_dir: str = field(default='./checkpoints')
    save_total_limit: int = field(default=5)
    save_interval: float = field(default=1.0)
    check_val_every_n_epoch: int = field(default=1)
    logging_steps: int = field(default=10)
    dataloader_num_workers: int = field(default=8)

    num_train_epochs: int = field(default=5)
    learning_rate_alignment: float = field(default=6e-4)
    learning_rate_enc: float = field(default=2e-5)
    learning_rate_lm: float = field(default=2e-5)

    w_decay_alignment: float = field(default=0)
    w_decay_enc: float = field(default=0.1)
    w_decay_lm: float = field(default=0)

    warmup_steps: int = field(default=0)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1)
    gradient_clip_val: float = field(default=1.0)
    bf16: bool = field(default=True)


if __name__ == '__main__':
    parser = HfArgumentParser(LBTrainingArguments)
    training_args: LBTrainingArguments
    training_args = parser.parse_args_into_dataclasses()[0]

    if os.path.isdir(training_args.output_dir):
        raise OSError(
            f"Directory '{training_args.output_dir}' already exists.")

    if training_args.alignments not in ['linear', 'ffn'] and training_args.num_latents == -1:
        raise ValueError(
            'num_latents must be specified when using non-linear alignments')

    seed_everything(training_args.seed)

    logging.basicConfig(
        format=f'%(asctime)s {training_args.run_name} %(message)s',
        datefmt='%H:%M:%S',
        force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ]
    )

    if training_args.output_exists != training_args.freeze_encoder:
        logger.warning(
            'output_exists and freeze_encoder should be the same value, see section D.1')

    # this must be a FastTokenizer to use dynamic length
    enc_tokenizer = AutoTokenizer.from_pretrained(
        training_args.enc_name_or_path, use_fast=True)
    try:
        lm_tokenizer = AutoTokenizer.from_pretrained(
            training_args.lm_name_or_path, use_fast=False)
    except:
        lm_tokenizer = AutoTokenizer.from_pretrained(
            training_args.lm_name_or_path)

    lm_tokenizer.padding_side = 'right'

    if not enc_tokenizer.pad_token:
        enc_tokenizer.pad_token = enc_tokenizer.eos_token
    if not lm_tokenizer.pad_token:
        lm_tokenizer.pad_token = lm_tokenizer.eos_token

    logger.info('loading model...')

    config = LangBridgeConfig(
        enc=training_args.enc_name_or_path,
        lm=training_args.lm_name_or_path,
        alignments=training_args.alignments,
        dim_enc=training_args.enc_hidden_size,
        dim_lm=training_args.lm_hidden_size,
        freeze_language_model=training_args.freeze_language_model,
        freeze_encoder=training_args.freeze_encoder,
    )

    model_class = LangBridgeModel

    if training_args.hf_checkpoint_path:
        logger.info('loading from HF checkpoint...')

        model = model_class.from_pretrained(
            training_args.hf_checkpoint_path, config=config)
    else:
        if training_args.eval_only:
            logger.warning('Evaluating an un-aligned model!')

        model = model_class(config)

    # this is true for all our experiments, explained in section D.1
    if training_args.add_new_lines_to_enc:
        logger.info('Adding whitespaces to encoder tokenizer')
        whitespace = list(string.whitespace)[1:]  # exclude single space
        whitespace = whitespace + ['  ', '   ', '    ']  # add multispace
        enc_tokenizer.add_special_tokens(
            {'additional_special_tokens': whitespace})

        if training_args.freeze_encoder:
            model.lb.enc.get_input_embeddings().weight.requires_grad = True
            logger.warning(
                'Unfreezing encoder embedding layer since new tokens were added')

    summary(model)

    if not training_args.eval_only:
        model.train()
    pl_model = AlignLBModule(model, enc_tokenizer,
                             lm_tokenizer, training_args)

    wandb_logger = WandbLogger(
        project='langbridge',
        name=training_args.run_name)

    trainer = Trainer(
        accelerator='gpu',
        strategy=training_args.strategy,
        devices=training_args.n_gpu,
        max_epochs=int(training_args.num_train_epochs),
        precision='bf16-mixed' if training_args.bf16 else 32,
        num_sanity_val_steps=0,
        log_every_n_steps=training_args.logging_steps,
        logger=wandb_logger,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        gradient_clip_val=training_args.gradient_clip_val,
        enable_checkpointing=False,
        limit_val_batches=0.0 if training_args.val_set_path is None else 1.0,
    )

    if not training_args.eval_only:
        if training_args.val_set_path is not None:
            trainer.validate(pl_model)
        trainer.fit(pl_model)
    else:
        trainer.validate(pl_model)
