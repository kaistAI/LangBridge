import os  # noqa

os.environ['TRANSFORMERS_CACHE'] = '/mnt/sda/dongkeun/huggingface'  # noqa
os.environ['HF_DATASETS_CACHE'] = '/mnt/sda/dongkeun/huggingface'  # noqa


from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


if __name__ == '__main__':
    DEVICE = 'cuda:1'
    LANGS = ['deu']

    model = AutoModelForCausalLM.from_pretrained('microsoft/Orca-2-7b')
    model.eval()
    model.to(DEVICE)

    lm_tokenizer = AutoTokenizer.from_pretrained(
        'microsoft/Orca-2-7b', use_fast=False)

    orca_prompt = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    system_message = ''

    for LANG in LANGS:
        ds = load_dataset('gsarti/flores_101', LANG)['dev']

        all_embs = []
        for example in tqdm(ds):
            sentence = example['sentence']

            text = orca_prompt.format(
                system_message=system_message, user_message=sentence)

            tokens = lm_tokenizer(text, return_tensors='pt').to(DEVICE)
            enc_input_ids = tokens['input_ids']

            with torch.no_grad():
                # emb = model.get_input_embeddings()(enc_input_ids).squeeze()
                emb = model(
                    enc_input_ids, output_hidden_states=True).hidden_states[-1].squeeze()
            mean = torch.mean(emb, dim=0)
            all_embs.append(mean)

        all_embs_tensor = torch.stack(all_embs, dim=0)

        # cast to float32 then to numpy
        all_embs_tensor = all_embs_tensor.float().cpu().numpy()

        print(all_embs_tensor.shape)

        np.save(
            f'embeddings/baseline_flores_lm_{LANG}.npy', all_embs_tensor)
