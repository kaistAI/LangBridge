import os  # noqa

os.environ['TRANSFORMERS_CACHE'] = '/mnt/sda/dongkeun/huggingface'  # noqa
os.environ['HF_DATASETS_CACHE'] = '/mnt/sda/dongkeun/huggingface'  # noqa


from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from langbridge import LangBridgeModel
import torch
import numpy as np


if __name__ == '__main__':
    DEVICE = 'cuda:3'
    LANGS = ['deu']

    model = LangBridgeModel.from_pretrained('kaist-ai/orca2-langbridge-9b')
    model.eval()
    model.to(DEVICE)

    enc_tokenizer = AutoTokenizer.from_pretrained(
        'kaist-ai/langbridge_encoder_tokenizer')
    lm_tokenizer = AutoTokenizer.from_pretrained(
        'kaist-ai/orca2-langbridge-9b')

    orca_prompt = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    system_message = ''

    for LANG in LANGS:
        ds = load_dataset('gsarti/flores_101', LANG)['dev']

        all_embs = []
        for example in tqdm(ds):
            sentence = example['sentence']

            text = orca_prompt.format(
                system_message=system_message, user_message=sentence)

            tokens = enc_tokenizer(text, return_tensors='pt').to(DEVICE)
            enc_input_ids = tokens['input_ids']
            enc_input_mask = tokens['attention_mask']

            input_ids = torch.LongTensor([lm_tokenizer.bos_token_id])
            input_ids = input_ids.repeat(
                enc_input_ids.shape[0], 1).to(DEVICE)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                # emb = model.get_input_embeddings()(enc_input_ids).squeeze()
                emb = model(
                    enc_input_ids, enc_input_mask, input_ids, attention_mask, output_hidden_states=True).hidden_states[-1].squeeze()
            mean = torch.mean(emb, dim=0)
            all_embs.append(mean)

        all_embs_tensor = torch.stack(all_embs, dim=0)

        # cast to float32 then to numpy
        all_embs_tensor = all_embs_tensor.float().cpu().numpy()

        print(all_embs_tensor.shape)

        np.save(
            f'embeddings/lb_flores_lm_{LANG}.npy', all_embs_tensor)
