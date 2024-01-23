export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=3

python eval_langbridge.py \
  --checkpoint_path kaist-ai/llemma-langbrige_9b\
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --num_fewshot 8\
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --batch_size 1 \
  --output_path eval_outputs/mgsm/llemma-langbrige_9b \
  --device cuda:0 \
  --no_cache


