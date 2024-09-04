export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=1

python eval_langbridge.py \
  --checkpoint_path kaist-ai/metamath-langbridge-9b \
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --tasks mgsm_en\
  --instruction_template metamath \
  --batch_size 1 \
  --output_path eval_outputs/mgsm/metamath-langbridge_9b \
  --device cuda:0 \
  --no_cache


