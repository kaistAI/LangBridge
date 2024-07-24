export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=0

python eval_langbridge.py \
  --checkpoint_path kaist-ai/orca2-langbridge-9b \
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --tasks copa,xcopa_et,xcopa_ht,xcopa_it,xcopa_id,xcopa_qu,xcopa_sw,xcopa_zh,xcopa_ta,xcopa_th,xcopa_tr,xcopa_vi \
  --instruction_template orca \
  --batch_size 32 \
  --output_path eval_outputs/xcopa/orca2-langbrige_9b \
  --device cuda:0 \
  --no_cache \