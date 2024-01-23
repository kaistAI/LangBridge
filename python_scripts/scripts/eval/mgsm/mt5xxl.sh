export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=1

python eval_baseline.py \
  --model hf-seq2seq \
  --model_args dtype="bfloat16",pretrained=DKYoon/mt5-xxl-lm-adapt \
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --num_fewshot 8\
  --batch_size 1 \
  --output_path eval_outputs/mgsm/mt5-xxl-lm-adapt  \
  --device cuda:0 \
  --no_cache