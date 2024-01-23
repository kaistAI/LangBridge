export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=2

python eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=Mathoctopus/Parallel_xRFT_7B \
  --tasks msvamp_en,msvamp_es,msvamp_fr,msvamp_de,msvamp_ru,msvamp_zh,msvamp_ja,msvamp_th,msvamp_sw,msvamp_bn\
  --instruction_template mathoctopus \
  --batch_size 1 \
  --output_path eval_outputs/msvamp/mathoctopus-7b-xrft  \
  --device cuda:0 \
  --no_cache