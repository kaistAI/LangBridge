export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=2

python eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=meta-math/MetaMath-7B-V1.0 \
  --tasks msvamp_en,msvamp_es,msvamp_fr,msvamp_de,msvamp_ru,msvamp_zh,msvamp_ja,msvamp_th,msvamp_sw,msvamp_bn\
  --instruction_template metamath \
  --batch_size 1 \
  --output_path eval_outputs/msvamp/MetaMath-7B-V1.0  \
  --device cuda:0 \
  --no_cache