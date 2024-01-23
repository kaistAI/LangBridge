export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=2

python eval_baseline.py \
  --model hf-causal-experimental \
  --model_args dtype="bfloat16",pretrained=meta-math/MetaMath-7B-V1.0 \
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --instruction_template metamath \
  --batch_size 1 \
  --output_path eval_outputs/mgsm/MetaMath-7B-V1.0  \
  --device cuda:0 \
  --no_cache