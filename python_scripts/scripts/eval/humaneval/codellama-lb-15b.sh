export CUDA_VISIBLE_DEVICES=3
export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets


accelerate launch eval_code.py \
  --model kaist-ai/codellama-langbridge-15b \
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --trust_remote_code \
  --tasks humaneval#en,humaneval#en_anon,humaneval#sw,humaneval#sw_anon,humaneval#bn,humaneval#bn_anon,humaneval#mr,humaneval#mr_anon,humaneval#ne,humaneval#ne_anon,humaneval#pa,humaneval#pa_anon,humaneval#te,humaneval#te_anon,humaneval#ur,humaneval#ur_anon \
  --load_data_path ../data/humaneval/humaneval_english.json,../data/humaneval/humaneval_english_anon.json,../data/humaneval/humaneval_swahili.json,../data/humaneval/humaneval_swahili_anon.json,../data/humaneval/humaneval_bengali.json,../data/humaneval/humaneval_bengali_anon.json,../data/humaneval/humaneval_marathi.json,../data/humaneval/humaneval_marathi_anon.json,../data/humaneval/humaneval_nepali.json,../data/humaneval/humaneval_nepali_anon.json,../data/humaneval/humaneval_punjabi.json,../data/humaneval/humaneval_punjabi_anon.json,../data/humaneval/humaneval_telugu.json,../data/humaneval/humaneval_telugu_anon.json,../data/humaneval/humaneval_urdu.json,../data/humaneval/humaneval_urdu_anon.json \
  --do_sample False \
  --n_samples 1 \
  --batch_size 1 \
  --allow_code_execution \
  --modeltype langbridge \
  --precision bf16 \
  --save_generations \
  --metric_output_path eval_outputs/humaneval/codellama-langbridge-15b/results.json \