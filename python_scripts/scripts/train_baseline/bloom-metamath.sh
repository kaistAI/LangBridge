export OMP_NUM_THREADS=8
export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=4

ARGS="
--n_gpu $NUM_GPU
--strategy deepspeed_stage_2
--output_dir checkpoints/bloom7b-metamath
--run_name bloom7b-metamath
--seed 42
--train_set_path DKYoon/metamath-200k
--output_exists True
--lm_name_or_path bigscience/bloom-7b1
--max_length 1152
--learning_rate 2e-5
--warmup_steps 0
--per_device_train_batch_size 4
--per_device_eval_batch_size 16
--gradient_accumulation_steps 8
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 
"

echo $ARGS
if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python train_baseline.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node train_baseline.py $ARGS
fi