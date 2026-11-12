set -e

solver_model_path=$1
questioner_model_path=$2
experiment_name=$3
solver_train_data="${STORAGE_PATH:?请先设置 STORAGE_PATH}/generated_question/${experiment_name}_train.parquet"

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=0 VLLM_DISABLE_COMPILE_CACHE=1

echo 'start generate question'
generation_pids=()
for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu python question_generate/question_generate.py \
        --model "$questioner_model_path" --suffix "$gpu" --num_samples 1000 --save_name "$experiment_name" &
    generation_pids+=("$!")
done
generation_status=0
for pid in "${generation_pids[@]}"; do
    wait "$pid" || generation_status=1
done
(( generation_status == 0 )) || exit 1

echo 'start evaluate generated question'
scoring_pids=()
for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu python question_evaluate/evaluate.py \
        --model "$solver_model_path" --suffix "$gpu" --save_name "$experiment_name" &
    scoring_pids+=("$!")
done
scoring_status=0
for pid in "${scoring_pids[@]}"; do
    wait "$pid" || scoring_status=1
done
(( scoring_status == 0 )) || exit 1
echo 'save Solver training data locally'
python question_evaluate/upload.py --local_output "$solver_train_data" --num_shards 2 \
    --max_score 0.8 --min_score 0.3 --experiment_name "$experiment_name" || exit 1
echo 'start train'

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    trainer.n_gpus_per_node=2 \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.40 \
    worker.rollout.enforce_eager=true \
    data.rollout_batch_size=16 \
    data.val_batch_size=16 \
    worker.actor.global_batch_size=8 \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    "data.train_files=$solver_train_data" \
    trainer.total_epochs=100 \
    trainer.max_steps=20 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    trainer.val_freq=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor

sleep 10

echo "solver training finished"

# 原版外部评测含四卡任务，导出 Solver 模型后结束。
