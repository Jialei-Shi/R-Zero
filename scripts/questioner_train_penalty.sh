#!/bin/bash
set -e

solver_model_path=$1        # 打分用，上一轮迭代后的solver
questioner_model_path=$2    # 进化对象，上一轮迭代后的questioner
save_path=$3                # 本轮questioner后保存的存档目录
echo "save_path: $save_path"
export VLLM_USE_V1=0 VLLM_DISABLE_COMPILE_CACHE=1
: "${STORAGE_PATH:?请先激活 Data-Research}"
if curl -s --noproxy 127.0.0.1 --max-time 2 http://127.0.0.1:5000/ >/dev/null; then
    echo "打分端口 5000 已被占用" >&2
    exit 1
fi

# Solver 打分与 Questioner 训练共用 GPU 0。
CUDA_VISIBLE_DEVICES=0 python vllm_service_init/start_vllm_server.py \
    --port 5000 --model_path "$solver_model_path" --gpu_mem_util 0.20 &
scorer_pid=$!
stop_scorer() {
    kill "$scorer_pid" 2>/dev/null || true
    wait "$scorer_pid" 2>/dev/null || true
}
trap stop_scorer EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

deadline=$((SECONDS + 600))
until curl -s --noproxy 127.0.0.1 --max-time 2 http://127.0.0.1:5000/ >/dev/null; do
    if ! kill -0 "$scorer_pid" 2>/dev/null || (( SECONDS >= deadline )); then
        echo "Solver 打分服务启动失败或超时" >&2
        exit 1
    fi
    sleep 2
done

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

# Questioner 训练参数：先读取 config 指定的 YAML，再用下面的 key=value 覆盖对应配置。
# CUDA_VISIBLE_DEVICES=0：Questioner 在物理 GPU 0 上训练。
# max_response_length=4096：限制每条 Questioner 输出的最大生成长度（token 数）。
# model_path：从 questioner_model_path 加载要更新的 Questioner（actor）权重。
# experiment_name：用 save_path 标识本轮实验。
# save_checkpoint_path：将训练 checkpoint 写到 ${STORAGE_PATH}/models/$save_path。
# total_epochs=1000：外层数据遍历上限 1000 epoch；max_steps=6：本轮训练最多执行 6 个全局步。
# reward_function：冒号前是奖励函数文件，冒号后是函数名；compute_score 调用 Solver 服务给题目打分。
# val_freq=-1：关闭训练中的周期验证；训前和训后验证由训练器另行处理。
# format_prompt：用 questioner.jinja 把输入组织成出题提示。
# rollout.n=4：每个提示生成 4 条响应，当前每批 8 个提示共生成 32 条响应。
# actor.global_batch_size=4：每个更新分组对应 4 个提示，共 16 条响应。
# save_freq=1：每个训练步保存一次 checkpoint；下方合并命令取第 5 步的权重。

# 缩小例子（同一个全局步）
# 采样：A、B、C、D 各生成 4 条响应，共 16 条。
# GRPO 算优势：每个输入只比较自己的 4 条响应。
# 更新权重：假设每个 minibatch 含 2 个输入，先用 A+B 的 8 条响应更新一次，
# 再用 C+D 的 8 条响应更新一次。处理完这两组，就完成一个全局步。
# 实际配置每个全局步有 8 ÷ 4 = 2 个 minibatch，每组含 16 条响应。

CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.rollout_batch_size=8 \
    data.val_batch_size=8 \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=1 \
    worker.reward.reward_function_kwargs.reward_workers=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.gpu_memory_utilization=0.30 \
    worker.rollout.enforce_eager=true \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    trainer.max_steps=6 \
    trainer.save_freq=1

stop_scorer
trap - EXIT

# 合并模型
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor


echo "questioner training finished"
