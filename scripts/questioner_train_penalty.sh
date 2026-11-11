#!/bin/bash

solver_model_path=$1        # 打分用，上一轮迭代后的solver
questioner_model_path=$2    # 进化对象，上一轮迭代后的questioner
save_path=$3                # 本轮questioner后保存的存档目录
echo "save_path: $save_path"
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N) # Bash 先执行 date +%s%N，再把它打印的时间戳存进 RUN_ID
export RUN_ID # 让子进程也能从环境变量里读到这个值

echo "RUN_ID=$RUN_ID"

# 启动 vllm 服务（记录 PID） （启动solver服务）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# 开始训练 Questioner
echo "Start training questioner: $questioner_model_path -> $save_path"

# Questioner 训练参数：先读取 config 指定的 YAML，再用下面的 key=value 覆盖对应配置。
# CUDA_VISIBLE_DEVICES=0,1,2,3：让训练进程看到这 4 张卡；n_gpus_per_node=4 指定每节点训练卡数。
# max_response_length=4096：限制每条 Questioner 输出的最大生成长度（token 数）。
# model_path：从 questioner_model_path 加载要更新的 Questioner（actor）权重。
# experiment_name：用 save_path 标识本轮实验。
# save_checkpoint_path：将训练 checkpoint 写到 ${STORAGE_PATH}/models/$save_path。
# total_epochs=1000：外层数据遍历上限 1000 epoch；max_steps=6：本轮训练最多执行 6 个全局步。
# reward_function：冒号前是奖励函数文件，冒号后是函数名；compute_score 调用 Solver 服务给题目打分。
# val_freq=-1：关闭训练中的周期验证；训前和训后验证由训练器另行处理。
# format_prompt：用 questioner.jinja 把输入组织成出题提示。
# rollout.n=4：每个出题提示采样 4 条响应；YAML 默认每批 512 个提示，共采样 2048 条响应。
# actor.global_batch_size=16：每个 actor 更新分组对应 16 个提示，即 16 × 4 = 64 条响应，再分到各卡。
# save_freq=1：每个训练步保存一次 checkpoint；下方合并命令取第 5 步的权重。

# 缩小例子（同一个全局步）
# 采样：A、B、C、D 各生成 4 条响应，共 16 条。
# GRPO 算优势：每个输入只比较自己的 4 条响应。
# 更新权重：假设每个 minibatch 含 2 个输入，先用 A+B 的 8 条响应更新一次，
# 再用 C+D 的 8 条响应更新一次。处理完这两组，就完成一个全局步。
# 实际配置每个全局步有 512 ÷ 16 = 32 个 minibatch，每组含 64 条响应。

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1000 \
    worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    data.format_prompt=./examples/format_prompt/questioner.jinja \
    worker.rollout.n=4 \
    worker.actor.global_batch_size=16 \
    trainer.max_steps=6 \
    trainer.save_freq=1

sleep 5

# 合并模型
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python

echo "questioner training finished"
