set -e

Base_model=$1
Model_abbr=$2
echo "Model_abbr: $Model_abbr"
: "${STORAGE_PATH:?请先设置 STORAGE_PATH}"
if [[ ! "$Model_abbr" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "实验名只能包含字母、数字、下划线和连字符" >&2
    exit 2
fi
for round in {1..2}; do
    for role in questioner solver; do
        output="$STORAGE_PATH/models/${Model_abbr}_${role}_v${round}"
        if [[ -e "$output" || -L "$output" ]]; then
            echo "实验目录已存在，请换一个实验名：$output" >&2
            exit 1
        fi
    done
done

# Initialize first iteration with base model
# 初始化第一轮迭代：

# 初始训练questioner
# 第一个参数：初始solver模型（打分）
# 第二个参数：初始questioner模型（进化对象）
# 第三个参数：保存这一轮迭代后的questioner的v1版本目录（进化后保存）
bash scripts/questioner_train_penalty.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1
# 初始训练solver
# 第一个参数：初始solver模型（进化对象）
# 第二个参数：本轮进化后的quetioner模型（出题）
# 第三个参数：保存这一轮迭代后的solver的v1版本目录（进化后保存）
bash scripts/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1
# STORAGE_PATH：运行时从外部提供的存储根目录，`main.sh` 里没有给它赋值
# 整个代码中统一用questioner_vx / solver——vx代表第几轮迭代后的questioner / solver

# 加上初始迭代，一共迭代2轮
# questioner 每一轮训练5步后保存模型权重checkpoint
# solver     每一轮训练15步后保存模型权重checkpoint
for i in {2..2}; do
    prev=$((i-1)) # 用于代表前一轮的迭代结果
    
    # Train questioner
    # 第一个参数：用于打分的solver，经过上一轮迭代的版本(prev)
    # 第二个参数：进化对象，经过上一轮迭代的版本(prev)
    # 第三个参数：本轮questioner迭代产生的结果的保存目录（{i}）
    bash scripts/questioner_train_penalty.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    # Train solver
    # 第一个参数：进化对象，经过上一轮迭代的版本(prev)
    # 第二个参数：出题的questioner，经过本轮首先进行迭代后的版本({i})
    # 第三个参数：本轮solver迭代产生的结果的保存目录({i})
    bash scripts/solver_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
        ${Model_abbr}_solver_v${i}

    # 两个新模型均已导出，清理上一轮的模型目录。
    rm -rf -- "$STORAGE_PATH/models/${Model_abbr}_questioner_v${prev}" \
        "$STORAGE_PATH/models/${Model_abbr}_solver_v${prev}"
done

# 原版外部基准评测含四卡任务，训练与导出结束后退出。
