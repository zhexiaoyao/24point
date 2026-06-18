# GRPO 24-Point Solver

基于 Qwen2.5-1.5B-Instruct 和 TRL GRPO 的 24 点游戏强化学习项目。目标是让模型按 R1 风格输出：

```text
<think>...</think><answer>...</answer>
```

其中 `<answer>` 内必须是合法算式：只使用题目给出的 4 个数字各一次、运算符 `+ - * /` 和括号，并且结果等于 24。

## 云端环境

推荐选择截图里的 PyTorch x86 镜像之一：

- `2.1.0-cuda12.1-py3.10.6-ubuntu22.04-x86_64`
- 或 `2.1.0-cuda12.1-py3.9.11-ubuntu22.04-x86_64`

Python 3.10 更推荐。镜像已含 CUDA/PyTorch 时，先不要重复安装 torch。

## 快速开始

```bash
git clone <your-github-repo-url>
cd 24point
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`requirements.txt` 会以 editable 模式安装本项目，因此后续脚本可以直接导入 `twentyfour` 包。

如果云端 PyTorch 镜像是 `PyTorch-2.1.0`，不要安装最新版 `transformers/trl/peft`。本项目已固定一组兼容 PyTorch 2.1 的版本；如果之前已经装过最新版，执行：

```bash
python -m pip install --force-reinstall -r requirements.txt
```

如果平台 GPU 驱动较旧，例如 `nvidia-smi` 显示 `Driver Version: 470.xx / CUDA Version: 11.4`，需要使用 CUDA 11.x 版 PyTorch：

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install "numpy<2"
```

先跑本地逻辑测试：

```bash
python -m pytest -q
```

快速检查数据集处理和 prompt：

```bash
python scripts/prepare_data.py --train-limit 8 --eval-limit 8 --out-dir data/processed
```

如果 Hugging Face 数据集缓存报 `TypeError: must be called with a dataclass type or instance`，说明缓存版本不兼容，强制重新下载：

```bash
python scripts/prepare_data.py --out-dir data/processed --force-download
```

启动 GRPO 训练：

```bash
accelerate launch scripts/train_grpo.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/train_nlile_solvable.jsonl \
  --output_dir outputs/qwen2.5-1.5b-24point-grpo \
  --max_steps 800 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --learning_rate 5e-6 \
  --bf16
```

如果纯 GRPO 训练长期出现 `loss: 0.0`、`grad_norm: nan`、`kl: nan` 且正确率奖励一直为 0，先做 SFT warmup：

```bash
python scripts/build_sft_data.py --out-file data/processed/sft_train.jsonl

python scripts/train_sft.py \
  --model_name models/Qwen2.5-1.5B-Instruct \
  --train_file data/processed/sft_train.jsonl \
  --output_dir outputs/qwen2.5-1.5b-24point-sft \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --use_peft \
  --lora_r 8 \
  --lora_alpha 16 \
  --report_to none
```

P100 上 SFT 如果第一步出现 `Non-finite loss ... nan`，不要加 `--fp16`；用 fp32 做 SFT warmup 更稳。GRPO 阶段可以继续使用 `--fp16` 节省显存。

然后把 GRPO 的 `--model_name` 改成 SFT 输出目录：

```bash
accelerate launch scripts/train_grpo.py \
  --model_name outputs/qwen2.5-1.5b-24point-sft \
  --train_file data/processed/train_nlile_solvable.jsonl \
  --output_dir outputs/qwen2.5-1.5b-24point-grpo \
  --max_steps 800 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_generations 2 \
  --max_completion_length 128 \
  --learning_rate 5e-6 \
  --fp16 \
  --use_peft \
  --lora_r 8 \
  --lora_alpha 16 \
  --report_to none
```

显存紧张时可加：

```bash
  --use_peft --lora_r 16 --lora_alpha 32 --gradient_checkpointing
```

训练后评测：

```bash
python scripts/evaluate.py \
  --model_path outputs/qwen2.5-1.5b-24point-grpo \
  --split hard \
  --limit 100
```

使用本地 hard split 做 best-of-N 评测：

```bash
python scripts/evaluate.py \
  --model_path outputs/qwen2.5-1.5b-24point-grpo-shaped \
  --eval_file data/processed/eval_game_of_24_hard.jsonl \
  --limit 100 \
  --num_samples 8 \
  --max_new_tokens 128
```

如果训练阶段再次遇到 Hugging Face dataset cache 报错，确认先执行过 `scripts/prepare_data.py`，并在训练命令里保留 `--train_file data/processed/train_nlile_solvable.jsonl`。这样训练会直接读取本地 JSONL，不再重新解析远端数据集缓存。

## 项目结构

```text
src/twentyfour/
  data.py        数据集加载、字段归一化、prompt 构造
  prompts.py     R1 风格提示词
  rewards.py     GRPO 奖励函数
  verifier.py    24 点答案解析与程序化校验
scripts/
  prepare_data.py
  train_grpo.py
  evaluate.py
  infer.py
tests/
  test_verifier.py
```

## 数据集

训练默认使用 `nlile/24-game` 中可解样本；评测使用：

- `test-time-compute/game-of-24`：尤其适合取论文常用 `indices 900-1000` 难题。
- `nlile/24-game` 中不可解样本：检查模型是否会瞎编。

数据集会通过 Hugging Face `datasets` 在线下载。云服务器如果无法访问 Hugging Face，请先配置镜像源或提前缓存数据。

## 监控指标

训练脚本会记录可验证奖励：

- `answer_format_reward`：是否包含合法 `<answer>` 标签。
- `valid_expression_reward`：算式是否合法、数字是否刚好使用一次。
- `proximity_reward`：合法算式的结果越接近 24，奖励越高。
- `correct_reward`：算式是否等于 24。

默认奖励采用分层权重：格式正确 `+0.5`、格式错误 `-0.5`；合法表达式 `+1`、非法表达式 `-1`；距离奖励最高 `+0.1`；精确等于 24 额外 `+5`。这样可避免 proximity reward 压过任务的精确目标。

定量分析建议报告：

- in-distribution success rate。
- hard split success rate。
- unsolvable split false-positive rate。
- 若启用 OOD，可加入 Countdown 任务扩展结果。

## GitHub 上传

本地初始化并提交：

```bash
git init
git add .
git commit -m "Initial GRPO 24-point solver"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
