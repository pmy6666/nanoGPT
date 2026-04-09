# 02_trainer 学习路线

这个目录按“先吃透单卡，再理解优化，再看框架封装”的顺序组织。

## 第一阶段：单卡基础训练脚本

1. 代码：[stage1_minimal_train.py](./stage1_minimal_train.py)
2. 文档：[docs/stage1_minimal_train.md](./docs/stage1_minimal_train.md)

这一阶段重点吃透：

1. dataloader / batch 的构造
2. forward / loss / backward / optimizer step
3. label 右移一位
4. loss reshape
5. gradient clipping
6. optimizer.step() 后再 scheduler.step()

## 第二阶段：4 个关键优化

### 1. AMP 混合精度

1. 代码：[stage2_amp_train.py](./stage2_amp_train.py)
2. 文档：[docs/stage2_amp_train.md](./docs/stage2_amp_train.md)

### 2. 梯度累积

1. 代码：[stage2_grad_accum_train.py](./stage2_grad_accum_train.py)
2. 文档：[docs/stage2_grad_accum_train.md](./docs/stage2_grad_accum_train.md)

### 3. Gradient Checkpointing

1. 代码：[stage2_activation_checkpoint_train.py](./stage2_activation_checkpoint_train.py)
2. 文档：[docs/stage2_activation_checkpoint_train.md](./docs/stage2_activation_checkpoint_train.md)

### 4. Flash Attention / SDPA

1. 代码：[stage2_flash_attention_sdpa_train.py](./stage2_flash_attention_sdpa_train.py)
2. 文档：[docs/stage2_flash_attention_sdpa_train.md](./docs/stage2_flash_attention_sdpa_train.md)

这一阶段重点不是会开开关，而是要能解释：

1. 省的是哪部分显存
2. 会不会变慢
3. 对稳定性的影响
4. 应该插在训练脚本的什么位置

## 第三阶段：框架化训练脚本

### 1. Hugging Face Trainer

1. 代码：[stage3_hf_trainer_style.py](./stage3_hf_trainer_style.py)
2. 文档：[docs/stage3_hf_trainer_style.md](./docs/stage3_hf_trainer_style.md)

### 2. Accelerate

1. 代码：[stage3_accelerate_style.py](./stage3_accelerate_style.py)
2. 文档：[docs/stage3_accelerate_style.md](./docs/stage3_accelerate_style.md)

### 3. DeepSpeed

1. 代码：[stage3_deepspeed_style.py](./stage3_deepspeed_style.py)
2. 文档：[docs/stage3_deepspeed_style.md](./docs/stage3_deepspeed_style.md)

### 4. FSDP

1. 代码：[stage3_fsdp_style.py](./stage3_fsdp_style.py)
2. 文档：[docs/stage3_fsdp_style.md](./docs/stage3_fsdp_style.md)

## 配置文件

统一配置放在 [config.yaml](./config.yaml)。

每个阶段都有自己独立的配置段，例如：

1. `stage1_minimal_train`
2. `stage2_amp_train`
3. `stage2_grad_accum_train`
4. `stage2_activation_checkpoint_train`
5. `stage2_flash_attention_sdpa_train`
6. `stage3_*`

调参时优先改 YAML，不需要进 Python 代码里改硬编码。

## 共享模块

共享实现放在 [common.py](./common.py)。

里面集中放了：

1. tokenizer / dataset 构造
2. GPT 模型
3. attention 的 manual / SDPA 两种实现
4. gradient checkpointing 开关
5. optimizer / scheduler / eval helper
6. `config.yaml` 读取逻辑

## 推荐学习顺序

1. 先只看第一阶段代码，自己手写一遍。
2. 再逐个切换第二阶段四个优化。
3. 最后把第一阶段代码和第三阶段脚本逐行对照，建立“封装前后”的映射关系。
