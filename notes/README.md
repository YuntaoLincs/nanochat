# nanochat 学习笔记

> 学习方式：苏格拉底式重建 —— 从第一性原理出发，一步一步亲手重建 nanochat 代码仓。

## 学习路径

每一课对应一个核心模块或概念。在写代码之前，先回答问题；写完代码之后，再与原版对比。

| 课次 | 主题 | 文件 | 状态 |
|------|------|------|------|
| 第 01 课 | 语言模型是什么？Transformer 的骨架 | `lesson_01_transformer_skeleton.md` | 🟡 进行中 |
| 第 02 课 | 注意力机制：从点积到多头 | `lesson_02_attention.md` | 🟡 进行中 |
| 第 03 课 | 位置编码：RoPE | `lesson_03_rope.md` | ⬜ 未开始 |
| 第 04 课 | MLP、激活函数与残差流 | `lesson_04_mlp_residual.md` | ⬜ 未开始 |
| 第 05 课 | 初始化策略与 muP 思想 | `lesson_05_init.md` | ⬜ 未开始 |
| 第 06 课 | 优化器：AdamW + Muon | `lesson_06_optimizer.md` | ⬜ 未开始 |
| 第 07 课 | 分词器：BPE 从零实现 | `lesson_07_tokenizer.md` | ⬜ 未开始 |
| 第 08 课 | 数据加载器：流式 Token 流 | `lesson_08_dataloader.md` | ⬜ 未开始 |
| 第 09 课 | 训练循环：调度、梯度累积、日志 | `lesson_09_training_loop.md` | ⬜ 未开始 |
| 第 10 课 | 推理引擎：KV Cache | `lesson_10_kv_cache.md` | ⬜ 未开始 |

## 分支说明

- **master** — 原版完整代码
- **learn/rebuild** — 本学习分支，笔记 + 逐步重建的代码

## 重建代码的位置

```
rebuild/
├── gpt.py          # 第 01–05 课逐步构建
├── optim.py        # 第 06 课
├── tokenizer.py    # 第 07 课
├── dataloader.py   # 第 08 课
└── train.py        # 第 09 课
```
