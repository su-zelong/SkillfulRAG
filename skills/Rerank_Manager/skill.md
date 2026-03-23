---
name: Rerank_Manager
description: 知识精排与上下文重组器。通过 Cross-Encoder 模型对初筛结果进行语义校验，剔除无关噪音，为 LLM 生成提供高质量上下文。
---

# 精排管理器 (Rerank_Manager)

## 概述
Rerank_Manager 是检索质量的最后一道防线。向量检索（Vector Search）虽然快，但在处理“否定词”、“逻辑细节”和“专有名词”时容易出错。Rerank 负责纠正这些偏差。

## 核心功能指令集

| 指令 | 逻辑描述 | 适用场景 |
| :--- | :--- | :--- |
| `python rerank_ops.py --query "[Q]" --candidates "[JSON_PATH]"` | 对初筛出的候选集进行打分并重排。 | 标准 RAG 检索流。 |
| `python rerank_ops.py score --pair "[Q]" "[Doc]"` | 计算单对文本的相关性评分。 | 检索准确率测试/Debug。 |

## 数据契约 (Data Contract)
1. **输入格式**：接收来自 `Vector_Manager` 的 DataFrame 或 JSONL 列表（通常包含 `content`, `score`, `metadata`）。
2. **输出格式**：返回精简后的 `Top K` 文本块，且附带重排后的 `rerank_score`。

## 约束原则
1. **输入限制**：Rerank 属于重计算任务。禁止直接对数据库全量数据进行 Rerank，必须仅对 `Vector_Manager` 召回的 Top 50-100 进行操作。
2. **模型对齐**：建议使用与 `Embed_Manager` 语言分布一致的模型（如 BGE-Reranker-v2）。
3. **窗口保护**：输出的总字数不得超过 `config.yaml` 中定义的 `max_context_length`，防止 LLM 幻觉或 Token 溢出。

## 资源引用
- **脚本**: `/scripts/rerank_ops.py`
- **模型**: `BAAI/bge-reranker-v2-m3` (或对应的 API 接口)