---
name: Retriever_Manager
description: 语义检索与上下文召回器。负责执行相似度搜索，从向量数据库中提取最相关的知识片段。
---

# 检索管理器 (Retriever_Manager)

## 概述 (Overview)
Retriever_Manager 是 SkillfulRAG 的“大脑索引”。它负责接收用户的自然语言提问，协同 `Embed_Manager` 生成查询向量，并从 `LanceDB_manager` 维护的数据库中召回 Top-K 个最相关的文档切片，为大模型的生成提供事实依据。

---

## 核心工作流 (Core Workflow)

### 1. 语义解析 (Semantic Mapping)
- **向量化查询**：调用 `Embed_Manager` 将用户输入的 `query`（如“如何配置 K8s 备份？”）转化为高维向量。
- **配置读取**：
    - **检索深度**：从 `config.yaml` 读取 `LanceDB.top_k`（默认 `5`）。
    - **相似度阈值**：读取 `LanceDB.threshold`（如 `0.7`），过滤相关度过低的低质量噪声。

### 2. 执行检索指令 (Search Engine)
AI 根据用户意图调用 `/scripts/retrieve_ops.py`，调用规范如下：

| 意图 | 对应命令行指令 | 业务逻辑 |
| :--- | :--- | :--- |
| **基础检索 (Search)** | `python retrieve_ops.py --q "[问题]"` | 返回最相关的 Top-K 个 JSON 片段。 |
| **混合检索 (Hybrid)** | `python retrieve_ops.py --q "[问题]" --keyword "[词]"` | 结合语义向量和关键词匹配，提升专有名词命中率。 |
| **带过滤检索 (Filter)** | `python retrieve_ops.py --q "[问题]" --source "[文件名]"` | 仅在指定的源文件中进行检索。 |
| **检索评估 (Eval)** | `python retrieve_ops.py --test` | 模拟检索并输出相似度得分（Score），用于调优 Embedding 模型。 |

### 3. 上下文重组 (Context Assembly)
- **数据清洗**：对召回的片段进行去重，并按照相似度从高到低排序。
- **格式化输出**：将结果封装为 Markdown 引用块或标准 JSON，供提示词工程（Prompt Engineering）直接引用。
- **溯源信息**：每个召回片段必须包含 `source`（来源文件）和 `index`（切片编号），确保回答可追溯。

---

## 约束与原则 (Constraints)

1. **链式调用优先**：`Retriever_Manager` 严禁直接读取数据库文件，必须通过 `LanceDB_manager` 的接口进行查询。
2. **拒绝空回答**：若所有片段的相似度均低于阈值，必须明确反馈“未找到相关知识库内容”，而不是返回无关信息。
3. **性能监控**：检索延迟应控制在 500ms 以内。若延迟过高，需提示用户检查 LanceDB 的索引配置（如 IVF-PQ 索引）。
4. **上下文配额**：召回的总字符数不得超过 `config.yaml` 中定义的 `max_context_length`，防止撑爆 LLM 的上下文窗口。

---

## 资源引用 (Resources)
- **脚本**: `/scripts/retrieve_ops.py` (检索逻辑与重排实现)
- **配置**: `config.yaml` (包含 `top_k`, `threshold` 等检索参数)
- **联动技能**: `Embed_Manager` (负责 Query 向量化), `LanceDB_manager` (负责向量比对)