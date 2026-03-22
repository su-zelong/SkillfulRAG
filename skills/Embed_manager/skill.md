---
name: Embed_Manager
description: 语义向量化处理器。负责将非结构化文本、JSON 切片转化为高维向量，并提供维度校验与批量预处理功能。
---

# 向量化管理器 (Embed_Manager)

## 概述 (Overview)
Embed_Manager 是 SkillfulRAG 的核心计算组件。它负责对接远程或本地的 Embedding 模型（如 OpenAI v3, BGE 等），将 `Chunk_manager` 生成的文本块转换为 LanceDB 可识别的向量特征。它是连接“文字”与“语义空间”的桥梁。

---

## 核心工作流 (Core Workflow)

### 1. 资源探测与初始化 (Initialization)
- **环境检查**：启动前必须确认环境变量 `EMBEDDING_API_KEY` 或 `config.yaml` 中的 `api_key` 已就绪。
- **配置对齐**：
    - **模型选择**：从 `Embedding.model` 读取（默认 `text-embedding-v3`）。
    - **维度对齐**：读取 `Embedding.dim`（如 `768` 或 `1536`），确保输出向量与数据库表结构匹配。
    - **输入路径**：默认监控 `data/chunk` 目录下的 `.jsonl` 文件。

### 2. 执行向量化指令 (Action Engine)
AI 根据用户意图调用 `/scripts/embed_ops.py`，调用规范如下：

| 意图 | 对应命令行指令 | 业务逻辑 |
| :--- | :--- | :--- |
| **单句转化 (Embed)** | `python embed_ops.py --text "[内容]"` | 将单条文本转化为向量列表，用于实时搜索测试。 |
| **批量预处理 (Batch)** | `python embed_ops.py batch --input [路径]` | 遍历目录，将所有 JSONL 文本批量转化为“文本+向量”格式。 |
| **维度校验 (Check)** | `python embed_ops.py info` | 返回当前配置模型的向量维度，供数据库建表参考。 |
| **数据清洗 (Clean)** | `python embed_ops.py format --file [路径]` | 清洗非法字符并确保输出的 JSON 格式符合 LanceDB 写入规范。 |

### 3. 数据流闭环 (Data Loop)
- **输入来源**：接收来自 `Chunk_manager` 切分后的结构化 JSON 数据或用户指定的文本。
- **输出反馈**：
    - 成功时：输出带有 `vector` 字段的增强型 JSON 数据，作为 `LanceDB_manager` 的输入。
    - 失败时：反馈具体的 API 错误码或网络异常信息。

---

## 约束与原则 (Constraints)

1. **配置隔离**：禁止在脚本中硬编码 API Key 或 URL，必须通过 `os.getenv` 或引导脚本读取 `config.yaml`。
2. **职责单一**：`Embed_Manager` 只负责“计算向量”，不负责“存入数据库”。存储动作必须通过调用 `LanceDB_manager` 完成。
3. **批量优先**：在处理 `data/chunk` 目录时，必须使用 Batch 模式以减少网络往返开销并节省 Token。
4. **异常鲁棒性**：若远程 API 返回限流（Rate Limit），脚本应具备简单的指数退避（Exponential Backoff）重试机制。

---

## 资源引用 (Resources)
- **脚本**: `/scripts/embed_ops.py` (Embedding 逻辑实现)
- **配置**: `config.yaml` (全局唯一参数来源，包含 `Embedding` 配置块)
- **依赖**: `requests`, `pyyaml`, `json`