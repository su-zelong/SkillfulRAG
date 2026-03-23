---
name: Embed_Manager
description: 语义向量化处理器。负责将 Chunk 转化为高维向量，管理 Embedding 资产的本地持久化与维度对齐。
---

# 向量化管理器 (Embed_Manager)

## 概述 (Overview)
Embed_Manager 是 SkillfulRAG 知识图谱构建的关键。它通过调用 Embedding 模型将“原始文本”映射为“语义向量”，是实现向量检索（Vector Search）的前提。该组件不仅提供计算功能，还负责管理中间态的 `Embedded JSONL` 数据资产。

---

## 核心工作流 (Core Workflow)

### 1. 自动配置寻踪 (Auto-Config)
- **多级加载**：优先读取 `kwargs` 参数，其次检索 `config.yaml`，最后回退至 `os.getenv`。
- **动态寻址**：自动扫描 `Embedding.input_path` (默认 `data/chunk`)，并确保 `Embedding.output_path` (默认 `data/embedded/<model_name>`) 存在，实现输入输出闭环。

### 2. 执行指令集 (Instruction Set)
AI Agent 应根据任务上下文选择最轻量的指令：

| 场景意图 | 对应命令行指令 | 核心逻辑 |
| :--- | :--- | :--- |
| **单句检索测试** | `python embed_ops.py --text "[Content]"` | 快速转化用户查询，不产生持久化文件。 |
| **离线全量构建** | `python embed_ops.py batch` | 遍历目录，将所有 Chunk 转化为“文本+向量”并存入 `data/embedded/<model_name>`。 |
| **断点续传/同步** | `python embed_ops.py sync` | **[优化]** 对比输入输出目录，仅对新增或修改的 Chunk 进行增量 Embedding。 |
| **数据库 Schema 参考**| `python embed_ops.py info` | 显式返回 `dim` 和 `model_name`，防止跨模型检索导致的语义错乱。 |

### 3. 数据管线原则 (Pipeline Logic)
- **解耦存储**：`Embed_Manager` 输出标准的 `.jsonl` 向量包。它可以被 `LanceDB_manager` 摄入，也可以作为独立资产迁移。
- **流式写入**：为了处理大规模文档（如 SKE 源码库），必须采用 **Generator 模式** 边计算边写入本地，防止 OOM (内存溢出)。

---

## 约束与增强原则 (Constraints & Best Practices)

1. **原子性操作**：每条 Embedding 记录必须包含原始 `id` 和 `metadata`，确保在 LanceDB 中可追溯。
2. **幂等性校验**：重复执行 `batch` 指令不应产生重复的 API 调用（建议通过 `content_hash` 校验）。
3. **模型锁定**：严禁在同一张 LanceDB 表中混合不同模型的向量。执行前必须校验 `model_name` 是否一致。
4. **异常重试策略**：
    - **429 (Rate Limit)**：必须执行指数退避重试（Exponential Backoff）。
    - **5xx (Server Error)**：记录失败的行号并跳过，确保整体任务不中断。
5. **隐私合规**：在调用公网 API (如 OpenAI) 前，敏感数据（如密钥、公司内部 IP）应在 `Chunk_manager` 阶段完成脱敏。

---

## 资源与依赖 (Resources)
- **主程序**: `/scripts/embed_ops.py` (内含 `EmbedManager` 类)
- **数据资产**: 
    - 输入：`data/chunk/*.jsonl`
    - 输出：`data/embedded/<model_name>/*.jsonl` (由 `.gitignore` 保护)
- **核心库**: `requests`, `pyyaml`, `lancedb` (用于 Schema 定义)