---
name: LanceDB_manager
description: 向量数据库全生命周期管理器。支持基于语义的增(Add)、删(Delete)、改(Update)、查(Query)操作。
---

# 向量数据库管理器 (LanceDB_manager)

## 概述 (Overview)
LanceDB_manager 是 RAG 系统的数据持久化层。它负责将 `Chunk_manager` 产生的 JSON 切片通过远程 Embedding 模型转化为向量，并存储在 LanceDB 中。

---

## 核心工作流 (Core Workflow)

### 1. 配置注入 (Config-First)
- **读取约定**：所有操作前必须解析根目录下的 `config.yaml`中的LanceDB的配置。
- **动态映射**：
    - 数据库路径：读取 `LanceDB.path`。
    - 目标表名：读取 `LanceDB.table_name`（默认 `chunks`）。
    - 检索数量：读取 `LanceDB.top_k`（默认 `5`）。

### 2. 执行 CRUD 指令 (Action Engine)
AI 根据用户意图调用 `/scripts/lancedb_ops.py`，调用规范如下：

| 意图 | 对应命令行指令 | 业务逻辑 |
| :--- | :--- | :--- |
| **入库 (Add)** | `python lancedb_ops.py add --file [JSON路径]` | 批量向量化并写入新切片。 |
| **检索 (Query)** | `python lancedb_ops.py query --q "[问题]"` | 执行语义搜索，返回 Top-K 相关片段。 |
| **删除 (Delete)** | `python lancedb_ops.py delete --source "[文件名].md"` | 按源文件名清理数据库。 |
| **更新 (Update)** | `python lancedb_ops.py update --file [JSON路径]` | 先删后加，确保文档版本一致性。 |

### 3. 数据流闭环 (Data Loop)
- **输入来源**：如果用户提供了切片文件路径，优先使用用户提供的，否则接收来自 `data/chunk_database` 的 JSON 文件
- **输出反馈**：
    - 检索时：直接将脚本输出的 JSON 片段展示给用户或传递给总结技能。
    - 写入时：反馈“成功入库/更新 [N] 条数据”。

---

## 约束与原则 (Constraints)

1. **配置隔离**：禁止在指令中硬编码 IP 地址或路径，必须引导脚本读取 `config.yaml`。
2. **批量优先**：在执行 `add` 操作时，确保脚本使用 Batch Embedding 模式以节省模型服务器资源。
3. **精准删除**：执行 `delete` 时必须指定 `--source`，严禁误删整个数据库。
4. **异常处理**：若脚本返回 `Error`，需提示用户检查模型服务器的联通性。

---

## 资源引用 (Resources)
- **脚本**: `/scripts/lancedb_ops.py` (CRUD 逻辑实现)
- **配置**: `config.yaml` (全局唯一参数来源)