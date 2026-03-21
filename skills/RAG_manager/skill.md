---
name: RAG_Manager
description: RAG 全链路协同调度器。负责从原始文档解析、语义切片到向量入库的全流程自动化编排与状态监控。
---

# RAG 全链路调度管理器 (RAG_Manager)

## 概述 (Overview)
RAG_Manager 是系统的“中枢神经”，负责编排文档从原始状态（PDF）到可检索向量状态（Vector DB）的完整生命周期。它通过逻辑判断顺序驱动 `DocParse_manager`、`Chunk_manager` 与 `VectorDB_manager` 模块。

---

## 核心工作流 (Core Workflow)

### 1. 自动化流水线编排 (Orchestration)
当 `data/raw` 目录出现新文档或用户指定特定 PDF 时，触发以下链式操作：
1.  **解析 (Parse)**: 调用 `skills/DocParse_manager/script/doc_parser.py`。
    - **输入**: `data/raw/*.pdf`
    - **输出**: `data/processed/*.md`
2.  **切片 (Chunk)**: 调用 `skills/Chunk_manager/script/chunk_process.py`。
    - **输入**: 上一步生成的 `.md` 文件。
    - **输出**: `data/chunks/*_chunks.json`
3.  **索引 (Index)**: 调用 `skills/VectorDB_manager/script/vector_db_process.py`。
    - **输入**: 上一步生成的 `.json` 切片。
    - **持久化**: 更新本地向量数据库索引。

### 2. 智能控制逻辑 (Intelligence Logic)
- **幂等性检查 (Idempotency)**：执行前扫描 `data/processed` 和 `data/chunks`。若目标 MD 或 JSON 已存在且源文件未更新，则**自动跳过**对应阶段。
- **原子化清理**：`DocParse_manager` 阶段产生的 `data/temp_output` 必须在每步完成后由 RAG_Manager 确认已清空。
- **并发控制**：支持对 `data/raw` 下的多个 PDF 进行批处理，但向量入库阶段需保持线程安全。

---

## 技能依赖契约 (Skill Dependency)

RAG_Manager 作为调度层，严格遵循以下依赖路径：
* **解析引擎**: `skills.DocParse_manager.script.doc_parser`
* **切片算法**: `skills.Chunk_manager.script.chunk_process`
* **存储引擎**: `skills.VectorDB_manager.script.vector_db_process`

---

## 资源引用 (Resources)

### 关键路径 (Critical Paths)
- **调度逻辑**: `skills/RAG_manager/script/pipeline.py`
- **全局配置**: `config.yaml` (管理 Embedding Model, Chunk Size, DB Path)

### 数据流向 (Data Flow)
`data/raw` (PDF) $\rightarrow$ `data/processed` (MD) $\rightarrow$ `data/chunks` (JSON) $\rightarrow$ `Vector Store`

---

## 交互示例 (Example)

**User**: "把 `data/raw` 下的新论文全处理了。"

**AI 动作**:
1. **扫描**: 发现 `paper_A.pdf` (新) 和 `paper_B.pdf` (已存在 MD)。
2. **调度**: 
   - [跳过] `paper_B` 解析阶段。
   - [执行] `python skills/DocParse_manager/script/doc_parser.py --file data/raw/paper_A.pdf`
   - [执行] `python skills/Chunk_manager/script/chunk_process.py --file data/processed/paper_A.md`
3. **反馈**: "✅ 增量更新完成。新增 1 篇文档，共生成 12 个切片，已同步至向量库。"

---

## 错误处理契约 (Error Handling)

当 Pipeline 运行失败时，RAG_Manager 必须返回以下格式的错误报告：
- **error_code**: 标识哪个环节出错（PARSE_ERR, CHUNK_ERR, DB_ERR）。
- **diagnostic_info**: 具体的报错堆栈或原因。
- **recovery_suggestion**: 针对该错误的建议操作（如：检查 PDF 是否加密、检查磁盘空间）。

### AI 应对策略：
- 如果报错是 `FILE_NOT_FOUND`：提醒用户核对 `data/raw` 路径。
- 如果报错是 `MINERU_CRASH`：建议用户检查 `magic-pdf` 的配置并重试。