---
name: Chunk_manager
description: Markdown 文档语义切片工具。支持弹性长度控制、标题路径追踪及自动化 JSON 数据库构建。
---

# Markdown 文档切片管理器 (Chunk_manager)

## 概述 (Overview)
Chunk_manager 负责将非结构化 Markdown 转换为结构化切片。它通过调用 `/scripts/chunk_process.py` 确保在不破坏段落、公式和列表完整性的前提下，生成带有深度元数据的 JSON 片段。

---

## 核心工作流 (Core Workflow)

### 1. 路径寻址与前置转换
- **寻址**：优先读取用户指定路径，缺省则扫描 `data/md_database`。
- **格式守卫**：遍历文件，若发现非 `.md` 文件，**强制先调用** `Document_Parser` 技能。
- **用户提示**：转换前发送：“发现非 Markdown 文件，正在预处理为 MD 格式...”

### 2. 调用切片脚本 (Script Execution)
- **调用方式**：通过命令行执行 `/scripts/chunk_process.py`。
- **参数传递规范**：
    - `--file`: 必填，指向目标 Markdown 文件的绝对/相对路径。
    - `--size`: 选填，若用户指定了字数（如“每片1000字”），则必须传入该参数；否则缺省。
    - `--index`: 选填，文档在批处理中的序号。
- **语义约束**：
    - 允许 $L \pm 20\%$ 的波动。
    - **严禁切断**：代码块 (```)、LaTeX 公式 ($$)、Markdown 表格。

### 3. 持久化存储 (Data Persistence)
- **处理脚本输出**：脚本会返回一个标准 JSON 字符串（列表格式）。
- **文件写入**：将返回的 JSON 内容保存至 `data/chunk_database/{源文件名}_chunks.json`。
- **冲突处理**：若 JSON 已存在，询问用户：“检测到已存在切片数据库，是否覆盖？”

---

## 技能输出契约 (Output Contract)

每个切片字典必须包含以下字段：
- `chunk_id`: `{文件名}_{序号}.md`
- `content`: 实际文本内容
- `metadata`:
    - `heading_path`: 当前分片所属的完整标题路径（如 `["摘要", "引言"]`）
    - `keywords`: 3 个核心技术/领域关键词

---

## 资源引用 (Resources)

### scripts/
- `chunk_process.py`: 核心函数入口，负责 AST 解析与弹性切分。

### references/
- `config.yaml`: 默认参数配置文件。

---

## 交互示例 (Example)

**User**: "把 `data/md_database/` 下的文件都切了，每个片 500 字。"

**AI 动作**:
1. 扫描目录，发现 `paper1.md`。
2. 执行：`python /scripts/chunk_process.py --file data/md_database/paper1.md --size 500`。
3. 捕获脚本返回的 JSON 列表。
4. 写入文件：`data/chunk_database/paper1_chunks.json`。
5. **反馈**: "切片完成！已为 paper1 生成 8 个语义分片，存储在 data/chunk_database/。"