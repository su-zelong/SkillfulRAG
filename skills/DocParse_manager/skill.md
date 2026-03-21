---
name: Document_Parser
description: 基于 MinerU (magic-pdf) 的自动化文档解析工具。支持将 PDF 转换为结构化 Markdown，并具备自动路径管理与缓存清理功能。
---

# 文档解析管理器 (Document_Parser)

## 概述 (Overview)
Document_Parser 是系统的预处理核心，负责将非结构化的 PDF 文件转换为标准化的 Markdown 格式。它通过调用 `/scripts/parse_pdf.py`（封装了 `magic-pdf` 命令行工具）确保文档中的表格、公式和布局被高保真还原，为后续的 `Chunk_manager` 提供高质量的输入源。

---

## 核心工作流 (Core Workflow)

### 1. 环境准备与配置加载
- **配置寻址**：启动时自动加载项目根目录下的 `config.yaml`。
- **目录初始化**：检查并确保输出目录 `data/md_database` 已存在，若缺失则自动创建。
- **模式确认**：默认使用 `auto` 模式（自动识别文本/OCR），确保解析成功率。

### 2. 调用解析脚本 (Script Execution)
- **调用方式**：通过命令行执行 `/scripts/parse_pdf.py`。
- **参数传递规范**：
    - `--file`: **必填**，指向待解析 PDF 文件的路径。
- **执行逻辑**：
    - 创建 `data/temp_output` 临时目录存放 MinerU 中间产物。
    - 调用系统命令：`magic-pdf -p {file} -o data/temp_output -m auto`。

### 3. 后处理与原子性操作 (Post-Processing)
- **文件提取**：解析完成后，从临时目录中精准定位生成的 `.md` 主文件。
- **持久化转移**：将 `.md` 文件重命名并移动至最终目的地：`data/md_database/{源文件名}.md`。
- **自动清理 (Self-Cleaning)**：**强制执行**临时文件夹 `data/temp_output` 的删除操作，确保磁盘空间不被图片和 JSON 碎片占用。

---

## 技能输出契约 (Output Contract)

解析任务结束后，必须向系统或用户返回以下状态：
- **status**: `success` / `failed`
- **output_path**: `data/md_database/{filename}.md`
- **artifacts**: 确认已生成的 Markdown 文件包含原始文档的层级结构（Header）。

---

## 资源引用 (Resources)

### scripts/
- `parse_pdf.py`: 封装了 `subprocess` 调用 MinerU 的核心 Python 逻辑。

### references/
- `config.yaml`: 存储解析模式及编码格式设置。
- **System Tool**: `magic-pdf