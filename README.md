# 🚀 SkillfulRAG: A Semantic-Aware RAG Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: MinerU](https://img.shields.io/badge/PDF_Parse-MinerU-orange.svg)](https://github.com/opendatalab/MinerU)

**SkillfulRAG** 是一个专为科研和复杂文档设计的 RAG (Retrieval-Augmented Generation) 数据处理引擎。它拒绝粗暴地按字符数切碎文档，而是通过 **“技能化 (Skill-based)”** 的架构，深度理解 Markdown 语法树 (AST)，确保生成的知识块具备高完整性与层级上下文（Heading Stack）。

> **💡 核心理念**：像管理云原生插件一样管理文档解析任务，为 AI Agent 提供精准的“知识颗粒”。

---

## 🛠️ 核心功能 (Features)

* **📂 智能文档解析 (Smart Parsing)**
    * 深度集成 **MinerU (Magic-PDF)**，完美还原学术论文中的公式、表格与多栏布局。
    * 针对 **WSL2** 环境优化：内置显存分配保护，支持 `pipeline` 与 `vllm` 模式动态切换。
* **🧩 语义层级切片 (Semantic Chunking)**
    * **结构化切分**：基于 `marko` 解析器，确保切片边界永远落在段落或标题处，告别断句。
    * **上下文继承**：每个 Chunk 自动携带 `Heading Stack`（例如：`[研究背景 > 相关工作 > CNN模型]`），极大提升 RAG 检索精度。
* **📐 弹性 Overlap 算法**
    * 真正的滑动窗口：支持基于字符长度的“回溯窗口”逻辑，保证知识点在块与块之间平滑过渡。
* **🆔 确定性数据指纹**
    * 引入 **UUID5 (Namespace-based)**：基于文件名生成固定 ID，确保向量库在增量更新时不会出现重复文档。
* **🔌 AI-Native 技能架构**
    * 模块化 Skill 设计，所有参数均为 `Optional`。既能配合 `config.yaml` 自动化运行，也能由 **AI Agent** 通过 Function Calling 动态调参。

---

## 🏗️ 项目结构 (Architecture)

```text
.
├── config.yaml          # ⚙️ 全局配置中心 (切片大小、阈值、路径、环境开关)
├── main.py              # 🎮 任务调度器 (Dispatcher)
├── skills/              # 📦 核心技能库 (The "Skills" Vault)
│   ├── pdf_skill.py     # PDF 深度解析 (MinerU Wrapper)
│   ├── chunk_skill.py   # 语义弹性切片核心逻辑
│   ├── keyword_skill.py # TF-IDF 关键词提取扩展
│   └── registry.py      # 技能注册表 (Map files to specific skills)
└── data/                # 💾 数据流转层
    ├── raw/             # 待处理原始 PDF
    ├── process/         # 解析后的结构化 Markdown
    └── chunk/           # 最终生成的 JSONL (Ready for LanceDB/VectorDB)
```

---

## 🚦 快速开始 (Quick Start)

### 1. 环境准备
```bash
# 克隆仓库
git clone [https://github.com/SuZeLong/SkillfulRAG.git](https://github.com/SuZeLong/SkillfulRAG.git)
cd SkillfulRAG

# 安装核心依赖
pip install -r requirements.txt
```

### 2. 配置你的 Baseline
在 `config.yaml` 中定义你的默认实验参数：
```yaml
Chunk:
  chunk_size: 600      # 理想切片字数
  threshold: 0.15      # 允许 15% 的长度弹性缩放
  overlap: 200         # 语义重叠区大小
```

### 3. 一键流水线示例 (Pipeline)
```python
from skills.pdf_skill import parse_pdf
from skills.chunk_skill import chunk_text

# 1. 解析 PDF 并还原结构
md_file = parse_pdf("data/raw/thesis_baseline.pdf")

# 2. 执行语义化弹性切片
# 即使不传参，系统也会自动对齐 YAML 配置
chunks = chunk_text(file_path=md_file, user_size=800)
```

---

## 🧪 工程权衡 (Engineering Decisions)

1.  **为什么放弃 `RecursiveCharacterTextSplitter`?**
    传统的字符切分不理解语义。SkillfulRAG 坚持解析 AST，确保标题（Heading）与其下的首个段落不会被强行拆散，这在学术论文 RAG 场景中至关重要。
2.  **为什么采用 Optional + Config 设计?**
    为了平衡“自动化”与“灵活性”。这使得项目既可以作为稳定的后台服务运行，也可以让 AI Agent 在对话中即时调整切片粒度。
3.  **WSL2 兼容性补丁**
    针对开发中遇到的 `vllm` 显存分配不均报错，我们在 `pdf_skill` 中内置了探测逻辑，优先保障解析流程的闭环。

---

## 📅 路线图 (Roadmap)

- [x] 基于 MinerU 的 PDF 结构化解析 (MVP)
- [x] 基于 Markdown AST 的语义切片算法
- [x] 基于 UUID5 的文档唯一标识系统
- [ ] 🚀 接入 **LanceDB** 向量存储 Skill
- [ ] 🤖 增加 **Intent-Dispatcher** (基于 LLM 自动路由解析技能)
- [ ] 📊 知识块检索评估 Dashboard

---

## 🤝 贡献
如果你对 **Cloud-Native (K8s)**、**AIOps** 或 **医学图像处理** 感兴趣，欢迎提交 PR 或 Issue。

**Author:** Su Zelong (szl)
**Context:** Developed during postgraduate research on AI Segmentation.
