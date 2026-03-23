# 🚀 SkillfulRAG: A Semantic-Aware & Agentic RAG Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![VectorDB: LanceDB](https://img.shields.io/badge/VectorDB-LanceDB-green.svg)](https://lancedb.com/)
[![Framework: MinerU](https://img.shields.io/badge/PDF_Parse-MinerU-orange.svg)](https://github.com/opendatalab/MinerU)

**SkillfulRAG** 是一个专为科研、复杂工业文档和代码仓库设计的端到端 RAG (Retrieval-Augmented Generation) 引擎。它拒绝“切碎”文档，而是通过 **“技能化 (Skill-based)”** 的架构，深度理解 Markdown 语法树 (AST)，并构建了一套包含 **混合检索 (Hybrid Search)** 与 **重排序 (Rerank)** 的工业级流水线。

> **💡 核心理念**：不仅仅是存储，而是通过语义层级（Heading Stack）和高性能检索链路，为 AI Agent 提供具备“深度上下文”的精准知识颗粒。

---

## 🛠️ 核心功能 (Key Features)

### 1. 📂 智能文档解析与结构化 (Smart Parsing)
* **MinerU 深度集成**：完美还原学术论文/技术文档中的公式、表格与多栏布局。
* **语义层级切片**：基于 `marko` 解析器，确保切片边界落在段落或标题处，每个 Chunk 自动继承 `Heading Stack`（例如：`[容器云 > SKE > 存储管理]`），极大提升检索召回精度。
* **确定性数据指纹**：引入 **UUID5** 机制，基于文件名生成固定 ID，确保向量库在增量更新时不会出现重复文档。

### 2. ⚡ 高性能向量与混合检索 (Storage & Search)
* **LanceDB 嵌入式驱动**：利用 Lance 格式实现毫秒级磁盘检索，原生支持 **向量 (Vector)** + **全文搜索 (FTS)** 混合模式，彻底解决专有名词（如错误码、Hash值）搜不到的痛点。
* **流式向量化**：内置 `EmbedManager` 支持 Generator 模式，在处理大规模文档时保持极低的内存占用。

### 3. 🎯 工业级检索流水线 (Retrieval Pipeline)
* **语义重排序 (Rerank)**：集成 **BGE-Reranker-v2**，在初筛候选集中执行二次精排，有效过滤 90% 以上的检索噪音。
* **上下文重组引擎**：自动将检索出的碎片整理为 LLM 易读的结构化 Context，并支持自动化来源引用（Citing）。

### 4. 🔌 AI-Native 技能架构 (Architecture)
* **模块化 Manager 设计**：`Embed`、`Vector`、`Rerank`、`LLM` 四大管理器逻辑解耦，参数配置严格遵循 `Environment Variables > config.yaml > Default` 优先级，确保生产环境安全性。

---

## 🏗️ 项目结构 (Architecture)

```text
.
├── config.yaml          # ⚙️ 全局配置中心 (切片阈值、API URL、System Prompt)
├── rag_engine.py        # 🧠 中枢神经 (Orchestrator: 串联所有组件)
├── scripts/             # 🛠️ 执行动力室 (Managers)
│   ├── embed_ops.py     # 向量计算与本地持久化
│   ├── vector_ops.py    # LanceDB 混合检索逻辑
│   ├── rerank_ops.py    # 语义打分与结果重组
│   └── llm_ops.py       # LLM 适配器与 Prompt 注入
├── skills/              # 📦 技能说明书 (Skill Manifests for Agents)
│   ├── chunk_skill.md   # 切片算法与 AST 规范
│   └── vector_skill.md  # 检索能力与字段描述
└── data/                # 💾 数据流转层 (Raw -> Processed -> Embedded)
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

### 2. 配置环境变量
```bash
# 敏感 Key 建议通过环境导出，避免写入配置文件
export LLM_API_KEY="sk-xxxx"
export EMBEDDING_API_KEY="sk-xxxx"
export RERANK_API_KEY="sk-xxxx"
```

### 3. 运行对话引擎 (Pipeline)
```python
from rag_engine import SkillfulRAGEngine

# 1. 初始化引擎 (自动对齐 YAML 配置与环境变量)
engine = SkillfulRAGEngine()

# 2. 一键获取专业回答
query = "SKE 1.8.6 的存储快照如何配置？"
response = engine.run(query)

print(f"🌟 AI Answer:\n{response}")
```

---

## 🧪 工程权衡 (Engineering Decisions)

1. **为什么选择混合检索？**
   在处理 Sangfor 内部代码和配置时，语义向量往往难以覆盖特定的哈希值或函数名。通过 **Vector + BM25 (FTS)**，我们实现了“模糊语义”与“精确匹配”的平衡。
2. **为什么 Rerank 是必须的？**
   单纯的向量相似度只代表“长得像”。在复杂的科研论文或技术文档中，Rerank (Cross-Encoder) 能识别出逻辑上的细微差别，显著降低 LLM 的幻觉率。
3. **WSL2 兼容性优化**
   针对 MinerU 运行时的显存分配问题，内置了显存探测逻辑，支持 `pipeline` 与 `vllm` 模式动态切换。

---

## 📅 路线图 (Roadmap)

- [x] 基于 MinerU 的 PDF 结构化解析 (MVP)
- [x] 基于 Markdown AST 的语义弹性切片
- [x] 基于 LanceDB 的向量/全文混合检索
- [x] 集成重排序 (Rerank) 模块
- [ ] 🚀 **多轮对话上下文 (Memory)**：支持基于会话历史的查询重写。
- [ ] 📊 **评估系统 (RAGAS)**：增加检索准确率与回答质量的自动化打分。
- [ ] 🤖 **Agentic Dispatcher**：由 Planner 自动决定是否需要调用外部工具补充知识。

---

## 🤝 贡献
如果你对 **Cloud-Native (K8s)**、**AIOps** 或 **医学图像处理** 感兴趣，欢迎提交 PR 或 Issue。

**Author:** Su Zelong (szl)  
**Affiliation:** Developed during research on Cloud-Native Infrastructure & Medical Image Segmentation.
