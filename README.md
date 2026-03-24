-----

# 🚀 SkillfulRAG: An Agentic & Semantic-Aware RAG Engine

[](https://www.python.org/downloads/)
[](https://lancedb.com/)
[](https://www.langchain.com/langgraph)

**SkillfulRAG** 不仅仅是一个 RAG 管道，它是一个具备**自主调度能力**的 AI Agent 框架。它通过深度理解 Markdown 语法树 (AST) 进行语义切片，并利用 **动态技能注册 (Skill Registry)** 机制，让 AI 能够根据用户指令自主规划“解析-切片-检索-生成”的完整链路。

> **💡 核心进化**：从“固定的 Pipeline”转向“动态的 Dispatcher”。系统不再死板地执行步骤，而是由 Orchestrator 根据任务复杂度实时拆解并调用对应技能。

-----

## 🛠️ 核心进化功能 (Key Features)

### 1\. 🤖 自主调度大脑 (Agentic Orchestrator)

  * **任务自动拆解**：基于 `LangGraph` 状态机，自动将模糊指令（如“处理这篇论文并入库”）拆解为有序的原子任务序列。
  * **参数级联覆盖**：严格遵循 `Runtime Args > Environment Variables > config.yaml > Default` 的优先级。用户通过口语指定的参数（如“切片设为 500”）能实时覆盖全局配置。

### 2\. 📂 语义感知切片 (Semantic-Aware Chunking)

  * **Heading Stack 继承**：基于 `marko` 解析器，每个 Chunk 自动携带完整的标题路径（如：`容器云 > SKE > 存储`），为 LLM 提供极致的上下文锚点。
  * **弹性边界提取**：结合 `jieba` 与正则引擎，自动为学术/工业文档提取高辨识度关键词与缩写（如 CNN, K8s），强化 FTS 全文搜索权重。

### 3\. 🔌 插件化技能架构 (Skill Registry)

  * **即插即用 (PnP)**：只需在 `skills/` 下新建文件夹并编写 `skill.md`，Orchestrator 即可感知并学会使用新技能，无需修改核心代码。
  * **数据总线 (Data Bus)**：通过 `Dispatcher` 实现任务间的数据透传，上一步的解析路径自动成为下一步的切片输入。

-----

## 🏗️ 项目架构 (Architecture)

```text
.
├── core/                # 🧠 调度中枢 (The Brain)
│   ├── agent_graph.py   # LangGraph 状态机定义
│   ├── orchestrator.py  # 任务规划器 (Planner)
│   ├── registry.py      # 技能注册中心 (Skill Registry)
│   └── dispatcher.py    # 任务执行分发器 (Executor)
├── skills/              # 📦 技能插件池 (The Muscles)
│   ├── DocParse_manager/
│   ├── Chunk_manager/   # 包含语义切片与 AST 逻辑
│   └── VectorDB_manager/# LanceDB 混合检索实现
├── config.yaml          # ⚙️ 全局配置 (默认切片、API 节点)
└── data/                # 💾 数据流转 (Raw -> Processed -> Chunk)
```

-----

## 🚦 快速开始 (Quick Start)

### 1\. 环境准备

```bash
git clone https://github.com/SuZeLong/SkillfulRAG.git
pip install -r requirements.txt
cp .env.example .env # 配置你的 API_KEY
```

### 2\. 启动 Agent 对话

```python
# 运行主入口，体验自动调度
python main.py

# 👤 用户 > 帮我把 data/raw/ske.pdf 解析了，切片设为 600，存入 my_db
# 📍 [节点进度]: planner 执行完毕
# 🧠 AI 思考: 用户需要处理 PDF 并指定了切片大小，流程为 Parse -> Chunk -> VectorDB
# 🚀 [Dispatcher]: 执行 Chunk_manager.chunk_text(size=600) ...
```

-----

## 🧪 工程权衡 (Engineering Decisions)

1.  **为什么引入 Dispatcher？**
    为了解决任务间的“硬编码”问题。通过 `internal_data` 状态总线，系统实现了文件路径的自动流转，极大地降低了各 Manager 之间的耦合度。
2.  **为什么坚持 Markdown AST？**
    纯文本切片会丢失层级关系。保留 `Heading Path` 让 Agent 在回答时能清晰地定位到“这在手册的哪个章节”，这对工业生产环境至关重要。
3.  **LanceDB 的选型优势**
    支持磁盘直接查询，无需启动复杂的数据库服务，极其适合单机环境及边缘侧部署（如 K8s 节点侧的 AIOps 助手）。

-----

## 🤝 贡献与联系

**Author:** Su Zelong (szl)  
**Research:** Cloud-Native Infrastructure & Medical Image Segmentation.

-----
