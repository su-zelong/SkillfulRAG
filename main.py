import os
import logging
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Literal

# LangGraph & LangChain 核心
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# 导入你开发的底层“肌肉” (假设路径已配置)
from skills.Embed_manager.script.embed_manager import EmbedManager
from skills.VectorDB_manager.script.vectordb_manager import LanceDBManager
from skills.Rerank_manager.script.rerank_manager import RerankManager
from skills.DocParse_manager.script.docparse_manager import DocParser
from core.llm_ops import LLMManager

# ==========================================
# 配置日志
# ==========================================
os.makedirs("logs", exist_ok=True)
log_file = f"logs/skillful_rag_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger("SkillfulRAG-Agent")

# ==========================================
# 环境变量
# ==========================================

## 决策模型
PLAN_MODEL_NAME = os.getenv("PLAN_MODEL_NAME", "gpt-4-turbo")
PLAN_MODEL_API = os.getenv("PLAN_MODEL_API", "https://api.openai.com/v1/chat/completions")
PLAN_MODEL_API_KEY = os.getenv("PLAN_MODEL_API_KEY", "")

## 配置文件地址
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")


# ==========================================
# 2. 状态定义 (State Schema)
# ==========================================
class AgentState(TypedDict):
    query: str                       # 原始问题
    plan_args: Optional[Dict]         # AI 读完 skill.md 后的决策参数
    search_results: List[Dict]       # 检索到的原始数据
    refined_context: List[Dict]      # Rerank 后的精选数据
    grading_score: float             # 质量评分 (0-1)
    loop_count: int                  # 修正循环计数
    final_answer: str                # 最终输出

# ==========================================
# 3. 核心节点实现 (Nodes)
# ==========================================

# 初始化所有业务 Manager
vector_mgr = LanceDBManager(CONFIG_PATH)
rerank_mgr = RerankManager(CONFIG_PATH)
llm_mgr = LLMManager(CONFIG_PATH)
embed_mgr = EmbedManager(CONFIG_PATH)

# 定义决策大脑 (GPT-4 级别建议用于 Planning)
router_llm = ChatOpenAI(model=PLAN_MODEL_NAME, temperature=0)

def planner_node(state: AgentState):
    """【意图识别】读取 skill.md 描述，决定搜索策略"""
    logger.info("--- [Node: Planner] 分析意图并对齐技能规范 ---")
    
    # 动态加载技能描述
    with open("skills/VectorDB_manager/skill.md", "r", encoding="utf-8") as f:
        vector_skill_desc = f.read()
    with open("skills/Rerank_manager/skill.md", "r", encoding="utf-8") as f:
        rerank_skill_desc = f.read()
    with open("skills/Embed_manager/skill.md", "r", encoding="utf-8") as f:
        embed_skill_desc = f.read()
    with open("skills/DocParse_manager/skill.md", "r", encoding="utf-8") as f:
        doc_parse_skill_desc = f.read()
    with open("skill/Retrieval_manager/skill.md", "r", encoding="utf-8") as f:
        retrieval_skill_desc = f.read()
    with open("skill/Chunk_manager/skill.md", "r", encoding="utf-8") as f:
        chunk_skill_desc = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个 RAG 技术专家。根据以下技能规范描述，提取检索参数：\n{skill_desc}"),
        ("human", "用户问题：{query}\n请以 JSON 格式输出参数：{{'query': '关键词', 'version': '版本号或null'}}")
    ])
    
    # 模拟 AI 提取过程
    chain = prompt | router_llm
    ai_msg = chain.invoke({"query": state["query"]})
    
    # 解析 AI 输出的参数 (实际建议使用 with_structured_output)
    try:
        args = json.loads(ai_msg.content)
    except:
        args = {"query": state["query"], "version": None}
        
    logger.info(f"Planner 决策结果: {args}")
    return {"plan_args": args, "loop_count": state.get("loop_count", 0) + 1}

def retrieve_node(state: AgentState):
    """【物理执行】调用 LanceDB 进行混合检索"""
    logger.info("--- [Node: Retrieve] 正在执行数据库检索 ---")
    args = state["plan_args"]
    
    # 生成向量
    q_vec = embed_mgr.embed_text(args.get("query"))
    v_filter = f"version == '{args['version']}'" if args.get("version") else None
    
    # 执行检索
    results = vector_mgr.search(q_vec, limit=15, filter=v_filter)
    logger.info(f"检索完成，找到 {len(results)} 条候选分片。")
    return {"search_results": results}

def grade_and_rerank_node(state: AgentState):
    """【质量评估】Rerank 并给出相关性分值"""
    logger.info("--- [Node: Grade & Rerank] 正在进行结果精排与质量评分 ---")
    query = state["query"]
    candidates = state["search_results"]
    
    if not candidates:
        logger.warning("检索结果为空，质量分数归零。")
        return {"grading_score": 0.0, "refined_context": []}

    # 执行重排序
    refined = rerank_mgr.rerank(query, candidates)
    
    # 简单启发式评分：基于 Rerank 最高分或数量
    top_score = refined[0].get('rerank_score', 0) if refined else 0
    logger.info(f"精排完成，最高相关性分值: {top_score}")
    
    return {"refined_context": refined[:5], "grading_score": top_score}

def generate_node(state: AgentState):
    """【生成回答】整合上下文并回复"""
    logger.info("--- [Node: Generate] 正在合成最终专业回答 ---")
    context = rerank_mgr.format_context(state["refined_context"])
    answer = llm_mgr.generate_answer(state["query"], context)
    return {"final_answer": answer}

# ==========================================
# 4. 路由逻辑 (Adaptive Routing)
# ==========================================

def should_continue(state: AgentState) -> Literal["replan", "finish"]:
    """判断是否需要启动自我修正循环"""
    score = state["grading_score"]
    count = state["loop_count"]
    
    if score < 0.6 and count < 2:
        logger.warning(f"检索质量不达标 (Score: {score}), 正在尝试第 {count} 次修正...")
        return "replan"
    
    logger.info(f"流程结束，进入生成环节。最终质量分: {score}")
    return "finish"

# ==========================================
# 5. 构建图 (Graph Construction)
# ==========================================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_and_rerank_node)
workflow.add_node("generate", generate_node)

# 设置逻辑连线
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retrieve")
workflow.add_edge("retrieve", "grade")

# 核心：条件路由
workflow.add_conditional_edges(
    "grade",
    should_continue,
    {
        "replan": "planner",  # 回到 Planner 节点重新提取关键词
        "finish": "generate"   # 直达生成
    }
)

workflow.add_edge("generate", END)

# 编译应用
app = workflow.compile()

# ==========================================
# 6. 运行入口
# ==========================================
if __name__ == "__main__":
    logger.info("=== SkillfulRAG Agent Session Started ===")
    
    user_question = "SKE 1.8.6 的存储快照怎么配置？"
    
    # 运行图流
    for output in app.stream({"query": user_question, "loop_count": 0}):
        for key, value in output.items():
            logger.debug(f"State Update from Node: {key}")

    # 获取结果 (在生产环境中建议使用 thread_id 维护状态)
    # final_result = app.invoke({"query": user_question})
    # print(f"\nFinal Answer:\n{final_result['final_answer']}")
    
    logger.info("=== Session Finished ===")