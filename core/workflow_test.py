import os
import logging
from skills.Embed_manager.script.embed_manager import EmbedManager
from skills.VectorDB_manager.script.vectordb_manager import LanceDBManager
from skills.Rerank_manager.script.rerank_manager import RerankManager
from core.llm_ops import LLMManager

# 配置日志，方便调试检索链路
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SkillfulRAGEngine:
    def __init__(self, config_path: str = "config.yaml"):
        logging.info("🚀 正在初始化 SkillfulRAG 引擎...")
        
        # 1. 实例化所有 Manager
        # 每个 Manager 内部都会自行处理 环境变量 > config.yaml 的逻辑
        self.embed_mgr = EmbedManager(config_path=config_path)
        self.vector_mgr = LanceDBManager(config_path=config_path)
        self.rerank_mgr = RerankManager(config_path=config_path)
        self.llm_mgr = LLMManager(config_path=config_path)
        
        logging.info("✅ 所有组件初始化完成。")

    def run(self, query: str, top_k_vector: int = 20, final_n: int = 5):
        """
        执行完整的 RAG 流程
        """
        logging.info(f"🔍 收到查询: {query}")

        # --- STEP 1: 向量化 (Embedding) ---
        query_vector = self.embed_mgr.embed_text(query)
        if not query_vector:
            return "Error: 向量化失败，请检查 API 网络。"

        # --- STEP 2: 初筛检索 (Vector Search / Hybrid Search) ---
        # 假设你的 VectorManager 有一个 search 方法返回 List[Dict]
        initial_results = self.vector_mgr.search(query_vector, limit=top_k_vector)
        logging.info(f"📦 初筛完成，召回了 {len(initial_results)} 条候选文档。")

        if not initial_results:
            return "抱歉，在知识库中未找到相关内容。"

        # --- STEP 3: 精排 (Rerank) ---
        # 传入原始 Query 和 初筛结果，返回打分排序后的结果
        reranked_results = self.rerank_mgr.rerank(query, initial_results)
        # 只要前 final_n 条作为上下文
        final_candidates = reranked_results[:final_n]
        logging.info(f"🎯 精排完成，选出前 {len(final_candidates)} 条最相关上下文。")

        # --- STEP 4: 上下文组装 (Context Construction) ---
        context_str = self.rerank_mgr.format_context(final_candidates)

        # --- STEP 5: LLM 生成答案 ---
        logging.info("🤖 正在请求 LLM 生成回答...")
        answer = self.llm_mgr.generate_answer(query, context_str)
        
        return answer

# 快捷测试
if __name__ == "__main__":
    # 确保你的环境变量已经 export
    engine = SkillfulRAGEngine()
    
    while True:
        user_input = input("\n👤 用户提问 (输入 q 退出): ")
        if user_input.lower() == 'q':
            break
            
        response = engine.run(user_input)
        print(f"\n🌟 AI 回答：\n{response}")