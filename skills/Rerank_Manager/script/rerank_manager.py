import os
import json
import yaml
import requests
from typing import List, Dict, Any, Optional
from core.logger import get_logger

logger = get_logger("Rerank")

def load_config(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class RerankManager:
    def __init__(self, config_path: str = "config.yaml", **kwargs):
        config = load_config(config_path)
        rr_cfg = config.get("Rerank", {})

        # 优先级：参数 > 配置文件 > 环境变量
        self.url = kwargs.get("url") or os.getenv("RERANK_API_URL", "")
        self.api_key = kwargs.get("api_key") or os.getenv("RERANK_API_KEY", "")
        self.model = kwargs.get("model") or os.getenv("RERANK_MODEL", "")
        
        self.top_n = kwargs.get("top_n") or rr_cfg.get("top_n", 5)
        self.threshold = kwargs.get("threshold") or rr_cfg.get("threshold", 0.01)

        if not (self.url and self.api_key):
            logger.warning("[ReRank]: ⚠️ Warning: Rerank API URL or Key is missing. Rerank will be bypassed.")

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对候选文档进行重排
        documents 格式: [{"content": "...", "metadata": {...}}, ...]
        """
        if not self.url or not documents:
            return documents[:self.top_n]

        # 构造 API 请求 (以兼容 BGE/OpenAI 标准的 Rerank 接口为例)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 提取文本内容用于对比
        doc_contents = [doc.get("content", "") for doc in documents]
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_contents,
            "top_n": self.top_n
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            # 解析重排索引和分数
            # 常见格式为: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            reranked_results = []
            for item in result.get("results", []):
                idx = item["index"]
                score = item["relevance_score"]
                
                # 过滤低相关度噪音
                if score >= self.threshold:
                    original_doc = documents[idx]
                    original_doc["rerank_score"] = score
                    reranked_results.append(original_doc)
            
            return reranked_results

        except Exception as e:
            logger.error(f"[Rerank]: ❌ Rerank API Error: {e}")
            # 如果 Rerank 失败，作为降级方案，返回原始的前 N 个结果
            return documents[:self.top_n]

    def format_context(self, reranked_docs: List[Dict[str, Any]]) -> str:
        """
        将重排后的结果组装成 LLM 易读的字符串上下文
        """
        context_blocks = []
        for i, doc in enumerate(reranked_docs, 1):
            content = doc.get("content", "").replace("\n", " ")
            source = doc.get("metadata", {}).get("source", "Unknown")
            block = f"[{i}] (Source: {source}): {content}"
            context_blocks.append(block)
        
        return "\n\n".join(context_blocks)

# 调试代码
if __name__ == "__main__":
    # 模拟数据
    test_query = "如何安装 SKE 插件？"
    test_docs = [
        {"content": "SKE 插件安装指南...", "metadata": {"source": "doc1.md"}},
        {"content": "这是关于 CoreDNS 的说明", "metadata": {"source": "doc2.md"}},
        {"content": "深度学习模型训练教程", "metadata": {"source": "doc3.md"}} # 无关干扰项
    ]
    
    manager = RerankManager()
    final_docs = manager.rerank(test_query, test_docs)
    logger.info(manager.format_context(final_docs))