import os
import yaml
import requests
from typing import List, Dict, Any, Optional

def load_config(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class LLMManager:
    def __init__(self, config_path: str = "config.yaml", **kwargs):
        # 1. 加载本地配置
        config = load_config(config_path)
        llm_cfg = config.get("LLM", {})

        # 2. 严格按优先级获取核心参数：环境变量 > 传入参数 > 配置文件
        # 敏感信息及模型名优先从环境获取
        self.api_key = os.getenv("LLM_API_KEY") or kwargs.get("api_key") or llm_cfg.get("api_key")
        self.url = os.getenv("LLM_API_URL") or kwargs.get("url") or llm_cfg.get("url")
        self.model = os.getenv("LLM_MODEL_NAME") or kwargs.get("model") or llm_cfg.get("model", "gpt-3.5-turbo")

        # 3. 策略参数（Temperature等）从配置文件获取
        self.temperature = llm_cfg.get("temperature", 0.3)
        self.timeout = llm_cfg.get("timeout", 60)
        self.system_prompt = llm_cfg.get("system_prompt", "你是一个专业的技术助手。")

        if not (self.url and self.api_key):
            raise ValueError("❌ 错误: 缺少 LLM API URL 或 Key！请检查环境变量或 config.yaml")

    def generate_answer(self, query: str, context: str) -> str:
        """
        结合 Rerank 后的上下文生成最终答案
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 组装 Prompt
        user_content = f"【参考资料】：\n{context}\n\n【用户问题】：\n{query}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": self.temperature
        }

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"❌ LLM 生成异常: {str(e)}"

# 调试代码
if __name__ == "__main__":
    # 模拟经过 Rerank 后的上下文
    sample_context = "[1] SKE 1.8.6 支持通过 ConfigMap 修改 CoreDNS 配置。\n[2] 修改后需重启 Pod 生效。"
    sample_query = "1.8.6 的 CoreDNS 怎么改配置？"

    try:
        llm = LLMManager()
        answer = llm.generate_answer(sample_query, sample_context)
        print(f"回答：\n{answer}")
    except Exception as e:
        print(e)