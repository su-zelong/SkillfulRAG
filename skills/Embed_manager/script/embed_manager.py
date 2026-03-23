import yaml
import json
import os
import sys
import requests
from typing import Optional, List, Dict, Any, Generator

def load_config(config_path: str = "config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

class EmbedManager:
    def __init__(self, 
                 config_path: str = "config.yaml",
                 **kwargs):
        
        config = load_config(config_path)
        emb_cfg = config.get("Embedding", {})
        
        # 优先级：kwargs > config.yaml > 环境变量 > 默认值
        self.url = kwargs.get("url") or os.getenv("EMBEDDING_API")
        self.api_key = kwargs.get("api_key") or os.getenv("EMBEDDING_API_KEY")
        self.model_name = kwargs.get("model_name") or os.getenv("EMBEDDING_MODEL_NAME")

        self.dim = kwargs.get("dim") or emb_cfg.get("dim", 768)
        self.input_path = kwargs.get("input_path") or emb_cfg.get("input_path", "data/chunk")
        self.output_path = kwargs.get("output_path") or emb_cfg.get("output_path", f"data/embedded/{self.model_name}")

        # 变量拼写修正：self.mode_name -> self.model_name
        if not (self.url and self.api_key):
            print(f"Error: API URL or Key missing!")
            sys.exit(1)

        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)

    def embed_text(self, text: str) -> List[float]:
        """单条文本向量化"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"input": text, "model": self.model_name}
        
        try:
            # 增加重试机制或简单的异常捕获
            response = requests.post(self.url, headers=headers, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json().get("data", [])
            return data[0].get("embedding", []) if data else []
        except Exception as e:
            print(f"Embedding API error: {e}")
            return []

    def _get_files(self) -> List[str]:
        """获取待处理的文件列表"""
        if os.path.isdir(self.input_path):
            return [os.path.join(self.input_path, f) for f in os.listdir(self.input_path) if f.endswith('.jsonl')]
        return [self.input_path] if os.path.exists(self.input_path) else []

    def process_generator(self) -> Generator[Dict[str, Any], None, None]:
        """
        [核心优化] 使用生成器逐行处理，防止大文件撑爆内存
        """
        files = self._get_files()
        for file_path in files:
            print(f"Processing: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        content = item.get("content")
                        if content:
                            vector = self.embed_text(content)
                            if vector:
                                item["vector"] = vector
                                yield item
                    except Exception as e:
                        print(f"Skip error line: {e}")

    def save_to_local(self, filename: str = "embedded_data.jsonl"):
        """
        将 Embedding 结果持久化到本地文件夹
        """
        full_path = os.path.join(self.output_path, filename)
        count = 0
        with open(full_path, "w", encoding="utf-8") as f:
            for item in self.process_generator():
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
                if count % 10 == 0:
                    print(f"Saved {count} items to {full_path}")
        
        print(f"✅ Success: All {count} items saved to {full_path}")
        return full_path

# 使用示例
if __name__ == "__main__":
    manager = EmbedManager()
    
    # 场景 1：直接存到本地（推荐，防止断电丢失）
    # manager.save_to_local("skill_base_v1.jsonl")
    
    # 场景 2：流式同步到 LanceDB
    # for data_point in manager.process_generator():
    #     lancedb_table.add([data_point])