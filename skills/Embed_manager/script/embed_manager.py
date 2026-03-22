import yaml
import json
import os
import sys
import requests
from typing import Optional, List, Dict, Any, Generator

def load_config(config_path: str = "config.yaml"):
    # 增加文件存在校验
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f: 
        # 建议使用 safe_load 防止 YAML 注入风险
        config = yaml.safe_load(f)
    return config

class Embed_Manager:
    def __init__(self, 
                 dim: Optional[int] = None, 
                 chunk_path: Optional[str] = None, 
                 config_path: str = "config.yaml",
                 url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        
        config = load_config(config_path)
        
        # 优先级：参数 > 配置文件 > 默认值
        self.dim = dim or config.get("Embedding", {}).get("dim", 768)
        self.chunk_path = chunk_path or config.get("Embedding", {}).get("input_path", "data/chunk")
        
        # 优先级：参数 > 环境变量 
        self.url = url or os.getenv("EMBEDDING_API")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.mode_name = model_name or os.getenv("EMBEDDING_MODEL_NAME")
        
        if not (self.url and self.api_key):
            print(f"Error: API URL or Key missing! URL: {self.url}")
            sys.exit(1)

    def embed(self, text: str) -> List[float]:
        """调用远程 API 生成向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": text,
            "model": self.model_name
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            # 兼容 OpenAI 标准格式 result['data'][0]['embedding']
            return result.get("data", [{}])[0].get("embedding", [])
        except Exception as e:
            print(f"Embedding API error: {e}")
            return []

    def _read_file(self, path: str) -> Generator[Dict[str, Any], None, None]:
        """逐行读取 JSONL 文件并解析"""
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 注意：json.loads 用于字符串，json.load 用于文件对象
                    data = json.loads(line)
                    yield data
                except json.JSONDecodeError as e:
                    print(f"Skip line {line_no} in {path}: Invalid JSON")

    def batch_process_to_lancedb(self) -> List[Dict[str, Any]]:
        """
        [业务逻辑] 遍历 chunk 目录，生成带向量的数据
        可以直接把这个返回值丢给 LanceDB Skill
        """
        all_data = []
        # 如果 chunk_path 是目录，遍历里面的文件
        if os.path.isdir(self.chunk_path):
            files = [os.path.join(self.chunk_path, f) for f in os.listdir(self.chunk_path) if f.endswith('.jsonl')]
        else:
            files = [self.chunk_path]

        for file_path in files:
            for item in self._read_file(file_path):
                content = item.get("content")
                if content:
                    vector = self.embed(content)
                    if vector:
                        item["vector"] = vector
                        all_data.append(item)
        return all_data

# 使用示例
if __name__ == "__main__":
    manager = Embed_Manager()
    # data_ready_for_lancedb = manager.batch_process_to_lancedb()