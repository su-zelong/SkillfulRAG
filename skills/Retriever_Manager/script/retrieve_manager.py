import os 
import yaml
import json
import sys

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

class RetrieverManager:
    def __init__(self, 
                 config_path: str = "config.yaml",
                 **kwargs):
        
        config = load_config(config_path)
        ret_cfg = config.get("Retriever", {})
        
        # 优先级：kwargs > config.yaml > 环境变量 > 默认值
        self.url = kwargs.get("url") or os.getenv("RETRIEVER_API") or ret_cfg.get("url")
        self.api_key = kwargs.get("api_key") or os.getenv("RETRIEVER_API_KEY")