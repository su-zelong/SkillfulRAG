import lancedb
import json
import requests
import pandas as pd
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any

class LanceDBManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
        # 严格对齐你的 YAML 结构
        ldb_cfg = self.config.get("LanceDB", {})
        self.db_path = ldb_cfg.get("path", "data/vector_database")
        self.table_name = ldb_cfg.get("table_name", "chunks")
        self.top_k = ldb_cfg.get("top_k", 5)
        self.api_url = ldb_cfg.get("embedding_api")
        self.dim = ldb_cfg.get("vector_dim", 768)

        # 确保路径存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _get_vectors(self, texts: List[str]) -> List[List[float]]:
        """调用远程 IP 获取向量"""
        try:
            resp = requests.post(self.api_url, json={"input": texts}, timeout=15)
            return [item["embedding"] for item in resp.json()["data"]]
        except Exception as e:
            print(f"API Error: {e}")
            return [[0.0] * self.dim] * len(texts)

    def add(self, json_file: str):
        """【增】"""
        db = lancedb.connect(self.db_path)
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # 批量处理
        vectors = self._get_vectors([c["content"] for c in chunks])
        
        data = [{
            "vector": vectors[i],
            "text": c["content"],
            "chunk_id": c["chunk_id"],
            "source": c["source_file"],
            "heading": " > ".join(c["metadata"]["heading_path"]),
            "keywords": ", ".join(c["metadata"]["keywords"])
        } for i, c in enumerate(chunks)]
        
        if self.table_name in db.table_names():
            db.open_table(self.table_name).add(pd.DataFrame(data))
        else:
            db.create_table(self.table_name, data=pd.DataFrame(data))
        print(f"Added {len(data)} items to {self.table_name}")

    def query(self, text: str):
        """【查】"""
        db = lancedb.connect(self.db_path)
        table = db.open_table(self.table_name)
        v = self._get_vectors([text])[0]
        
        res = table.search(v).limit(self.top_k).to_pandas()
        print(res[["chunk_id", "text", "heading"]].to_json(orient="records", force_ascii=False, indent=2))

    def delete(self, source_name: str):
        """【删】"""
        db = lancedb.connect(self.db_path)
        table = db.open_table(self.table_name)
        table.delete(f'source = "{source_name}"')
        print(f"Deleted records from: {source_name}")

    def update(self, json_file: str):
        """【改】原子替换"""
        with open(json_file, 'r') as f:
            source = json.load(f)[0]["source_file"]
        self.delete(source)
        self.add(json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["add", "query", "delete", "update"])
    parser.add_argument("--file", help="JSON file path")
    parser.add_argument("--q", help="Query string")
    parser.add_argument("--source", help="Source file name")
    
    args = parser.parse_args()
    mgr = LanceDBManager()

    if args.action == "add": mgr.add(args.file)
    elif args.action == "query": mgr.query(args.q)
    elif args.action == "delete": mgr.delete(args.source)
    elif args.action == "update": mgr.update(args.file)