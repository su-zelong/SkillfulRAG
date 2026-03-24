import lancedb
import json
import pandas as pd
import yaml
import argparse
import os

from pathlib import Path
from typing import List, Dict, Any, Optional
from core.logger import get_logger

logger = get_logger("VectorDB")

class VectorDBManager:
    def __init__(self, config_path: str = "config.yaml", embed_dir: Optional[str] = None):
        """
        :param embed_dir: 用户输入的 embed 加载目录，优先级最高
        """
        self.config = self._load_config(config_path)
        
        # 1. 基础配置对齐
        ldb_cfg = self.config.get("VectorDB", {})
        self.db_path = ldb_cfg.get("path", "data/vector_database")
        self.table_name = ldb_cfg.get("table_name", "chunks")
        self.top_k = ldb_cfg.get("top_k", 5)
        
        # 2. 确定 Embedding 数据来源路径 (优先级：输入 > Config > 默认)
        # 假设 config 结构为 Embedding: output_path: "..."
        config_embed_path = self.config.get("Embedding", {}).get("output_path")
        
        self.embed_load_path = Path(
            embed_dir or 
            config_embed_path or 
            "data/embed"
        )
        
        logger.info(f"📍 VectorDB 数据源目录设置为: {self.embed_load_path}")

        # 确保数据库父路径存在
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[VectorDB]: ❌ 配置文件加载失败: {e}")
            return {}

    def _load_embedded_data(self, source_filename: str) -> List[Dict]:
        """
        从离线目录读取已经向量化好的 JSON 文件
        """
        # 假设文件名对齐，例如 source.pdf -> source.json
        file_path = self.embed_load_path / f"{Path(source_filename).stem}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"❌ 未找到离线向量文件: {file_path}，请先运行 EmbedManager。")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def add(self, source_file: str):
        """
        【增】从离线向量文件导入数据
        """
        db = lancedb.connect(self.db_path)
        
        # 1. 加载已经算好的向量数据
        try:
            embedded_chunks = self._load_embedded_data(source_file)
        except Exception as e:
            logger.error(e)
            return

        # 2. 格式化为 LanceDB 记录
        data = []
        for c in embedded_chunks:
            # 兼容处理：确保存在 vector 字段
            if "vector" not in c:
                logger.warning(f"[VectorDB]: ⚠️ Chunk {c.get('chunk_id')} 缺少向量数据，跳过")
                continue
                
            data.append({
                "vector": c["vector"],
                "text": c["content"],
                "chunk_id": c["chunk_id"],
                "source": c["source_file"],
                "heading": " > ".join(c["metadata"].get("heading_path", [])),
                "keywords": ", ".join(c["metadata"].get("keywords", []))
            })
        
        if not data:
            logger.error("[VectorDB]: ❌ 无有效数据可导入")
            return

        # 3. 写入数据库
        df = pd.DataFrame(data)
        if self.table_name in db.table_names():
            table = db.open_table(self.table_name)
            table.add(df)
        else:
            db.create_table(self.table_name, data=df)
            
        logger.info(f"[VectorDB]: ✅ 成功从离线文件导入 {len(data)} 条记录到表: {self.table_name}")

    def query(self, query_vector: List[float]):
        """
        【查】注意：这里的查询现在需要传入已经算好的 vector 
        因为 VectorDBManager 不再持有 API Key，不负责算向量。
        """
        db = lancedb.connect(self.db_path)
        if self.table_name not in db.table_names():
            logger.error("[VectorDB]: ❌ 数据库表不存在")
            return []

        table = db.open_table(self.table_name)
        res = table.search(query_vector).limit(self.top_k).to_pandas()
        
        # 格式化输出
        results = res[["chunk_id", "text", "heading", "_distance"]].to_dict(orient="records")
        logger.info(f"🔍 检索完成，找到 {len(results)} 条相关结果")
        return results

    def delete(self, source_name: str):
        """【删】"""
        db = lancedb.connect(self.db_path)
        table = db.open_table(self.table_name)
        table.delete(f'source = "{source_name}"')
        logger.info(f"[VectorDB]: Deleted records from: {source_name}")

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
    mgr = VectorDBManager()

    if args.action == "add": mgr.add(args.file)
    elif args.action == "query": mgr.query(args.q)
    elif args.action == "delete": mgr.delete(args.source)
    elif args.action == "update": mgr.update(args.file)