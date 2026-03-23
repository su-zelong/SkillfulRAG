import os
import yaml
import re
import json
import uuid
import logging
import marko
import jieba.analyse
from pathlib import Path
from typing import List, Dict, Any, Optional
from marko.block import Heading

# 获取项目统一 logger
logger = logging.getLogger("SkillfulRAG.ChunkManager")

class ChunkManager:
    def __init__(self, config_path: str = "config.yaml", **kwargs):
        """
        初始化：加载配置并支持初始化时的参数覆盖
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # 统一从 config.yaml 的 'Chunk' 字段获取
        self.c_cfg = self.config.get("Chunk", {})
        
        # 基础硬编码默认值（作为最后一道防线）
        self.defaults = {
            "chunk_size": 800,
            "overlap": 150,
            "threshold": 0.2,
            "output_path": "data/chunk",
            "input_path": ""
        }
        
        # 初始化时也允许通过 kwargs 预设一些值
        self.init_kwargs = kwargs

    def _load_config(self, path: str) -> Dict:
        try:
            if Path(path).exists():
                with open(path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    return content if content else {}
        except Exception as e:
            logger.error(f"Failed to load config at {path}: {e}")
        return {}

    def _get_param(self, key: str, user_val: Any = None) -> Any:
        """
        核心优先级逻辑实现：
        优先级：1. 方法调用参数 (user_val) > 2. 初始化参数 (init_kwargs) > 3. YAML配置 (c_cfg) > 4. 默认值 (defaults)
        """
        if user_val is not None:
            return user_val
        if key in self.init_kwargs:
            return self.init_kwargs[key]
        if key in self.c_cfg:
            return self.c_cfg[key]
        return self.defaults.get(key)

    def extract_keywords(self, content: str, top_k: int = 3) -> List[str]:
        """学术/技术场景关键词提取"""
        if not content or len(content.strip()) < 10:
            return ["General", "Tech"]

        text = re.sub(r'\$.*?\$|\\.*?(?=\s|$)|\[\d+\]', '', content).strip()
        keywords = jieba.analyse.extract_tags(text, topK=top_k, allowPOS=('n', 'nz', 'nw', 'vn', 'eng'))
        abbreviations = re.findall(r'\b[A-Z]{2,}\b', content)
        
        final = []
        seen = set()
        for word in (abbreviations + keywords):
            w = word.strip()
            if len(w) > 1 and w.lower() not in seen:
                final.append(w)
                seen.add(w.lower())
        return final[:top_k]

    def chunk_text(self, file_path: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        语义化切片主方法
        """
        # 1. 优先级参数获取 (级联查找)
        # 对应 YAML 字段：input_path, chunk_size, overlap, threshold, output_path
        input_file_path = self._get_param("input_path", file_path)
        target_size = self._get_param("chunk_size", kwargs.get("size"))
        overlap_size = self._get_param("overlap", kwargs.get("overlap"))
        threshold = self._get_param("threshold", kwargs.get("threshold"))
        output_dir = Path(self._get_param("output_path", kwargs.get("output")))
        
        # 额外参数
        doc_index = kwargs.get("index", str(uuid.uuid4())[:8])
        max_size = target_size * (1 + threshold)

        if not input_file_path or not os.path.exists(input_file_path):
            logger.error(f"❌ Target file not found: {input_file_path}")
            return []

        # 2. Markdown 解析
        with open(input_file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        
        doc = marko.parse(md_content)
        source_name = os.path.basename(input_file_path)
        
        chunks = []
        current_batch = []
        current_len = 0
        heading_stack = []

        # 3. 迭代切片逻辑
        for node in doc.children:
            if isinstance(node, Heading):
                level = node.level
                title = "".join([getattr(c, 'children', str(c)) for c in node.children]).strip()
                heading_stack = heading_stack[:level-1]
                heading_stack.append(title)
                node_raw = f"{'#' * level} {title}\n\n"
            else:
                node_raw = marko.render(node)

            node_len = len(node_raw)

            if (current_len + node_len > max_size) and current_batch:
                # 结算当前块
                chunks.append(self._build_dict(current_batch, source_name, doc_index, len(chunks)+1, heading_stack))
                
                # 重叠逻辑
                overlap_batch = []
                tmp_len = 0
                for item in reversed(current_batch):
                    overlap_batch.insert(0, item)
                    tmp_len += len(item)
                    if tmp_len >= overlap_size: break
                current_batch = overlap_batch
                current_len = tmp_len

            current_batch.append(node_raw)
            current_len += node_len

        # 收尾
        if current_batch:
            chunks.append(self._build_dict(current_batch, source_name, doc_index, len(chunks)+1, heading_stack))

        # 4. 持久化
        if chunks:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{Path(source_name).stem}.jsonl"
            with open(save_path, "w", encoding="utf-8") as f:
                for c in chunks:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            logger.info(f"✨ Chunking Success: {len(chunks)} units -> {save_path}")

        return chunks

    def _build_dict(self, blocks, source, doc_idx, c_idx, headings):
        content = "".join(blocks).strip()
        return {
            "chunk_id": f"{Path(source).stem}_{c_idx}",
            "content": content,
            "source_file": source,
            "document_index": doc_idx,
            "metadata": {
                "chunk_index": c_idx,
                "heading_path": list(headings),
                "keywords": self.extract_keywords(content)
            }
        }

if __name__ == "__main__":
    # 测试代码
    mgr = ChunkManager(config_path="config.yaml")
    # 模拟手动指定参数覆盖配置文件
    mgr.chunk_text(file_path="data/process/test.md", size=500, overlap=50)