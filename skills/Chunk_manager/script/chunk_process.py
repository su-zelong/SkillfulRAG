import os
import yaml
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import marko
from marko.block import Heading
import jieba.analyse

def load_config(config_path: str = "config.yaml"):
    """加载配置文件"""
    try:
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return {"Chunk": {"size": 800, "overlap": 150, "threshold": 0.2}}

def extract_keywords(content: str) -> List[str]:
    """
    使用 TextRank 算法提取 3 个语义核心关键词
    """
    # 1. 预处理：由于学术论文包含大量公式，先简单清洗掉一些 LaTeX 标记，防止干扰分词
    clean_content = re.sub(r'\$.*?\$', '', content) 
    
    # 2. 调用 TextRank 提取
    # allowPOS: 仅提取名词(n)、专有名词(nr, nz)、动词(v)、英文(eng)
    # topK: 提取前 3 个
    keywords = jieba.analyse.textrank(
        content, 
        topK=3, 
        withWeight=False, 
        allowPOS=('n', 'nz', 'v', 'eng')
    )
    
    # 3. 兜底逻辑：如果 TextRank 没提出来（内容太少），再回退到正则或默认词
    if not keywords:
        candidates = re.findall(r'\b[A-Z]{2,}\b', content)
        keywords = list(dict.fromkeys(candidates))[:3]
        
    return keywords if keywords else ["General", "Topic", "Context"]

def chunk_text(
    file_path: str, 
    user_size: Optional[int] = None, 
    document_index: int = 1,
    config_path: str = "config.yaml"
) -> List[Dict[str, Any]]:
    """
    这是核心 Skill 调用函数。
    输入：文件路径，可选的切片大小。
    输出：符合 Skill 定义的标准字典列表。
    """
    config = load_config(config_path)
    target_size = user_size or config.get("Chunk", {}).get("size", 800)
    threshold = config.get("Chunk", {}).get("threshold", 0.2)
    max_size = target_size * (1 + threshold)
    overlap_size = config.get("Chunk", {}).get("overlap", 150)

    if not os.path.exists(file_path):
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = marko.parse(md_text)
    source_file = os.path.basename(file_path)
    
    chunks = []
    current_batch = []
    current_len = 0
    heading_stack = []

    for node in doc.children:
        if isinstance(node, Heading):
            level = node.level
            title_text = "".join([c.children for c in node.children if hasattr(c, 'children')] or [str(node)]).strip()
            heading_stack = heading_stack[:level-1]
            heading_stack.append(title_text)
            node_raw = f"{'#' * level} {title_text}\n\n"
        else:
            node_raw = marko.render(node)

        node_len = len(node_raw)

        # 弹性切分逻辑
        if (current_len + node_len > max_size) and current_batch:
            # 结算
            chunks.append(build_chunk_dict(current_batch, source_file, document_index, len(chunks)+1, heading_stack))
            # 处理 Overlap (保留最后一个块以维持连贯)
            current_batch = [current_batch[-1]] if len(current_batch) > 0 else []
            current_len = sum(len(b) for b in current_batch)

        current_batch.append(node_raw)
        current_len += node_len

    if current_batch:
        chunks.append(build_chunk_dict(current_batch, source_file, document_index, len(chunks)+1, heading_stack))

    return chunks

def build_chunk_dict(blocks, source, doc_idx, chunk_idx, headings) -> Dict:
    content = "".join(blocks).strip()
    return {
        "chunk_id": f"{Path(source).stem}_{chunk_idx}.md",
        "content": content,
        "source_file": source,
        "document_index": doc_idx,
        "metadata": {
            "chunk_index": chunk_idx,
            "heading_level": len(headings),
            "heading_path": list(headings),
            "keywords": extract_keywords(content)
        }
    }

if __name__ == "__main__":
    # 提供命令行入口，方便 Skill 调度器通过 shell 调用
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--index", type=int, default=1)
    args = parser.parse_args()

    results = chunk_text(args.file, user_size=args.size, document_index=args.index)
    print(json.dumps(results, ensure_ascii=False))