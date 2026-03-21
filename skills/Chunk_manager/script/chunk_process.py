import os
import yaml
import re
import json
import argparse
import uuid
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

def extract_keywords(content: str, top_k: int = 3) -> List[str]:
    """
    结合 TF-IDF 和正则匹配，为学术 Chunk 提取高辨识度关键词
    """
    if not content or len(content.strip()) < 10:
        return ["General", "Analysis", "Paper"]

    # 1. 深度预处理：清洗 LaTeX、引用标记 [1]、多余空白
    # 去除 $...$ 公式，去除 \begin{...} 等指令，去除 [12] 这种引用
    text = re.sub(r'\$.*?\$', '', content)
    text = re.sub(r'\\.*?(?=\s|$)', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = text.strip()

    # 2. 策略组合：优先使用 TF-IDF (学术场景下比 TextRank 更能抓准领域特有词)
    # allowPOS 剔除掉普通动词 'v'，增加 'n' 系列和 'eng'
    # 因为 RAG 检索更依赖名词和专有名词
    keywords = jieba.analyse.extract_tags(
        text, 
        topK=top_k, 
        withWeight=False, 
        allowPOS=('n', 'nz', 'nw', 'vn', 'eng') 
    )

    # 3. 语义增强：针对 AIOps/医学论文，抓取大写缩写 (如 CNN, Kubernetes)
    # 论文的核心往往都在缩写里
    abbreviations = re.findall(r'\b[A-Z]{2,}\b', content)
    
    # 4. 融合与去重
    # 将缩写排在前面，然后是提取的关键词，保持顺序且去重
    final_candidates = []
    seen = set()
    
    for word in (abbreviations + keywords):
        clean_word = word.strip()
        if len(clean_word) > 1 and clean_word.lower() not in seen:
            final_candidates.append(clean_word)
            seen.add(clean_word.lower())

    # 5. 截取并兜底
    result = final_candidates[:top_k]
    
    return result if result else ["Paper", "Research", "Tech"]

# 将单个文件切分，默认保存到：data/chunk/{file_stem}.jsonl下
def chunk_text(
    file_path: Optional[str] = None, 
    user_size: Optional[int] = None, 
    document_index: Optional[int] = None,
    config_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    语义化切片 Skill：将 Markdown 按照层级和长度切分为符合 RAG 要求的 Chunks。
    """
    # 1. 动态加载配置
    final_config_path = config_path or "config.yaml"
    config = load_config(final_config_path)
    c_cfg = config.get("Chunk", {})

    # 2. 切片需要上一步解析好的 md 文件
    final_file_path = file_path or c_cfg.get("input_path", "")
    if not final_file_path or not os.path.exists(final_file_path):
        print(f"⚠️ 跳过切片：找不到目标文件 {final_file_path}")
        return []

    # 用户优先，如果用户指定切片大小优先使用用户的
    target_size = user_size if user_size is not None else c_cfg.get("chunk_size", 500)
    threshold = c_cfg.get("threshold", 0.2)
    max_size = target_size * (1 + threshold)
    overlap_size = c_cfg.get("overlap", 150)
    output_dir = Path(c_cfg.get("output_path", "data/chunk_database"))
    
    final_doc_index = document_index if document_index is not None else str(uuid.uuid4())

    # --- 开始解析 ---
    with open(final_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    
    # 采用marko解析markdown树
    doc = marko.parse(md_text)
    source_file = os.path.basename(final_file_path)
    
    chunks = []    # 所有文档切片
    current_batch = []    # 每个切片的节点
    current_len = 0    # 当前节点长度
    heading_stack = []     # 标题

    for node in doc.children:
        if isinstance(node, Heading):
            level = node.level
            # 这里的提取逻辑稍作健壮性处理
            title_text = "".join([getattr(c, 'children', str(c)) for c in node.children]).strip()
            # 更新标题栈
            heading_stack = heading_stack[:level-1]
            heading_stack.append(title_text)
            node_raw = f"{'#' * level} {title_text}\n\n"
        else:
            node_raw = marko.render(node)

        node_len = len(node_raw)

        # 弹性切分逻辑
        if (current_len + node_len > max_size) and current_batch:
            # 1. 结算当前块（传入 heading_stack 的快照）
            chunk_data = build_chunk_dict(
                current_batch, 
                source_file, 
                final_doc_index, 
                len(chunks) + 1, 
                list(heading_stack) # 快照防止引用污染
            )
            chunks.append(chunk_data)

            # 2. 真正的 Overlap 逻辑
            overlap_batch = []
            tmp_len = 0
            for item in reversed(current_batch):
                overlap_batch.insert(0, item)
                tmp_len += len(item)
                if tmp_len >= overlap_size:
                    break
            
            current_batch = overlap_batch
            current_len = tmp_len

        current_batch.append(node_raw)
        current_len += node_len

    # 处理收尾
    if current_batch:
        chunks.append(build_chunk_dict(current_batch, source_file, final_doc_index, len(chunks)+1, list(heading_stack)))

    # --- 3. 保存文件 ---
    if chunks:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{Path(source_file).stem}.jsonl"
        
        with open(save_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"✨ 切片完成: {source_file} -> {len(chunks)} chunks saved to {save_path}")

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