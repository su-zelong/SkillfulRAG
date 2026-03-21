import os
import sys
from pathlib import Path

from skills.DocParse_manager.script.doc_parser import parse_pdf
from skills.Chunk_manager.script.chunk_process import chunk_text
from skills.VectorDB_manager.script.vector_db_process import LanceDBManager

class RAGManager:
    def __init__(self, config_path="config.yaml"):
        self.db_path = Path("data/process")
        self.chunk_path = Path("data/chunk")
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chunk_path.mkdir(parents=True, exist_ok=True)
        self.VectorDB = LanceDBManager(config_path)

    def run_pipeline(self, pdf_path, chunk_size=800):
        file_stem = Path(pdf_path).stem
        print(f"🚀 开始全链路处理: {file_stem}")

        # --- Step 1: 解析 (Doc_Parser) ---
        md_file = self.db_path / f"{file_stem}.md"
        if not md_file.exists():
            print("阶段 1: 正在解析 PDF...")
            parse_pdf(pdf_path) 
        else:
            print("阶段 1: 检测到已存在 MD，跳过解析。")

        # --- Step 2: 切片 (Chunk_manager) ---
        json_file = self.chunk_path / f"{file_stem}_chunks.json"
        if not json_file.exists():
            print(f"阶段 2: 正在生成语义切片 (Size: {chunk_size})...")
            # 假设你的 chunk 脚本返回生成的路径
            chunk_text(str(md_file), size=chunk_size)
        else:
            print("阶段 2: 检测到已存在切片 JSON，跳过切片。")

        # --- Step 3: 入库 (VectorDB_manager) ---
        print("阶段 3: 正在同步至向量数据库...")
        success = self.VectorDB.add(str(json_file))
        
        if success:
            print(f"✅ 处理完成！文档 '{file_stem}' 已就绪。")
        else:
            print(f"❌ 入库失败，请检查数据库状态。")

if __name__ == "__main__":
    manager = RAGManager()
    # 可以从命令行接收参数
    if len(sys.argv) > 1:
        target_pdf = sys.argv[1]
        manager.run_pipeline(target_pdf)
    else:
        print("用法: python pipeline_rag.py <pdf_path>")