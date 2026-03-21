import os 
import yaml
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from skills.DocParse_manager.script.doc_parser import parse_pdf
from skills.Chunk_manager.script.chunk_process import chunk_text
from skills.VectorDB_manager.script.vector_db_process import LanceDBManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("data/pipeline.log"), logging.StreamHandler()]
)

CONFIG_PATH = "config.yaml"
RAW_PATH = "data/raw"
PROCESSED_PATH  = "data/process"
CHUNK_PATH = "data/chunk"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    raw_path = Path(RAW_PATH)
    processed_path = Path(PROCESSED_PATH)
    chunk_path = Path(CHUNK_PATH)

    for path in [raw_path, processed_path, chunk_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"---正在读取：{raw_path} 知识库下的pdf文件内容")
    raw_files = list(raw_path.glob("*.pdf"))
    logging.info(f"---扫描完成，共发现 {len(raw_files)} 个pdf 文件可供解析")

    for i, pdf_path in enumerate(raw_files):
        file_stem = pdf_path.stem
        logging.info(f"---正在解析第{i}个文档：{file_stem}")

        try:
            # 文档解析
            md_path = processed_path / f"{file_stem}.md"
            ## processed_path 文件夹下没有md文档，说明没有解析过，开始解析
            if not md_path.exists():
                parse_pdf(pdf_path)
            else:
                logging.info(f"---{md_path}已存在，跳过解析")
            
            # 切片
            json_path = chunk_path / f"{file_stem}.jsonl"
            ## json_path 存在说明切片过该文档，跳过切片
            if not json_path.exists():
                chunk_text(md_path)
            else:
                logging.info(f"---{json_path}已存在，跳过切片")
        
            # 存入知识库
            vector_manager = LanceDBManager(config_path=CONFIG_PATH)
            logging.info(f"---正在将文件：{json_path} 同步知识库")
            vector_manager.add(json_file=json_path)
            logging.info(f"---{json_path} 存入知识库完成")

        except Exception as e:
            logging.error(f"---存入文档 {file_stem} 时发生错误：{str(e)}，跳过处理")
            continue
    
    logging.info("---所有文档解析完成！")


if __name__ == "__main__":
    main()