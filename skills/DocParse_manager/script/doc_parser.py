import os
import subprocess
import argparse
import yaml
import shutil
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_pdf(file_path: str):
    config = load_config()
    # 根据你的 config.yaml 逻辑确定输出位置
    # 假设 MD 数据库与 Chunk 输出路径同级
    target_dir = Path("data/md_database")
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始解析文档: {file_path} ...")
    
    # 调用 magic-pdf 命令行工具
    # -p: 指定文件, -o: 指定输出目录, -m: 模式 (auto/ocr/txt)
    try:
        cmd = [
            "magic-pdf", 
            "-p", file_path, 
            "-o", "data/temp_output", # 先解析到临时目录
            "-m", "auto"
        ]
        subprocess.run(cmd, check=True)

        # MinerU 会生成一个以文件名为名的文件夹，里面包含 md
        file_stem = Path(file_path).stem
        generated_md = Path("data/temp_output") / file_stem / f"{file_stem}.md"
        
        if generated_md.exists():
            final_path = target_dir / f"{file_stem}.md"
            shutil.copy(generated_md, final_path)
            print(f"解析成功！Markdown 已保存至: {final_path}")
            
            # 清理临时文件
            shutil.rmtree("data/temp_output")
        else:
            print("解析失败：未找到生成的 Markdown 文件。")

    except subprocess.CalledProcessError as e:
        print(f"MinerU 执行出错: {e}")
    except Exception as e:
        print(f"解析过程发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="PDF 文件路径")
    args = parser.parse_args()
    parse_pdf(args.file)