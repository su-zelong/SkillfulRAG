import os
import subprocess
import argparse
import yaml
import shutil
from typing import Optional
from pathlib import Path

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_pdf(file_path: Optional[str] = None, output_path: Optional[str] = None):
    # 1. 配置加载与参数对齐
    config = load_config()
    d_cfg = config.get("DocumentParse", {})
    
    # 优先级：函数参数 > YAML配置 > 硬编码默认值
    file_path = file_path or d_cfg.get("input_path", "")
    output_path = output_path or d_cfg.get("output_path", "data/process")
    
    input_file = Path(file_path)
    target_dir = Path(output_path)
    
    if not input_file.exists():
        print(f"❌ 错误：找不到输入文件 {file_path}")
        return None

    target_dir.mkdir(parents=True, exist_ok=True)
    file_stem = input_file.stem
    
    # 使用 Path 对象管理路径，避免字符串拼接产生的斜杠问题
    temp_output_path = target_dir / "temp_mineru_output"
    
    print(f"🚀 开始解析文档: {file_stem} ...")

    # 2. 构建命令
    cmd = [
        "mineru", 
        "-p", str(input_file), 
        "-o", str(temp_output_path), 
    ]
    
    # 显存优化逻辑
    if not d_cfg.get("vllm", True):
        # 注意：这里确保你的 mineru 版本支持 -b 参数，之前报错过的话建议检查 help
        cmd.extend(["-b", "pipeline"])

    try:
        # 运行并捕获输出，方便报错时排查
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # 3. 动态定位生成的 Markdown
        # MinerU 的路径规则：temp_output / {stem} / {mode} / {stem}.md
        # 优化：使用 rglob 自动查找，防止 MinerU 版本更新导致的路径层级变动
        generated_files = list(temp_output_path.rglob(f"{file_stem}.md"))
        
        if generated_files:
            generated_md = generated_files[0]
            final_path = target_dir / f"{file_stem}.md"
            
            # 移动到保存目录下
            shutil.move(str(generated_md), str(final_path))
            print(f"✨ 解析成功！Markdown 已保存至: {final_path}")
            
            # 清理临时目录
            if temp_output_path.exists():
                shutil.rmtree(temp_output_path)
            return str(final_path)
        else:
            print(f"❌ 解析失败：在 {temp_output_path} 中未找到生成的 .md 文件")
            return None

    except subprocess.CalledProcessError as e:
        print(f"💥 MinerU 执行崩溃！\n错误码: {e.returncode}\n错误输出: {e.stderr}")
    except Exception as e:
        print(f"⚠️ 解析过程发生未知错误: {e}")
    finally:
        # 如果出错临时文件是否被清理取决于配置文件 auto_clean
        if temp_output_path.exists() and d_cfg.get("auto_clean", True):
            shutil.rmtree(temp_output_path)
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="PDF 文件路径")
    args = parser.parse_args()
    parse_pdf(args.file)