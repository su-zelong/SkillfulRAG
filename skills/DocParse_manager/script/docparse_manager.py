import os
import subprocess
import yaml
import shutil
import logging
from typing import Optional, Any
from pathlib import Path

# 接入项目统一的日志系统
logger = logging.getLogger("SkillfulRAG.PDFSkill")

class DocParseManager:
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化时加载配置文件，作为第二优先级
        """
        self.config = self._load_config(config_path)
        self.d_cfg = self.config.get("DocumentParse", {})

    def _load_config(self, path: str):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            return {}
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return {}

    def parse(self, file_path: Optional[str] = None, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        核心解析方法
        优先级实现 kwargs > config (self.d_cfg) > 默认值
        """
        # 1. 动态优先级参数获取逻辑
        # 使用 kwargs.get(key, self.d_cfg.get(key, default)) 实现三级覆盖
        
        target_file_str = file_path or self.d_cfg.get("input_path", "data/raw")
        target_output_str = output_path or self.d_cfg.get("output_path", "data/process")
        
        # 显存/加速器逻辑优先级
        use_vllm = kwargs.get("vllm", self.d_cfg.get("vllm", True))
        auto_clean = kwargs.get("auto_clean", self.d_cfg.get("auto_clean", True))
        
        # 2. 路径预处理
        input_file = Path(target_file_str)
        target_dir = Path(target_output_str)
        
        if not input_file.exists():
            logger.error(f"❌ 错误：找不到输入文件 {target_file_str}")
            return None

        target_dir.mkdir(parents=True, exist_ok=True)
        file_stem = input_file.stem
        temp_output_path = target_dir / "temp_mineru_output"
        
        logger.info(f"🚀 开始解析文档: {file_stem} (vLLM={use_vllm})")

        # 3. 构建命令 (严格保持原 mineru 命令结构)
        cmd = [
            "mineru", 
            "-p", str(input_file), 
            "-o", str(temp_output_path), 
        ]
        
        if not use_vllm:
            cmd.extend(["-b", "pipeline"])

        # 4. 执行解析
        try:
            # 运行并捕获输出
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # 5. 动态定位并移动结果
            generated_files = list(temp_output_path.rglob(f"{file_stem}.md"))
            
            if generated_files:
                generated_md = generated_files[0]
                final_path = target_dir / f"{file_stem}.md"
                
                # 覆盖式移动
                if final_path.exists():
                    final_path.unlink()
                
                shutil.move(str(generated_md), str(final_path))
                logger.info(f"✨ 解析成功！Markdown 已保存至: {final_path}")
                
                # 成功后清理临时目录
                if temp_output_path.exists():
                    shutil.rmtree(temp_output_path)
                return str(final_path)
            else:
                logger.error(f"❌ 解析失败：未找到生成的 .md 文件")
                return None

        except subprocess.CalledProcessError as e:
            logger.error(f"💥 MinerU 执行崩溃！错误码: {e.returncode}\n输出: {e.stderr}")
        except Exception as e:
            logger.exception(f"⚠️ 解析过程发生未知错误: {e}")
        finally:
            # 最终清理逻辑
            if temp_output_path.exists() and auto_clean:
                shutil.rmtree(temp_output_path)
                logger.debug("临时目录已清理")
        
        return None

if __name__ == "__main__":
    # 模拟外部调用
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="PDF 文件路径")
    args = parser.parse_args()
    
    # 示例：通过 kwargs 覆盖 config 中的 vllm 设置
    pdf_mgr = PDFManager()
    pdf_mgr.parse(args.file, vllm=False)