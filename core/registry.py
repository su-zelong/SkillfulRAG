import os 
import sys
import importlib
from pathlib import Path
from typing import Optional, Dict, Any
from core.logger import get_logger

logger = get_logger("Registry")

class SkillRegistry:
    def __init__(self, config_path: Optional[str] = None, skills_path: Optional[str] = None):
        self.skills_path = Path(skills_path or "skills")
        self.config_path = config_path or "config.yaml"
        self.instances: Dict[str, Any] = {}
        self.skill_prompts: Dict[str, str] = {}
        # 自动注册技能实例
        logger.info("sys path before import: " + str(sys.path))
        self._register_skills()

    def _register_skills(self):
        """自动扫描 skills 目录，加载技能描述和实例"""
        for skill_dir in self.skills_path.iterdir():
            logger.info(f"Scanning for skill in: {skill_dir}")
            if not skill_dir.is_dir():
                continue
            skill_name = skill_dir.name
            skill_path = skill_dir / "skill.md"
            logger.info(f"🔍 发现技能目录: {skill_name}")
            if skill_path.exists():
                self.skill_prompts[skill_name] = skill_path.read_text(encoding="utf-8")

            try:
                module_path = f"skills.{skill_name}.script.{skill_name.lower()}"
                logger.info(f"✅ 加载技能模块: {module_path}")
                module = importlib.import_module(module_path)

                class_name = skill_name.replace("_", "")
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.instances[skill_name] = cls(self.config_path)
                    logger.info(f"✅ 自动注册技能实例: {skill_name}")
            except Exception as e:
                logger.warning(f"⚠️ 技能实例 {skill_name} 加载跳过或失败: {e}")

    def get_full_skills_prompt(self) -> str:
        """将所有技能的描述整合成一个字符串，供 LLM 理解整体能力边界和调用规范"""
        all_prompts = "\n\n 所有技能如下："
        for name, desc in self.skill_prompts.items():
            all_prompts += f"\n\n---\n\n【{name}技能规范】：\n{desc}"
        return all_prompts
 
    def dispatch(self, skill_name: str, **kwargs) -> Any:
        """根据技能名称调用对应实例的 process 方法，传入参数"""
        if skill_name in self.instances:
            instance = self.instances[skill_name]
            if hasattr(instance, "process"):
                return instance.process(**kwargs)
            else:
                raise AttributeError(f"技能 {skill_name} 实例缺少 process 方法")
        else:
            raise ValueError(f"未找到技能实例: {skill_name}")
    
    def find_best_batch(self, description: str) -> Optional[str]:
        """根据描述匹配最合适的技能实例，返回技能名称"""
        best_match = None
        best_score = 0.0
        # 简单占位实现：若有实现，可改为实际相似度计算
        for name, prompt in self.skill_prompts.items():
            # 简易包含判断，避免额外依赖
            if description and description.lower() in prompt.lower():
                return name
            # 可以后续替换为真正的相似度计算
        return best_match
 
if __name__ == "__main__":
    register = SkillRegistry()
    logger.info("当前注册的技能实例：")
    logger.info(register.get_full_skills_prompt())
