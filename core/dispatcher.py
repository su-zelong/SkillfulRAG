from typing import Any, Dict, Optional
from core.logger import get_logger

logger = get_logger("Dispatcher")

class Dispatcher:
    def __init__(self, registry):
        """
        初始化分发器
        :param registry: 已经初始化并加载了所有技能实例的 SkillRegistry
        """
        self.registry = registry

    def execute_task(self, task: Any, internal_data: Dict[str, Any]) -> Any:
        """
        核心执行逻辑：动态分发指令到具体的 Manager
        :param task: Orchestrator 生成的 TaskAction 对象 (包含 skill_name, method, args)
        :param internal_data: AgentState 中的数据总线，用于处理任务间的上下文依赖
        """
        skill_name = task.skill_name
        method_name = task.method
        task_args = task.args.copy()  # 保护原始参数不被篡改

        logger.info(f"🚀 [Dispatcher] 准备执行: {skill_name} -> {method_name}")

        # 1. 从注册中心获取实例
        instance = self.registry.instances.get(skill_name)
        if not instance:
            raise ValueError(f"未找到技能实例: {skill_name}。请检查 skills 目录名是否正确。")

        # 2. 上下文参数注入 (关键逻辑！)
        # 如果上一个任务产出了文件路径，而当前任务没给路径，自动从 internal_data 补全
        self._inject_context_args(task_args, internal_data)

        # 3. 检查方法是否存在
        if not hasattr(instance, method_name):
            raise AttributeError(f"技能 {skill_name} 中不存在方法: {method_name}")

        # 4. 动态调用
        try:
            method = getattr(instance, method_name)
            
            # 这里的执行会自动应用你写的 "kwargs > config" 优先级逻辑
            result = method(**task_args)
            
            logger.info(f"✅ [Dispatcher] {skill_name} 执行成功")
            return result
            
        except Exception as e:
            logger.error(f"❌ [Dispatcher] 执行 {skill_name}.{method_name} 时崩溃: {str(e)}")
            raise e

    def _inject_context_args(self, args: Dict, internal_data: Dict):
        """
        数据流水线逻辑：自动将上一步的产出注入到下一步的输入
        """
        # 典型场景：DocParse 输出了路径，Chunk 需要这个路径作为 file_path
        if "file_path" not in args or not args["file_path"]:
            # 尝试从 internal_data 中寻找最近一次产出的路径
            last_path = internal_data.get("last_output_file")
            if last_path:
                args["file_path"] = last_path
                logger.debug(f"🔗 自动注入上下文参数 file_path: {last_path}")