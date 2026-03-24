import os
from typing import List, Dict, Any, Optional
from core.registry import SkillRegistry
from core.logger import get_logger
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

logger = get_logger("Orchestrator")

# ==========================================
# 1. 定义任务协议 (Task Protocol)
# ==========================================

class TaskAction(BaseModel):
    """原子任务动作：定义要谁做、做什么、带什么参数"""
    skill_name: str = Field(description="调用的技能名称，必须是 Registry 中注册的名字，如 DocParse_manager")
    method: str = Field(description="具体执行的方法名，如 parse 或 chunk_text")
    args: Dict[str, Any] = Field(default_factory=dict, description="传递给方法的参数，需符合方法定义，优先级最高")

class MissionPlan(BaseModel):
    """完整的任务执行方案"""
    thought: str = Field(description="AI 针对用户需求的思考过程和执行逻辑说明")
    tasks: List[TaskAction] = Field(description="有序的任务执行序列")
    final_goal: str = Field(description="该计划最终预期达到的目标")

# ==========================================
# 2. 调度大脑 (Orchestrator)
# ==========================================

class Orchestrator:
    def __init__(self, registry: SkillRegistry, model_name: Optional[str] = "gpt-4-turbo", api: Optional[List[str]] = None, api_key: Optional[str] = None):
        """
        初始化调度器
        :param registry: 传入已经加载好技能的 SkillRegistry 实例
        """
        self.registry = registry
        
        # 优先级：传入参数 > 环境变量 > 默认值
        model_name = model_name or os.getenv("Orchestrator_MODEL_NAME", "gpt-4-turbo")
        api = api or os.getenv("Orchestrator_API_URL")
        api_key = api_key or os.getenv("Orchestrator_API_KEY")

        # 初始化 LLM，并强制要求结构化输出 MissionPlan
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,  # 调度需要极其稳定，严禁发散
            openai_api_base=api,
            openai_api_key=api_key
        ).with_structured_output(MissionPlan)
        
        logger.info(f"🧠 Orchestrator 初始化成功，当前决策模型: {model_name}")

    def _generate_system_prompt(self) -> str:
        """从注册中心实时获取所有技能的‘说明书’，动态组装 System Prompt"""
        skills_context = self.registry.get_full_skills_prompt()
        
        prompt = f"""
你是一个顶级 RAG 系统专家和任务调度员。
你的任务是：阅读下方的【技能规范】，根据【用户需求】拆解为一系列有序的原子任务。

### 核心规则：
1. **链路逻辑**：
   - 物理文件入库流程：DocParse_manager(parse) -> Chunk_manager(chunk_text) -> VectorDB_manager(upsert)。
   - 知识检索流程：VectorDB_manager(search) -> Rerank_manager(rerank) -> LLMManager(generate_answer)。
2. **参数优先级**：如果用户提到了特定数值（如“切片大小 500”），必须在 TaskAction 的 args 中明确体现该参数。
3. **闭环执行**：确保任务序列能够完成用户的最终目标。
4. **方法对齐**：method 字段必须严格匹配技能规范中的函数名。

### 当前系统可用技能规范：
{skills_context}
"""
        return prompt

    def make_plan(self, user_query: str) -> MissionPlan:
        """
        核心方法：将自然语言转化为 MissionPlan 对象
        """
        logger.info(f"🔍 正在为用户请求生成执行方案: {user_query}")
        
        system_msg = self._generate_system_prompt()
        
        try:
            # 调用模型生成结构化计划
            plan = self.llm.invoke([
                ("system", system_msg),
                ("human", user_query)
            ])
            
            logger.info(f"✅ 计划生成完毕，包含 {len(plan.tasks)} 个步骤。")
            logger.debug(f"思考过程: {plan.thought}")
            
            return plan
            
        except Exception as e:
            logger.error(f"❌ 任务规划失败: {str(e)}")
            # 兜底：返回一个空的或简单的计划，防止程序崩溃
            return MissionPlan(
                thought="规划失败，请检查模型连接或技能规范。",
                tasks=[],
                final_goal="Error Handling"
            )

# ==========================================
# 3. 示例用法 (测试代码)
# ==========================================
if __name__ == "__main__":
    # 模拟一个简单的 Registry 接口供测试
    class MockRegistry:
        def get_full_skills_prompt(self):
            return """
            1. DocParse_manager: 
               - method: parse(file_path, vllm=True) -> 解析PDF为MD
            2. Chunk_manager:
               - method: chunk_text(file_path, size=800) -> MD切片
            3. VectorDB_manager:
               - method: upsert(collection_name) -> 存入数据库
            """

    # 实例化
    orchestrator = Orchestrator(registry=MockRegistry())
    
    # 模拟用户请求
    test_query = "帮我把 data/paper.pdf 解析了，切片大小设置成 600，然后存进 'my_paper_db' 库里。"
    
    plan = orchestrator.make_plan(test_query)
    
    print("\n--- [AI 任务规划报告] ---")
    print(f"思考过程: {plan.thought}")
    for i, task in enumerate(plan.tasks):
        print(f"步骤 {i+1}: 调用 [{task.skill_name}] 的 [{task.method}] 方法, 参数: {task.args}")
