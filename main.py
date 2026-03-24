import uuid
from dotenv import load_dotenv

load_dotenv()

from core.agent_graph import app
from core.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("Main")

CONFIG_PATH = "config.yaml"

# ==========================================
# 2. 启动 Agent 交互逻辑
# ==========================================

def run_agent(user_input: str):
    """
    运行 Agent 处理用户请求
    """
    # 初始化状态
    # 注意：这里我们赋予每个会话一个唯一的 thread_id，方便以后做状态持久化
    thread_id = str(uuid.uuid4())[:8]
    
    initial_state = {
        "query": user_input,
        "history": [],
        "mission_plan": None,
        "current_task_index": 0,
        "internal_data": {},
        "final_answer": "",
        "errors": []
    }

    logger.info(f'\n{"="*20} 任务开始 (ID: {thread_id}) {"="*20}')
    
    # 使用流式模式运行，可以看到每个节点的产出
    for event in app.stream(
        initial_state, 
        config={"configurable": {"thread_id": thread_id}}
    ):
        for node_name, output in event.items():
            # 这里的 node_name 对应你 workflow.add_node 时定义的名称
            logger.info(f"📍 [节点进度]: {node_name} 执行完毕")
            
            # 如果是 Planner 节点，打印一下它的思考过程
            if node_name == "planner" and "mission_plan" in output:
                plan = output["mission_plan"]
                logger.info(f"🧠 AI 思考: {plan.thought}")
                logger.info(f"📋 计划任务: {[f'{t.skill_name}.{t.method}' for t in plan.tasks]}")

            # 如果执行过程中报错了
            if "errors" in output and output["errors"]:
                logger.warning(f"⚠️ 警告: {output['errors'][-1]}")

    logger.info(f'\n{"="*20} 最终回复 {"="*20}')
    # 从最后的 state 中提取结果 (app.invoke 也可以直接拿结果)
    # 这里的最终结果在 responder 节点更新到了 state 中
    # 注意：stream 模式下需要从最后的累加 state 中获取
    final_state = app.get_state({"configurable": {"thread_id": thread_id}}).values
    logger.info(f"📄 最终回答: {final_state.get('final_answer', '未生成回答')}")
    logger.info(f'{"="*50}\n')

# ==========================================
# 3. 入口函数
# ==========================================

def main():
    print("""
    ========================================
       Welcome to SkillfulRAG AI Agent 
       Author: SuZelong
    ========================================
    您可以输入指令，例如：
    - "帮我解析 data/raw/***.pdf"
    - "把解析后的文件切片并存入数据库"
    - "搜索关于SkillfulRAG的备份配置"
    (输入 'exit' 退出)
    """)

    while True:
        try:
            user_query = input("👤 用户 > ").strip()
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("Bye!")
                break
            
            if not user_query:
                continue

            run_agent(user_query)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"发生系统错误: {e}")

if __name__ == "__main__":
    main()