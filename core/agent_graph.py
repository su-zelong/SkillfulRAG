from typing import TypedDict, List, Dict, Any, Optional, Literal, cast
from langgraph.graph import StateGraph, END
from core.dispatcher import Dispatcher

from core.registry import SkillRegistry
from core.orchestrator import Orchestrator, MissionPlan
from core.logger import get_logger

logger = get_logger("Graph")

_registry: Optional[SkillRegistry] = None
_orchestrator: Optional[Orchestrator] = None
_dispatcher: Optional[Dispatcher] = None


def _get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry(config_path="config.yaml")
    return _registry


def _get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(registry=_get_registry())
    return _orchestrator


def _get_dispatcher() -> Dispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = Dispatcher(_get_registry())
    return _dispatcher


class AgentState(TypedDict):
    query: str
    history: List[Dict]
    mission_plan: Optional[MissionPlan]
    current_task_index: int
    internal_data: Dict[str, Any]
    final_answer: str
    errors: List[str]


def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- [Node: Planner] 正在生成任务规划 ---")

    plan = _get_orchestrator().make_plan(state["query"])

    return cast(Dict[str, Any], {
        "mission_plan": plan,
        "current_task_index": 0,
        "errors": []
    })


def executor_node(state: AgentState) -> Dict[str, Any]:
    plan: Optional[MissionPlan] = state.get("mission_plan")
    current_idx = state["current_task_index"]
    errors = list(state.get("errors", []))

    if not plan or not plan.tasks or current_idx >= len(plan.tasks):
        return cast(Dict[str, Any], {
            "current_task_index": current_idx,
            "internal_data": state["internal_data"],
            "errors": errors + (["任务计划为空"] if not plan or not plan.tasks else [])
        })

    task = plan.tasks[current_idx]

    try:
        result = _get_dispatcher().execute_task(task, state["internal_data"])

        new_data = state["internal_data"].copy()

        if isinstance(result, str) and (result.endswith(".md") or result.endswith(".jsonl")):
            new_data["last_output_file"] = result

        return cast(Dict[str, Any], {
            "current_task_index": current_idx + 1,
            "internal_data": new_data,
            "errors": errors
        })
    except Exception as e:
        logger.error(f"❌ 任务执行失败: {task.skill_name}.{task.method} - {str(e)}")
        return cast(Dict[str, Any], {
            "current_task_index": current_idx + 1,
            "internal_data": state["internal_data"],
            "errors": errors + [f"{task.skill_name}.{task.method} 执行失败: {str(e)}"]
        })


def responder_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- [Node: Responder] 汇总任务执行结果 ---")

    mission = state.get("mission_plan")
    if mission:
        summary = f"我已经完成了您的指令：{mission.final_goal}\n"
    else:
        summary = "任务执行完成。\n"

    errors = list(state.get("errors", []))
    if errors:
        summary += f"但在过程中遇到了以下问题：{errors}"

    return cast(Dict[str, Any], {"final_answer": summary})


def should_continue(state: AgentState) -> Literal["execute", "finish", "replan"]:
    errors = state.get("errors", [])
    mission = state.get("mission_plan")

    if errors and len(errors) < 2:
        return "replan"

    if not mission or not mission.tasks:
        return "finish"

    current_idx = state["current_task_index"]
    if current_idx < len(mission.tasks):
        return "execute"

    return "finish"


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("responder", responder_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    should_continue,
    {
        "execute": "executor",
        "finish": "responder",
        "replan": "planner"
    }
)

workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "execute": "executor",
        "finish": "responder",
        "replan": "planner"
    }
)

workflow.add_edge("responder", END)

app = workflow.compile()
