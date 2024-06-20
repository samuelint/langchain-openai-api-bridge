import time
from typing import List, Optional
from openai.types.beta.threads import Run
from openai.types.beta.threads.run import RequiredAction, RunStatus, AssistantTool


def create_run(
    id: str,
    assistant_id: str,
    thread_id: str,
    model: str,
    status: RunStatus,
    instructions: str = "",
    required_action: Optional[RequiredAction] = None,
    parallel_tool_calls: bool = True,
    tools: List[AssistantTool] = [],
) -> Run:
    return Run(
        id=id,
        assistant_id=assistant_id,
        thread_id=thread_id,
        model=model,
        created_at=time.time(),
        instructions=instructions,
        object="thread.run",
        parallel_tool_calls=parallel_tool_calls,
        required_action=required_action,
        status=status,
        tools=tools,
    )
