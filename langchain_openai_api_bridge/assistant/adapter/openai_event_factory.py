import json
import time
from typing import Any, Literal, Optional, Union
from openai.types.beta.assistant_stream_event import (
    ThreadMessageCreated,
    ThreadMessageDelta,
    MessageDeltaEvent,
    ThreadMessageCompleted,
    ThreadRunStepCreated,
    ThreadRunStepCompleted,
    ThreadRunStepDelta,
    RunStep,
)
from langchain_core.messages.tool import ToolMessage

from openai.types.beta.threads import (
    Message,
)
from openai.types.beta.threads.runs import (
    FunctionToolCall,
    ToolCallsStepDetails,
)
from openai.types.beta.threads.runs import function_tool_call

from langchain_openai_api_bridge.assistant.adapter.openai_message_factory import (
    FromLanggraphMessageChunkContent,
    create_text_message_delta,
)
from pydantic import BaseModel


def create_thread_message_created_event(message: Message) -> ThreadMessageCreated:
    return ThreadMessageCreated(
        event="thread.message.created",
        data=message,
    )


def create_text_thread_message_delta(
    message_id: str, content: FromLanggraphMessageChunkContent, role: str
) -> ThreadMessageDelta:
    return ThreadMessageDelta(
        event="thread.message.delta",
        data=MessageDeltaEvent(
            id=message_id,
            delta=create_text_message_delta(content=content, role=role),
            object="thread.message.delta",
        ),
    )


def create_thread_message_completed(
    message: Message,
) -> ThreadMessageCompleted:

    return ThreadMessageCompleted(
        event="thread.message.completed",
        data=message,
    )


def create_langchain_tool_run_step_tools_created(
    step_id: str,
    assistant_id: str,
    thread_id: str,
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"],
    metadata: Optional[dict[str, Any]] = None,
    name: Optional[str] = None,
    arguments: Optional[dict[object]] = None,
    output: Optional[str] = None,
) -> ThreadRunStepDelta:
    metadata = {key: str(value) for key, value in metadata.items()}
    return ThreadRunStepCreated(
        event="thread.run.step.created",
        data=create_langchain_tool_run_step(
            step_id=step_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            status=status,
            metadata=metadata,
            name=name,
            arguments=arguments,
            output=output,
        ),
    )


def create_langchain_tool_thread_run_step_completed(
    step_id: str,
    assistant_id: str,
    thread_id: str,
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"],
    metadata: Optional[dict[str, Any]] = None,
    name: Optional[str] = None,
    arguments: Optional[Union[dict[object], float, str]] = None,
    output: Optional[Union[dict[object], float, str]] = None,
) -> ThreadRunStepCompleted:
    metadata = {key: str(value) for key, value in metadata.items()}
    return ThreadRunStepCompleted(
        event="thread.run.step.completed",
        data=create_langchain_tool_run_step(
            step_id=step_id,
            assistant_id=assistant_id,
            thread_id=thread_id,
            status=status,
            metadata=metadata,
            name=name,
            arguments=arguments,
            output=output,
        ),
    )


def create_langchain_tool_run_step(
    step_id: str,
    assistant_id: str,
    thread_id: str,
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"],
    metadata: Optional[dict[str, str] | BaseModel] = None,
    name: Optional[str] = None,
    arguments: Optional[Union[dict[object], float, str]] = None,
    output: Optional[Union[dict[object], float, str]] = None,
) -> RunStep:
    metadata = metadata.model_dump() if isinstance(metadata, BaseModel) else metadata
    return RunStep(
        id=step_id,
        assistant_id=assistant_id,
        created_at=int(time.time()),
        metadata=metadata,
        object="thread.run.step",
        run_id=step_id,
        status=status,
        step_details=ToolCallsStepDetails(
            type="tool_calls",
            tool_calls=[
                create_langchain_tool_tool_call(
                    id=step_id,
                    name=name,
                    arguments=arguments,
                    output=output,
                )
            ],
        ),
        thread_id=thread_id,
        type="tool_calls",
    )


def create_langchain_tool_tool_call(
    id: str,
    name: Optional[str] = "",
    arguments: Optional[Union[dict[object], float, str]] = None,
    output: Optional[Union[dict[object], float, str]] = None,
) -> FunctionToolCall:

    function = create_langchain_function(name=name, arguments=arguments, output=output)
    return function_tool_call.FunctionToolCall(
        id=id,
        function=function,
        type="function",
    )


def create_langchain_function(
    name: Optional[str] = "",
    arguments: Optional[Union[dict[object], float, str]] = None,
    output: Optional[Union[dict[object], float, str]] = None,
) -> function_tool_call.Function:
    arguments_json = json.dumps(arguments) if arguments else None

    output_json = _serialize_output(output=output)

    return function_tool_call.Function(
        name=name, arguments=arguments_json, output=output_json
    )


def _serialize_output(output: Optional[Union[dict[object], float, str]] = None):
    if output is None:
        return None

    if isinstance(output, ToolMessage):
        output_obj = {
            "content": output.content,
            "tool_call_id": output.tool_call_id,
            "status": output.status,
        }
        if output.artifact is not None:
            output_obj["artifact"] = output.artifact
        return json.dumps(output_obj)

    return json.dumps(output) if output else None
