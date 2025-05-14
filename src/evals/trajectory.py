import collections.abc
from typing import Any, Optional, List, Literal, Dict
from pydantic import BaseModel, Field, ConfigDict
from google.adk.events.event import Event


class LlmToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    call_id: Optional[str] = Field(default=None, description="Unique ID for the tool call, if provided by the LLM.")
    name: str = Field(description="The name of the function/tool to be called.")
    args: dict[str, Any] = Field(description="The arguments for the tool call.")


class ToolResult(BaseModel):
    model_config = ConfigDict(extra='forbid')
    call_id: Optional[str] = Field(default=None, description="ID of the LlmToolCall this result corresponds to, if available.")
    name: str = Field(description="The name of the function/tool that was executed.")
    result_data: dict[str, Any] = Field(description="The data returned by the tool execution.")
    is_error: bool = Field(default=False, description="True if the tool execution resulted in an error.")


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    timestamp: float = Field(description="Timestamp of the original event.")
    author: str = Field(description="'user' or the name of the agent who authored the original event.")
    role: Literal["user", "assistant", "tool"] = Field(description="The role of the entity that produced this message.")
    user_text_input: Optional[str] = Field(default=None, description="Text input from the user. (Populated if role='user')")
    assistant_text_response: Optional[str] = Field(default=None, description="Textual response from the LLM. (Populated if role='assistant')")
    assistant_tool_calls: Optional[List[LlmToolCall]] = Field(default=None, description="Tool calls requested by the LLM. (Populated if role='assistant')")
    tool_results: Optional[List[ToolResult]] = Field(default=None, description="Results of tool executions. (Populated if role='tool')")


class Trajectory(BaseModel):
    model_config = ConfigDict(extra='forbid')
    messages: List[Message]


def extract_event_metadata(event: Event) -> Dict[str, Any]:
    """Extract common metadata from an event."""
    return {
        "timestamp": event.timestamp,
        "author": event.author,
    }


def extract_text_from_parts(event: Event, exclude_function_parts: bool = True) -> List[str]:
    """Extract text parts from an event, optionally excluding function-related parts."""
    text_parts = []
    
    if not (event.content and event.content.parts):
        return text_parts
        
    for part in event.content.parts:
        has_text = part.text is not None
        is_function_part = part.function_call or part.function_response
        
        if has_text and (not exclude_function_parts or not is_function_part):
            text_parts.append(part.text)
            
    return text_parts


def process_user_message(event: Event, event_metadata: Dict[str, Any]) -> Optional[Message]:
    """Process a user event and extract a user message if present."""
    if event.author != "user":
        return None
        
    text_parts = extract_text_from_parts(event)
    
    if not text_parts:
        return None
        
    return Message(
        **event_metadata,
        role="user",
        user_text_input="".join(text_parts)
    )


def create_tool_result(function_response: Any) -> Optional[ToolResult]:
    """Create a ToolResult from a function response."""
    if function_response.name is None or function_response.response is None:
        return None
        
    is_error = (
        isinstance(function_response.response, collections.abc.Mapping) and 
        "error" in function_response.response
    )
    
    return ToolResult(
        call_id=function_response.id,
        name=function_response.name,
        result_data=function_response.response,
        is_error=is_error
    )


def process_tool_responses(event: Event, event_metadata: Dict[str, Any]) -> Optional[Message]:
    """Process tool responses from an event and create a tool message if present."""
    function_responses = event.get_function_responses()
    if not function_responses:
        return None
        
    tool_results = []
    
    for response in function_responses:
        tool_result = create_tool_result(response)
        if tool_result:
            tool_results.append(tool_result)
    
    if not tool_results:
        return None
        
    return Message(
        **event_metadata,
        role="tool",
        tool_results=tool_results
    )


def create_tool_call(function_call: Any) -> Optional[LlmToolCall]:
    """Create an LlmToolCall from a function call."""
    if function_call.name is None or function_call.args is None:
        return None
        
    return LlmToolCall(
        call_id=function_call.id,
        name=function_call.name,
        args=function_call.args
    )


def process_assistant_message(event: Event, event_metadata: Dict[str, Any]) -> Optional[Message]:
    """Process an assistant event and extract an assistant message if present."""
    if event.author == "user":
        return None
        
    text_parts = extract_text_from_parts(event)
    text_response = "".join(text_parts) if text_parts else None
    
    function_calls = event.get_function_calls()
    tool_calls = []
    
    if function_calls:
        for call in function_calls:
            tool_call = create_tool_call(call)
            if tool_call:
                tool_calls.append(tool_call)
    
    if not (text_response or tool_calls):
        return None
        
    return Message(
        **event_metadata,
        role="assistant",
        assistant_text_response=text_response,
        assistant_tool_calls=tool_calls if tool_calls else None
    )


def parse_events_to_trajectory(events: List[Event]) -> Trajectory:
    """Convert a list of Event objects into a Trajectory with structured messages."""
    trajectory_messages = []

    for event in events:
        event_metadata = extract_event_metadata(event)
        
        user_message = process_user_message(event, event_metadata)
        if user_message:
            trajectory_messages.append(user_message)
            
        tool_message = process_tool_responses(event, event_metadata)
        if tool_message:
            trajectory_messages.append(tool_message)
            
        assistant_message = process_assistant_message(event, event_metadata)
        if assistant_message:
            trajectory_messages.append(assistant_message)
                
    return Trajectory(messages=trajectory_messages)
