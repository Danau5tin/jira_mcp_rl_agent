from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

APP_NAME = "jira_mcp_agent"


async def create_agent(
    litellm_model_name: str,
    jira_url: str,
    jira_username: str,
    jira_api_token: str,
    enabled_tools: str,
) -> LlmAgent:
    """Creates an LlmAgent with the given configuration."""
    tools, _ = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",
                "-i",
                "--name",
                "mcp-atlassian",
                "-e",
                "JIRA_URL",
                "-e",
                "JIRA_USERNAME",
                "-e",
                "JIRA_API_TOKEN",
                "-e",
                "ENABLED_TOOLS",
                "ghcr.io/sooperset/mcp-atlassian:latest",
            ],
            env={
                "JIRA_URL": jira_url,
                "JIRA_USERNAME": jira_username,
                "JIRA_API_TOKEN": jira_api_token,
                "ENABLED_TOOLS": enabled_tools,
            },
        )
    )

    agent = LlmAgent(
        model=LiteLlm(model=litellm_model_name),
        name="JiraMCPAgent",
        tools=tools,
    )
    return agent


async def run_agent(
    agent: LlmAgent, user_id: str, session_id: str, prompt: str
):
    """Runs the agent with the given message."""
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )

    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=content
    ):
        if event.is_final_response():
            final_msg = event.content.parts[0].text
            print(f"Final response: {final_msg}")
