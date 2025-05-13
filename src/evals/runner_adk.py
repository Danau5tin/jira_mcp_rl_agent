import asyncio
import os
import subprocess
import sys

from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

USER_QUERY = "Can you get me MBA-6 please, and let me know what the title is?"

DOCKER_CONTAINER_NAME = "mcp-atlassian"
APP_NAME = "jira_mcp_agent"
USER_ID = "user_id"
SESSION_ID = "session_id"

async def main():
    jira_url = os.getenv("JIRA_URL")
    jira_username = os.getenv("JIRA_USERNAME")
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    enabled_tools = os.getenv("ENABLED_TOOLS")
    if not jira_url or not jira_username or not jira_api_token:
        raise ValueError("Please set JIRA_URL, JIRA_USERNAME, and JIRA_API_TOKEN in your environment variables.")
    
    litellm_model_name = os.getenv("LITE_LLM_MODEL_NAME")
    if not litellm_model_name:
        raise ValueError("Please set LITE_LLM_MODEL_NAME in your environment variables.")
    
    subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER_NAME], check=True, capture_output=True)
    print("Removed existing container.")
    
    tools, _ = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",
                "-i",
                "--name", DOCKER_CONTAINER_NAME,
                "-e", "JIRA_URL",
                "-e", "JIRA_USERNAME",
                "-e", "JIRA_API_TOKEN",
                "-e", "ENABLED_TOOLS",
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

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service
    )
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    message = types.Content(role="user", parts=[types.Part(text=USER_QUERY)])
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=message):
        if event.is_final_response():
            final_msg = event.content.parts[0].text
            print(f"Final response: {final_msg}")

    # Due to a problem with a hanging container, we need to remove it here and hard exit.
    # Bug report: https://github.com/sooperset/mcp-atlassian/issues/421
    print("Removing container...")
    print("\n\n\n\n*****\n\nError to follow due to hard exit\n\n*****\n\n\n\n")
    subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER_NAME], check=True, capture_output=True)
    sys.exit(0)



if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(main())
