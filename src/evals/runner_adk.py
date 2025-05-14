import asyncio
import os
import subprocess
import sys
from google.genai import types
from dotenv import load_dotenv

from load_data import load_eval_data
from run_agent import run_agent, create_agent

USER_QUERY = "Can you get me MBA-6 please, and let me know what the title is?"
DOCKER_CONTAINER_NAME = "mcp-atlassian"
USER_ID = "user_id"
SESSION_ID = "session_id"
CSV_FILE = "eval_data.csv"


async def main():
    load_dotenv()

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

    # Load eval data
    eval_data_list = await load_eval_data(CSV_FILE)
    if not eval_data_list:
        raise ValueError(f"No eval data found in {CSV_FILE}. Please check the file.")

    agent = await create_agent(litellm_model_name, jira_url, jira_username, jira_api_token, enabled_tools)

    # Use the first prompt for now
    first_eval_data = eval_data_list[0]
    message = types.Content(role="user", parts=[types.Part(text=first_eval_data.prompt)])

    await run_agent(agent, USER_ID, SESSION_ID, message)

    # Due to a problem with a hanging container, we need to remove it here and hard exit.
    # Bug report: https://github.com/sooperset/mcp-atlassian/issues/421
    print("Removing container...")
    print("\n\n\n\n*****\n\nError to follow due to hard exit\n\n*****\n\n\n\n")
    subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER_NAME], check=True, capture_output=True)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
