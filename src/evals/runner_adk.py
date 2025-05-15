"""
Evaluation runner for the Jira MCP agent.

This module provides functionality to run evaluations on the Jira MCP agent
using predefined test cases from a CSV file.
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import AsyncExitStack
from typing import Dict

from dotenv import load_dotenv
from mcp.types import TextContent

from src.agent import JiraMcpAgent
from src.evals.load_data import EvalDataPoint, load_eval_data, load_example_dp
from src.jira_mcp_server.server import JiraMCPServer

DOCKER_CONTAINER_NAME = "mcp-atlassian"
USER_ID = "user_id"
SESSION_ID = "session_id"
DEFAULT_CSV_FILE = "eval_data.csv"
REQUIRED_ENV_VARS = ["JIRA_URL", "JIRA_USERNAME", "JIRA_API_TOKEN", "LITE_LLM_MODEL_NAME"]


def validate_environment() -> Dict[str, str]:
    """
    Validate that all required environment variables are set.
    """
    env_vars = {}
    
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Please set {var} in your environment variables.")
        env_vars[var] = value
    
    env_vars["ENABLED_TOOLS"] = os.getenv("ENABLED_TOOLS", "")
    
    return env_vars


async def stop_docker_container() -> None:
    try:
        print(f"Removing container {DOCKER_CONTAINER_NAME}...")
        subprocess.run(
            ["docker", "rm", "-f", DOCKER_CONTAINER_NAME], 
            check=True, 
            capture_output=True
        )
        print(f"Container {DOCKER_CONTAINER_NAME} removed.")
    except subprocess.CalledProcessError as e:
        # It's okay if the container doesn't exist when trying to remove it
        if "No such container" in str(e.stderr):
            print(f"Container {DOCKER_CONTAINER_NAME} does not exist. Continuing...")
        else:
            print(f"Error managing Docker container: {e}")
            raise


def load_dps_from_csv(csv_file: str) -> list[EvalDataPoint]:
    eval_data_list = load_eval_data(csv_file)
    if not eval_data_list:
        raise ValueError(f"No evaluation data found in {csv_file}. Please check the file.")
    
    return eval_data_list


async def main(csv_file: str = DEFAULT_CSV_FILE) -> None:
    load_dotenv()
    
    try:
        env_vars = validate_environment()
    except ValueError as e:
        print(f"Environment validation error: {e}")
        sys.exit(1)
    
    async with AsyncExitStack() as stack:
        mcp_server = await JiraMCPServer.initialize(
            jira_url=env_vars["JIRA_URL"],
            jira_username=env_vars["JIRA_USERNAME"],
            jira_api_token=env_vars["JIRA_API_TOKEN"],
            enabled_tools=env_vars["ENABLED_TOOLS"],
            exit_stack=stack,
            container_name=DOCKER_CONTAINER_NAME
        )
        mcp_tools = mcp_server.get_tools()
        agent = JiraMcpAgent(
            litellm_model_name=env_vars["LITE_LLM_MODEL_NAME"],
            tools=mcp_tools,
        )
        eval_data_list = load_example_dp()

        for i, eval_data in enumerate(eval_data_list):
            print(f"\nRunning evaluation {i+1}/{len(eval_data_list)}...")

            try:
                trajectory = await agent.run(prompt=eval_data.prompt)
                print(f"Agent trajectory: {trajectory}")


                if eval_data.state_validation_config:
                    for validation in eval_data.state_validation_config.state_validation_calls:
                        result = await mcp_server.call_tool(
                            name=validation.tool_name,
                            arguments=validation.arguments
                        )
                        content: TextContent = result.content
                        content_dict = json.loads(content.text)
                        is_valid = validation.validate_response(response=content_dict)

                        if eval_data.state_validation_config.fail_fast and not is_valid:
                            print(f"Validation failed for {validation.tool_name}.")
                            ## TODO: Handle fail fast logic
                            pass

                    ## TODO: Do something with the validation results

            except Exception as e:
                print(f"Error running evaluation: {e}")

        
        print("\nAll evaluations completed.\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
        # Exit cleanly due to the hanging container issue
        sys.exit(0)


if __name__ == "__main__":
    csv_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(csv_file_arg))
