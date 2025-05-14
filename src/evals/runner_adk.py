"""
Evaluation runner for the Jira MCP agent.

This module provides functionality to run evaluations on the Jira MCP agent
using predefined test cases from a CSV file.
"""

import asyncio
import os
import subprocess
import sys
from typing import Dict, Literal, Optional
from dotenv import load_dotenv

from load_data import load_eval_data, EvalDataPoint
from run_agent import create_agent, run_agent

DOCKER_CONTAINER_NAME = "mcp-atlassian"
USER_ID = "user_id"
SESSION_ID = "session_id"
DEFAULT_CSV_FILE = "eval_data.csv"
REQUIRED_ENV_VARS = ["JIRA_URL", "JIRA_USERNAME", "JIRA_API_TOKEN", "LITE_LLM_MODEL_NAME"]


def validate_environment() -> Dict[str, str]:
    """
    Validate that all required environment variables are set.
    
    Returns:
        Dict[str, str]: Dictionary containing the validated environment variables.
    
    Raises:
        ValueError: If any required environment variable is missing.
    """
    env_vars = {}
    
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value:
            raise ValueError(f"Please set {var} in your environment variables.")
        env_vars[var] = value
    
    env_vars["ENABLED_TOOLS"] = os.getenv("ENABLED_TOOLS", "")
    
    return env_vars


async def manage_docker_container(action: Literal["stop"]) -> None:
    """
    Manage the Docker container (start or stop).
    
    Args:
        action (str): The action to perform ('start' or 'stop').
    """
    try:
        if action == "stop":
            print(f"Removing container {DOCKER_CONTAINER_NAME}...")
            subprocess.run(
                ["docker", "rm", "-f", DOCKER_CONTAINER_NAME], 
                check=True, 
                capture_output=True
            )
            print(f"Container {DOCKER_CONTAINER_NAME} removed.")
    except subprocess.CalledProcessError as e:
        # It's okay if the container doesn't exist when trying to remove it
        if action == "stop" and "No such container" in str(e.stderr):
            print(f"Container {DOCKER_CONTAINER_NAME} does not exist. Continuing...")
        else:
            print(f"Error managing Docker container: {e}")
            raise


async def run_evaluation(
    eval_data: EvalDataPoint, 
    model_name: str,
    jira_url: str,
    jira_username: str,
    jira_api_token: str,
    enabled_tools: str
) -> None:
    """
    Run a single evaluation with the given data.
    
    Args:
        eval_data (EvalDataPoint): The evaluation data point to use.
        model_name (str): The LiteLLM model name to use.
        jira_url (str): The Jira URL.
        jira_username (str): The Jira username.
        jira_api_token (str): The Jira API token.
        enabled_tools (str): Comma-separated list of enabled tools.
    """
    print(f"Creating agent for evaluation with prompt: {eval_data.prompt[:50]}...")
    agent = await create_agent(
        model_name, 
        jira_url, 
        jira_username, 
        jira_api_token, 
        enabled_tools
    )
    
    print("Running agent...")
    await run_agent(agent, USER_ID, SESSION_ID, eval_data.prompt)
    print("Agent run completed.")


async def main(csv_file: Optional[str] = None) -> None:
    """
    Main function to run the evaluation.
    
    Args:
        csv_file (Optional[str]): Path to the CSV file containing evaluation data.
            If not provided, uses the default CSV file.
    """
    load_dotenv()
    
    try:
        env_vars = validate_environment()
    except ValueError as e:
        print(f"Environment validation error: {e}")
        sys.exit(1)
    
    csv_file = csv_file or DEFAULT_CSV_FILE
    
    try:       
        eval_data_list = await load_eval_data(csv_file)
        if not eval_data_list:
            raise ValueError(f"No evaluation data found in {csv_file}. Please check the file.")
        print(f"Loaded {len(eval_data_list)} evaluation data points.")
        
        for i, eval_data in enumerate(eval_data_list):
            await manage_docker_container("stop")

            print(f"\nRunning evaluation {i+1}/{len(eval_data_list)}...")
            try:
                await run_evaluation(
                    eval_data,
                    env_vars["LITE_LLM_MODEL_NAME"],
                    env_vars["JIRA_URL"],
                    env_vars["JIRA_USERNAME"],
                    env_vars["JIRA_API_TOKEN"],
                    env_vars["ENABLED_TOOLS"]
                )
            except Exception as e:
                print(f"Error running evaluation: {e}")
            finally:
                # Due to a problem with a hanging container, we need to remove it after each run
                # Bug report: https://github.com/sooperset/mcp-atlassian/issues/421
                await manage_docker_container("stop")
        
        print("\nAll evaluations completed.")
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        await manage_docker_container("stop")
    
    # Exit cleanly due to the hanging container issue
    sys.exit(0)


if __name__ == "__main__":
    csv_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(csv_file_arg))
