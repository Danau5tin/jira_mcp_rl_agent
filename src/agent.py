import datetime
from typing import List, Optional 
from google.adk.agents.llm_agent import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.runners import Runner
from google.adk.tools.mcp_tool import MCPTool
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from src.evals.trajectory import Message, Trajectory, parse_events_to_trajectory

DEFAULT_APP_NAME = "jira_mcp_agent"


class JiraMcpAgent:
    """
    A class to encapsulate the creation and running of a Jira MCP LlmAgent.
    """

    def __init__(
        self,
        litellm_model_name: str,
        tools: List[MCPTool],
        agent_name: str = "JiraMCPAgent",
        app_name: str = DEFAULT_APP_NAME,
        session_service: Optional[BaseSessionService] = None,
    ):
        """
        Initializes the JiraMcpAgentRunner.

        Args:
            litellm_model_name: The name of the LiteLLM model to use.
            tools: A list of MCPTools for the agent.
            agent_name: The name for the LlmAgent.
            app_name: The application name.
            session_service: An optional SessionService instance.
                             If None, an InMemorySessionService will be created.
        """
        self.app_name = app_name
        self.agent_name = agent_name
        self.litellm_model_name = litellm_model_name
        self.tools = tools

        self.model = LiteLlm(model=self.litellm_model_name)
        self.agent = LlmAgent(
            model=self.model,
            name=self.agent_name,
            tools=self.tools,
        )

        if session_service is None:
            self.session_service = InMemorySessionService()
        else:
            self.session_service = session_service

        self.runner = Runner(
            agent=self.agent,
            app_name=self.app_name,
            session_service=self.session_service
        )


    async def run(
        self, prompt: str, user_id: str = "user_id", session_id: str = "session_id"
    ) -> Trajectory:
        """
        Runs the agent with the given prompt and session information.

        Args:
            prompt: The user's input prompt.
            user_id: The ID of the user.
            session_id: The ID of the session.

        Returns:
            A Trajectory object representing the agent's execution.
        """
        self.session_service.create_session(
            app_name=self.app_name, user_id=user_id, session_id=session_id
        )

        timestamp_usr_msg = datetime.datetime.now().timestamp()
        user_msg = Message(
            author="user",
            role="user",
            user_text_input=prompt,
            timestamp=timestamp_usr_msg,
        )

        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        new_events = []
        async for event in self.runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            new_events.append(event)

        trajectory = parse_events_to_trajectory(new_events)
        trajectory.messages.insert(0, user_msg)

        return trajectory