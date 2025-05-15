from contextlib import AsyncExitStack
from typing import Dict, Any, List


from google.adk.tools.mcp_tool.mcp_session_manager import MCPSessionManager, StdioServerParameters
from google.adk.tools.mcp_tool import MCPTool
from mcp import ListToolsResult
from mcp.client.session import ClientSession
from mcp.types import CallToolResult


APP_NAME = "jira_mcp_agent"


class JiraMCPServer:
    """
    Manages an MCP server instance for interacting with Jira tools.
    """
    _client_session: ClientSession

    def __init__(
        self, 
        mcp_session_manager: MCPSessionManager, 
        client_session: ClientSession,
        tools: List[MCPTool],
    ):
        """
        Private constructor. Use JiraMCPServer.initialize() to create an instance.
        """

        self._mcp_session_manager = mcp_session_manager
        self._client_session = client_session
        self._tools = tools

    @classmethod
    async def initialize(
        cls,
        jira_url: str,
        jira_username: str,
        jira_api_token: str,
        enabled_tools: str,
        exit_stack: AsyncExitStack,
        container_name: str = "mcp-atlassian",
    ) -> 'JiraMCPServer':
        """
        Initializes the MCP server in a Docker container and returns an instance
        of JiraMCPServer.
        """
        docker_args=[
            "run",
            "--rm",
            "-i",
            "--name", container_name,
            "-e", "JIRA_URL",
            "-e", "JIRA_USERNAME",
            "-e", "JIRA_API_TOKEN",
            "-e", "ENABLED_TOOLS",
            "ghcr.io/sooperset/mcp-atlassian:latest",
        ]

        docker_env={
            "JIRA_URL": jira_url,
            "JIRA_USERNAME": jira_username,
            "JIRA_API_TOKEN": jira_api_token,
            "ENABLED_TOOLS": enabled_tools,
        }

        conn_params = StdioServerParameters(
            command="docker",
            args=docker_args,
            env=docker_env,
        )
        
        mcp_session_manager = MCPSessionManager(
            connection_params=conn_params,
            exit_stack=exit_stack,
        )

        client_session = await mcp_session_manager.create_session()

        tools_response: ListToolsResult = await client_session.list_tools()
        # As used in google.adk.tools.mcp_toolset `MCPToolset.load_tools()` func
        mcp_tools = [
            MCPTool(
                mcp_tool=tool,
                mcp_session=client_session,
                mcp_session_manager=mcp_session_manager,
            )
            for tool in tools_response.tools
        ]

        return cls(
            mcp_session_manager=mcp_session_manager,
            client_session=client_session,
            tools=mcp_tools,
        )
    
    def get_tools(self) -> List[MCPTool]:
        """
        Returns the list of tools available in the MCP server.

        Returns:
            A list of MCPTool instances.
        """
        return self._tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """
        Calls a tool on the connected MCP server.
        """
        if not self._client_session:
            raise RuntimeError("JiraMCPServer not initialized. Call initialize() first.")
        
        return await self._client_session.call_tool(name=name, arguments=arguments)
