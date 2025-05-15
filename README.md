# Jira MCP Agent - trained via multi-turn RL

## Datasets
A single data point in the dataset consists of:
- [TaskContext](./data/dataset_entities/task_context.py), which consists of:
    - The goal of the task at hand
    - The initial message from the user to the agent
- [State Validation](./data/dataset_entities/state_validation.py), which consists of API calls to make after the agent is finished and their expected results.

## Project details

**Example Jira MCP I/O**
For reference, the files found in the [example_io dir](./workspace/example_io_jira_mcp_server/) show the available tools (in JSON schema) which are provided to the agent and their sample outputs as returned by the MCP server.