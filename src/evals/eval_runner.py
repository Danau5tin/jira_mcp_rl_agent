import json

from mcp.types import TextContent

from src.jira_mcp_server.server import JiraMCPServer
from src.agent import JiraMcpAgent
from src.evals.load_data import NewEvalDataPoint


async def run_single_eval(
        eval_dp: NewEvalDataPoint,
        agent: JiraMcpAgent,
        mcp_server: JiraMCPServer,
):
    trajectory = await agent.run(prompt=eval_dp.prompt)
    print(f"Agent trajectory: {trajectory}")


    if eval_dp.state_validation_config:
        for validation in eval_dp.state_validation_config.state_validation_calls:
            result = await mcp_server.call_tool(
                name=validation.tool_name,
                arguments=validation.arguments
            )
            content: TextContent = result.content
            content_dict = json.loads(content.text)
            is_valid = validation.validate_response(response=content_dict)

            if eval_dp.state_validation_config.fail_fast and not is_valid:
                print(f"Validation failed for {validation.tool_name}.")
                ## TODO: Handle fail fast logic
                pass

        ## TODO: Do something with the validation results




async def run_evals(
        agent: JiraMcpAgent,
        mcp_server: JiraMCPServer,
        eval_data_list: list[NewEvalDataPoint],
):
    for i, eval_data in enumerate(eval_data_list):
        print(f"\nRunning evaluation {i+1}/{len(eval_data_list)}...")
        try:
            await run_single_eval(
                eval_dp=eval_data,
                agent=agent,
                mcp_server=mcp_server
            )
        
        except Exception as e:
            print(f"Error running evaluation {i+1}: {e}")
            continue

        