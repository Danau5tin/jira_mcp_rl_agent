import csv
from dataclasses import dataclass
import os
from typing import List

from src.data.dataset_entities.state_validation import ApiCallValidation, StateValidationConfig
from src.data.dataset_entities.task_context import TaskContext

@dataclass
class EvalDataPoint:
    prompt: str
    expected_tools: List[str]
    final_msg_facts: str

async def load_eval_data(csv_file: str) -> List[EvalDataPoint]:
    """Loads eval data from a CSV file."""
    eval_data_list: List[EvalDataPoint] = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, csv_file)
    with open(csv_file, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            eval_data = EvalDataPoint(
                prompt=row['prompt'],
                expected_tools=row['expected_tools'].split(','),
                final_msg_facts=row['final_msg_facts']
            )
            eval_data_list.append(eval_data)
    return eval_data_list



@dataclass
class NewEvalDataPoint:
    task_context: TaskContext
    state_validation_config: StateValidationConfig

    @property
    def prompt(self) -> str:
        """Returns the prompt for the evaluation."""
        return self.task_context.intial_message

def load_example_dp() -> List[NewEvalDataPoint]:
    """Loads eval data from a CSV file."""
    # TODO: Temporary until tested
    dp = NewEvalDataPoint(
        task_context=TaskContext(
            goal="Create a single task in Jira",
            intial_message="Can you create a task for me in project MBA? Title is Discover prompt automation",
        ),
        state_validation_config=StateValidationConfig(
            state_validation_calls=[
                ApiCallValidation(
                    tool_name="jira_search",
                    arguments={
                        "jql": "project = MBA AND summary ~ 'Discover prompt automation'",
                        "limit": 1,
                    },
                    expected_fields={
                        "issues.0.summary": "Discover prompt automation",
                        "issues.0.status.name": "To Do"
                    },
                    expected_field_presence=["issues.0.key"],
                    expected_field_absence=None
                )
            ],
        ),
    )

    return [dp]
