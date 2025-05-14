import csv
from dataclasses import dataclass
import os
from typing import List

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
