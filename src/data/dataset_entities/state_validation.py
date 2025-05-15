"""
State validation models for Jira MCP agent evaluation.

This module defines the data structures for validating the state of the Jira system
after agent interactions, allowing for verification of actions taken by the agent.

Key features:
- Support for nested field validation using dot notation (e.g., 'issues.0.summary')
- Validation of field values, presence, and absence
- Array index access for validating elements in lists (e.g., 'issues.0' for first issue)
- Configurable validation with multiple API calls
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ApiCallValidation(BaseModel):
    """
    Represents an API call to validate the state after agent execution.
    
    This model defines both the API call to make and the expected response,
    allowing for verification that the agent's actions had the intended effects.
    
    Supports nested field validation using dot notation for accessing nested structures
    and array indices. For example:
    - 'issues.0.summary': Accesses the summary field of the first issue in the issues array
    - 'issues.0.status.name': Accesses the name field within the status object of the first issue
    """
    
    tool_name: str  # Name of the MCP tool to call for validation
    arguments: Dict[str, Any]  # Arguments to pass to the tool call
    
    expected_fields: Optional[Dict[str, Any]]  # Dictionary mapping field paths to expected values
                                               # Example: {"issues.0.summary": "Test issue", 
                                               #           "issues.0.status.name": "To Do"}
    
    expected_field_presence: Optional[List[str]]  # List of field paths that should exist
                                                  # Example: ["issues.0.key", "issues.0.id"]
    
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validates the response against the expected fields and conditions.
        
        This method checks:
        1. That specified fields have expected values (using expected_fields)
        2. That specified fields exist (using expected_field_presence)
        
        All field paths support nested access using dot notation and array indices.
        
        Args:
            response: The API response to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check expected fields
        if self.expected_fields:
            for field_path, expected_value in self.expected_fields.items():
                actual_value = self._get_nested_value(response, field_path)
                if actual_value != expected_value:
                    return False
        
        # Check field presence
        if self.expected_field_presence:
            for field_path in self.expected_field_presence:
                if self._get_nested_value(response, field_path) is None:
                    return False
        
        return True
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Gets a value from a nested dictionary or list using dot notation.
        
        This method supports:
        - Dictionary key access: 'user.name' accesses data['user']['name']
        - Array index access: 'issues.0' accesses data['issues'][0]
        - Mixed access: 'issues.0.status.name' accesses data['issues'][0]['status']['name']
        
        Args:
            data: The dictionary or list to search in
            path: The path to the value, using dot notation (e.g., 'issues.0.summary')
            
        Returns:
            The value at the specified path, or None if any part of the path is not found
        """
        parts = path.split('.')
        current = data
        
        for part in parts:
            # Handle array indices
            if part.isdigit():
                index = int(part)
                if isinstance(current, list) and 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            # Handle dictionary keys
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current


class StateValidationConfig(BaseModel):
    """
    Configuration for state validation after agent execution.
    
    This model defines a sequence of API calls to validate the state of the system
    after an agent has performed actions. It allows for verification that the
    agent's actions had the intended effects on the system.
    
    Example usage:
        config = StateValidationConfig(
            state_validation_calls=[
                ApiCallValidation(
                    tool_name="jira_search",
                    arguments={"jql": "project = MBA", "limit": 1},
                    expected_result_type="success",
                    expected_fields={"issues.0.summary": "Expected Summary"}
                )
            ],
            fail_fast=True
        )
    """
    
    state_validation_calls: List[ApiCallValidation] = Field(
        description="List of API calls to validate the state",
        default_factory=list
    )
    
    fail_fast: bool = Field(
        default=False,
        description="If True, stop validation on first failure"
    )
