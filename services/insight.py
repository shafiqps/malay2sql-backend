import pandas as pd
from sqlalchemy import create_engine, text
import openai
from malay2sql_service import Malay2SQLService  # Import the Malay2SQLService class
from typing import Dict, Any
import logging

class Insight:
    def __init__(self, db_url: str, openai_api_key: str):
        self.engine = create_engine(db_url)
        self.malay2sql_service = Malay2SQLService(openai_api_key)  # Initialize Malay2SQLService
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def get_top_trends(self) -> pd.DataFrame:
        prompt = """
        Generate an SQL query to find the top 5 most interesting trends in the data.
        The query should identify trends based on significant changes in data patterns, such as:
        - Count the total number of faculty members.
        - List the names and ages of all faculty members.
        - Retrieve the oldest faculty member's details.
        - Find faculty members born in a specific state (e.g., Texas).
        - Count faculty members in each state.
        - Identify faculty members whose age is above the average.
        - List all distinct states where faculty were born.
        - Show faculty details sorted by name alphabetically.
        - Count how many faculty members are from a specific state (e.g., California).
        - Display the youngest faculty member.
        - Count the total number of departments.
        - List all department names and their budgets.
        - Find the department with the highest budget.
        - Calculate the average budget for all departments.
        - Identify departments with a budget below a specified value (e.g., $10 billion).
        - Retrieve details of departments ranked in the top 5.
        - Show all departments created before a specific year (e.g., 2000).
        - Find the year with the most department creations.
        - Display the number of employees in each department.
        - Find departments with more than 100 employees.
        - List all department heads and their departments.
        - Identify departments with temporary acting heads.
        - Find all departments led by heads born in a specific state.
        - Count the number of departments managed by each head.
        - Retrieve the list of acting heads and their tenure duration.
        - Compute the average age of faculty members.
        - Find states with the highest number of faculty members.
        """
        try:
            query = self.malay2sql_service.generate_sql(prompt)
            self.logger.debug(f"Generated SQL query: {query}")
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
        except Exception as e:
            self.logger.error(f"Error in get_top_trends: {e}")
            return pd.DataFrame()

    def create_sql_schema(self, schema_json: Dict[str, Any]) -> str:
        """Convert JSON schema to SQL CREATE TABLE statement"""
        type_mapping = {
            "String": "VARCHAR(255)",
            "Int64": "INTEGER",
            "Float64": "DECIMAL(15,2)",
            "DateTime64(3)": "DATETIME"
        }
        
        columns = []
        for col_name, info in schema_json["columns"].items():
            data_type = type_mapping.get(info["data_type"], "VARCHAR(255)")
            comment = info["description"].replace("'", "''")
            columns.append(f"{col_name} {data_type} COMMENT '{comment}'")
            
        columns_str = ',\n    '.join(columns)
        create_statement = (
            f"CREATE TABLE {schema_json['table_name']} (\n"
            f"    {columns_str}\n"
            f");"
        )
        
        return create_statement

    def get_schema_json(self) -> Dict[str, Any]:
        """Retrieve the schema JSON (this is a placeholder, implement as needed)"""
        # Implement this method to return the schema JSON
        return {
            "table_name": "head",
            "columns": {
                "name": {"data_type": "String", "description": "Name of the faculty member"},
                "age": {"data_type": "Int64", "description": "Age of the faculty member"},
                "born_state": {"data_type": "String", "description": "State where the faculty member was born"},
                "department_id": {"data_type": "Int64", "description": "ID of the department the faculty member belongs to"}
            }
        }

# Example usage:
# insight = Insight(db_url="your_database_url", openai_api_key="your_openai_api_key")
# trends_df = await insight.get_top_trends()