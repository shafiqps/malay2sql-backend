from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import sqlite3
import json
import os

class DatabaseExecutor:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_path = "mock_database.db"
        self.connection = None
        
        # Default mock data
        self.default_mock_data = {
            "staf_penjawatan": [
                {
                    "TAHUN": "2023",
                    "BULAN": "03",
                    "NO_STAF": "A12345",
                    "NO_KAD_PENGENALAN": "890123456789",
                    "NAMA": "Ahmad bin Abdullah",
                    "EMAIL_UMMAIL": "ahmad@um.edu.my",
                    "KOD_PTJ": "F01",
                    "KTRGN_FAKULTI_PTJ_BM": "Fakulti Sains Komputer",
                    "KOD_JABATAN": "CS01",
                    "KTRGN_JABATAN_BM": "Jabatan Sistem Maklumat",
                    "TARIKH_LANTIKAN_MULA": "2020-01-15",
                    "GAJI_POKOK": 5000.00,
                    "KTRGN_JANTINA_BM": "Lelaki",
                    "KTRGN_BANGSA_BM": "Melayu",
                    "KTRGN_AGAMA_BM": "Islam"
                },
                {
                    "TAHUN": "2023",
                    "BULAN": "03",
                    "NO_STAF": "A12346",
                    "NO_KAD_PENGENALAN": "910123456789",
                    "NAMA": "Tan Mei Ling",
                    "EMAIL_UMMAIL": "meiling@um.edu.my",
                    "KOD_PTJ": "F01",
                    "KTRGN_FAKULTI_PTJ_BM": "Fakulti Sains Komputer",
                    "KOD_JABATAN": "CS02",
                    "KTRGN_JABATAN_BM": "Jabatan Kepintaran Buatan",
                    "TARIKH_LANTIKAN_MULA": "2019-06-20",
                    "GAJI_POKOK": 4800.00,
                    "KTRGN_JANTINA_BM": "Perempuan",
                    "KTRGN_BANGSA_BM": "Cina",
                    "KTRGN_AGAMA_BM": "Buddha"
                },
                {
                    "TAHUN": "2023",
                    "BULAN": "03",
                    "NO_STAF": "A12347",
                    "NO_KAD_PENGENALAN": "920123456789",
                    "NAMA": "Raj Kumar",
                    "EMAIL_UMMAIL": "raj@um.edu.my",
                    "KOD_PTJ": "F02",
                    "KTRGN_FAKULTI_PTJ_BM": "Fakulti Kejuruteraan",
                    "KOD_JABATAN": "EN01",
                    "KTRGN_JABATAN_BM": "Jabatan Kejuruteraan Elektrik",
                    "TARIKH_LANTIKAN_MULA": "2021-03-10",
                    "GAJI_POKOK": 5200.00,
                    "KTRGN_JANTINA_BM": "Lelaki",
                    "KTRGN_BANGSA_BM": "India",
                    "KTRGN_AGAMA_BM": "Hindu"
                }
            ]
        }
        
    def initialize_database(self, schema_json: Dict[str, Any], mock_data: Dict[str, List[Dict[str, Any]]] = None):
        """Initialize SQLite database with schema and mock data"""
        self.logger.info("Initializing mock database")
        
        # Use provided mock data or default mock data
        mock_data = mock_data or self.default_mock_data
        
        # Create new database connection
        if os.path.exists(self.db_path):
            os.remove(self.db_path)  # Remove existing database
            
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        try:
            # Extract schema name from table name
            schema_name = schema_json["table_name"].split('.')[0]
            
            # Create schema/database if it doesn't exist
            cursor.execute(f"ATTACH DATABASE '{schema_name}.db' AS {schema_name}")
            
            # Create table based on schema
            create_table_sql = self._create_table_sql(schema_json)
            self.logger.info(f"Creating table with SQL: {create_table_sql}")
            cursor.execute(create_table_sql)
            
            # Insert mock data
            if mock_data:
                table_name = schema_json["table_name"]  # Use full table name including schema
                columns = list(schema_json["columns"].keys())
                placeholders = ",".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
                
                # Convert mock data to list of tuples
                values = [
                    tuple(row.get(col) for col in columns)
                    for row in mock_data.get(table_name.split('.')[-1], [])  # Get data using table name without schema
                ]
                
                self.logger.info(f"Inserting {len(values)} rows of mock data")
                cursor.executemany(insert_sql, values)
            
            self.connection.commit()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            self.connection.rollback()
            raise
        
    def _create_table_sql(self, schema_json: Dict[str, Any]) -> str:
        """Convert JSON schema to CREATE TABLE statement"""
        type_mapping = {
            "String": "TEXT",
            "Int64": "INTEGER",
            "Float64": "REAL",
            "DateTime64(3)": "DATETIME",
            "Nullable(String)": "TEXT",
            "Nullable(Int64)": "INTEGER",
            "Nullable(Float64)": "REAL",
            "Nullable(DateTime64(3))": "DATETIME"
        }
        
        columns = []
        for col_name, info in schema_json["columns"].items():
            data_type = type_mapping.get(info["data_type"], "TEXT")
            # Remove backticks as they're not needed in SQLite
            col_name = col_name.replace('`', '')
            columns.append(f"{col_name} {data_type}")
            
        # Remove database prefix if present
        table_name = schema_json['table_name'].split('.')[-1]
            
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {', '.join(columns)}
        )
        """

    async def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query on SQLite database"""
        self.logger.info(f"Executing query: {sql_query}")
        start_time = datetime.now()
        
        if not self.connection:
            raise ValueError("Database not initialized")
            
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            
            # Fetch results for SELECT queries
            results = []
            if sql_query.strip().upper().startswith("SELECT"):
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                results = [
                    dict(zip(columns, row))
                    for row in rows
                ]
            
            self.connection.commit()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "results": results,
                "rows_affected": cursor.rowcount,
                "execution_time": execution_time,
                "query_type": sql_query.strip().split()[0].upper()
            }
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            self.connection.rollback()
            return {
                "status": "error",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

    def __del__(self):
        """Close database connection on cleanup"""
        if self.connection:
            self.connection.close() 