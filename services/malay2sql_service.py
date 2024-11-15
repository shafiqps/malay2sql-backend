from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import json

@dataclass
class QueryResult:
    malay_query: str
    english_translation: str
    sql_query: str
    relevant_columns: Dict[str, str]
    execution_time: float
    timestamp: str

class SchemaIndex:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        self.column_data = []
        
    def build_index(self, schema_json: Dict[str, Any]):
        """Build FAISS index from schema"""
        # Prepare column descriptions
        self.column_data = [
            {
                "column_name": col_name,
                "data_type": info["data_type"],
                "description": info["description"],
                "text": f"Column {col_name} ({info['data_type']}): {info['description']}"
            }
            for col_name, info in schema_json["columns"].items()
        ]
        
        # Create embeddings
        texts = [item["text"] for item in self.column_data]
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.cpu().numpy())
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant columns"""
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector, k)
        
        return [self.column_data[idx] for idx in indices[0]]

class Malay2SQLService:
    def __init__(
        self,
        openai_api_key: str,
        cache_client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        # OpenAI setup
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Translation model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Schema indexing setup
        self.schema_index = SchemaIndex()
        
        # Cache and logging setup
        self.cache_client = cache_client
        self.logger = logger or logging.getLogger(__name__)

    def _initialize_models(self):
        """Initialize translation model"""
        print("Loading translation model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('mesolitica/nanot5-small-malaysian-translation-v2')
        self.translation_model = T5ForConditionalGeneration.from_pretrained('mesolitica/nanot5-small-malaysian-translation-v2')
        self.translation_model = self.translation_model.to(self.device)
        print(f"Model loaded. Parameters: {self.translation_model.num_parameters():,}")

    def initialize_schema(self, schema_json: Dict[str, Any]):
        """Initialize schema index"""
        self.schema_index.build_index(schema_json)
        self.schema_json = schema_json

    def translate_malay_to_english(self, malay_text: str) -> str:
        """Translate Malay to English using local model"""
        # Check cache first
        if self.cache_client:
            cache_key = f"translation:{malay_text}"
            cached_translation = self.cache_client.get(cache_key)
            if cached_translation:
                return cached_translation.decode('utf-8')

        # Perform translation
        prefix = 'terjemah ke Inggeris: '
        input_text = f"{prefix}{malay_text}"
        input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.translation_model.generate(
                input_ids,
                max_length=1024,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                do_sample=True
            )
        
        # Process output
        all_special_ids = [0, 1, 2]
        outputs = [i for i in outputs[0] if i not in all_special_ids]
        translation = self.tokenizer.decode(outputs, spaces_between_special_tokens=False).strip()
        
        # Cache translation
        if self.cache_client:
            self.cache_client.set(cache_key, translation, ex=3600)
        
        return translation

    async def get_relevant_columns(self, query: str) -> Dict[str, str]:
        """Get relevant columns using FAISS"""
        relevant_cols = self.schema_index.search(query)
        return {
            col["column_name"]: col["description"]
            for col in relevant_cols
        }

    async def generate_sql_query(
        self,
        english_query: str,
        schema: str,
        relevant_columns: Dict[str, str]
    ) -> str:
        """Generate SQL query using OpenAI with context"""
        columns_context = "\n".join([
            f"- {col}: {desc}" 
            for col, desc in relevant_columns.items()
        ])
        
        prompt = f"""Given the following database schema and relevant columns:

Schema:
{schema}

Relevant columns for this query:
{columns_context}

Generate an SQL query for the following request:
{english_query}

Return only the SQL query without any explanation.
"""
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate precise SQL queries focusing on the relevant columns provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()

    async def process_query(self, malay_query: str) -> QueryResult:
        """Process a Malay query end-to-end"""
        start_time = datetime.now()
        
        try:
            # Check cache for identical query
            if self.cache_client:
                cache_key = f"full_query:{malay_query}"
                cached_result = self.cache_client.get(cache_key)
                if cached_result:
                    return QueryResult(**json.loads(cached_result))
            
            # Translate query
            english_translation = self.translate_malay_to_english(malay_query)
            
            # Get relevant columns
            relevant_columns = await self.get_relevant_columns(english_translation)
            
            # Generate SQL query
            schema_sql = self.create_sql_schema(self.schema_json)
            sql_query = await self.generate_sql_query(
                english_translation,
                schema_sql,
                relevant_columns
            )
            
            # Create result
            result = QueryResult(
                malay_query=malay_query,
                english_translation=english_translation,
                sql_query=sql_query,
                relevant_columns=relevant_columns,
                execution_time=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now().isoformat()
            )
            
            # Cache result
            if self.cache_client:
                self.cache_client.set(
                    cache_key,
                    json.dumps(result.__dict__),
                    ex=1800
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            raise

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
            
        return f"""CREATE TABLE {schema_json["table_name"]} (
            {',\n            '.join(columns)}
        );"""