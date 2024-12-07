from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM
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
from collections import defaultdict
from models.feedback import Feedback
from database import SessionLocal
import uuid

@dataclass
class QueryResult:
    malay_query: str
    english_translation: str
    sql_query: str
    relevant_columns: Dict[str, str]
    execution_time: float
    timestamp: str

@dataclass
class QueryFeedback:
    original_query: QueryResult
    corrected_sql: str
    timestamp: str

class GoldenQuery:
    def __init__(self, schema_index, golden_query_path: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing GoldenQuery with path: {golden_query_path}")
        self.encoder = schema_index.encoder
        self.golden_query_embedding = self._encode_query(self._load_golden_query(golden_query_path))
        self.logger.info("GoldenQuery initialized successfully")

    def _load_golden_query(self, path: str) -> str:
        self.logger.info(f"Loading golden query from: {path}")
        try:
            with open(path, 'r') as file:
                query = file.read().strip()
            self.logger.info("Golden query loaded successfully")
            return query
        except Exception as e:
            self.logger.error(f"Failed to load golden query: {str(e)}")
            raise

    def _encode_query(self, query: str):
        # Encode a query
        return self.encoder.encode([query])[0]

    def calculate_cosine_similarity(self, query: str) -> float:
        # Calculate the cosine similarity between a given query and the golden query.
        query_embedding = self._encode_query(query)
        return np.dot(query_embedding, self.golden_query_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(self.golden_query_embedding)
        )

    def rerank_queries(self, sql_queries: list) -> list:
        # Rerank a list of SQL queries based on their cosine similarity to the golden query.
        query_scores = [(query, self.calculate_cosine_similarity(query)) for query in sql_queries]
        return sorted(query_scores, key=lambda x: x[1], reverse=True)

class SchemaIndex:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing SchemaIndex with model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        self.user_indices = {}
        self.user_column_data = {}
        self.logger.info("SchemaIndex initialized successfully")
        
    def build_index(self, schema_json: Dict[str, Any], user_id: str):
        self.logger.info(f"Building index for user {user_id}")
        # Prepare column descriptions
        self.user_column_data[user_id] = [
            {
                "column_name": col_name,
                "data_type": info["data_type"],
                "description": info["description"],
                "text": f"Column {col_name} ({info['data_type']}): {info['description']}"
            }
            for col_name, info in schema_json["columns"].items()
        ]
        
        self.logger.info(f"Created column data for {len(self.user_column_data[user_id])} columns")
        
        # Create embeddings
        texts = [item["text"] for item in self.user_column_data[user_id]]
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        
        # Initialize FAISS index for user
        dimension = embeddings.shape[1]
        self.user_indices[user_id] = faiss.IndexFlatL2(dimension)
        self.user_indices[user_id].add(embeddings.cpu().numpy())
        self.logger.info(f"FAISS index built for user {user_id} with dimension {dimension}")
    
    def search(self, query: str, user_id: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant columns in user's schema"""
        if user_id not in self.user_indices:
            raise ValueError(f"No schema found for user {user_id}")
            
        query_vector = self.encoder.encode([query])
        distances, indices = self.user_indices[user_id].search(query_vector, k)
        
        return [self.user_column_data[user_id][idx] for idx in indices[0]]

class Malay2SQLService:
    def __init__(
        self,
        openai_api_key: str,
        cache_client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("Initializing Malay2SQLService")
        
        # OpenAI setup
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        # Translation model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Schema indexing setup
        self.schema_index = SchemaIndex()
        
        # Cache and logging setup
        self.cache_client = cache_client
        self.db = SessionLocal()
        self.logger.info("Loading feedback history from database")
        self.feedback_history = self._load_feedback_history()
        self.logger.info(f"Loaded {len(self.feedback_history)} feedback entries")
        self.user_schemas = {}

        # Initialize GoldenQuery
        golden_query_path = "data/golden_query.sql"
        try:
            self.golden_query = GoldenQuery(self.schema_index, golden_query_path)
        except Exception as e:
            self.logger.error(f"Failed to initialize GoldenQuery: {e}")
            raise

        # Initialize reranker model
        self._initialize_reranker()
        self.logger.info("Malay2SQLService initialization completed")

    def _initialize_reranker(self):
        """Initialize reranker model and tokenizer"""
        self.logger.info("Initializing reranker model and tokenizer")
        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
            self.reranker_model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma')
            self.yes_loc = self.reranker_tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
            self.reranker_model = self.reranker_model.to(self.device)
            self.reranker_model.eval()
            self.logger.info("Reranker model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize reranker model: {e}")
            raise

    def _initialize_models(self):
        """Initialize translation model"""
        print("Loading translation model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('mesolitica/nanot5-small-malaysian-translation-v2')
        self.translation_model = T5ForConditionalGeneration.from_pretrained('mesolitica/nanot5-small-malaysian-translation-v2')
        self.translation_model = self.translation_model.to(self.device)
        print(f"Model loaded. Parameters: {self.translation_model.num_parameters():,}")

    def initialize_schema(self, schema_json: Dict[str, Any], user_id: str):
        """Initialize schema index for specific user"""
        self.schema_index.build_index(schema_json, user_id)
        self.user_schemas[user_id] = schema_json

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

    async def get_relevant_columns(self, query: str, user_id: str) -> Dict[str, str]:
        """Get relevant columns using FAISS for specific user"""
        relevant_cols = self.schema_index.search(query, user_id)
        return {
            col["column_name"]: col["description"]
            for col in relevant_cols
        }
    
    async def store_feedback(self, original_result: QueryResult, corrected_sql: str) -> None:
        """Store user feedback for a query - shared across all users"""
        feedback = QueryFeedback(
            original_query=original_result,
            corrected_sql=corrected_sql,
            timestamp=datetime.now().isoformat()
        )
        
        self.feedback_history.append(feedback)
        
        # Store in database
        try:
            db_feedback = Feedback(
                id=str(uuid.uuid4()),
                original_query=original_result.__dict__,
                corrected_sql=corrected_sql,
                timestamp=datetime.now()
            )
            self.db.add(db_feedback)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Error storing feedback: {str(e)}")
            raise

    async def generate_sql_query(
        self,
        english_query: str,
        schema: str,
        relevant_columns: Dict[str, str]
    ) -> str:
        """Generate SQL query using OpenAI with context and feedback history"""
        
        # Create columns_context before using it in prompt
        columns_context = "\n".join([
            f"- {col}: {desc}" 
            for col, desc in relevant_columns.items()
        ])
        
        # Get relevant feedback examples
        feedback_examples = self._get_relevant_feedback(english_query)
        feedback_context = ""
        
        if feedback_examples:
            feedback_context = "\nHere are some example corrections from previous similar queries:\n"
            for feedback in feedback_examples:
                feedback_context += f"\nOriginal Query: {feedback.original_query.english_translation}"
                feedback_context += f"\nInitial SQL: {feedback.original_query.sql_query}"
                feedback_context += f"\nCorrected SQL: {feedback.corrected_sql}\n"
        
        prompt = f"""Given the following database schema and relevant columns:

    Schema:
    {schema}

    Relevant columns for this query:
    {columns_context}
    {feedback_context}

    Generate an SQL query for the following request:
    {english_query}

    Return only the raw SQL query without markdown formatting or backticks.
    """
        
        client = openai.OpenAI(api_key=self.openai_api_key)
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Generate precise SQL queries focusing on the relevant columns provided. Return only the raw SQL query without markdown formatting or backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        sql_query = response.choices[0].message.content
        # Strip markdown formatting if present
        sql_query = sql_query.replace('```sql\n', '').replace('\n```', '').strip()
        
        return sql_query

    def _get_relevant_feedback(self, query: str, max_examples: int = 3) -> List[QueryFeedback]:
        """Get relevant feedback examples based on query similarity - shared across users"""
        if not self.feedback_history:
            return []
        
        # Encode the current query
        query_embedding = self.schema_index.encoder.encode([query])[0]
        
        # Get embeddings for all feedback queries
        feedback_queries = [f.original_query.english_translation for f in self.feedback_history]
        feedback_embeddings = self.schema_index.encoder.encode(feedback_queries)
        
        # Calculate similarities
        similarities = np.dot(feedback_embeddings, query_embedding)
        most_similar_indices = np.argsort(similarities)[-max_examples:]
        
        return [self.feedback_history[i] for i in most_similar_indices if similarities[i] > 0.5]

    async def process_query(self, malay_query: str, user_id: str) -> QueryResult:
        """Process a Malay query end-to-end for specific user"""
        self.logger.info(f"Processing query for user {user_id}: {malay_query}")
        
        if user_id not in self.user_schemas:
            self.logger.error(f"No schema initialized for user {user_id}")
            raise ValueError(f"No schema initialized for user {user_id}")

        start_time = datetime.now()
        
        try:
            # Check cache
            if self.cache_client:
                cache_key = f"full_query:{user_id}:{malay_query}"
                cached_result = self.cache_client.get(cache_key)
                if cached_result:
                    self.logger.info("Query result found in cache")
                    return QueryResult(**json.loads(cached_result))
            
            # Translate query
            self.logger.info("Translating query to English")
            english_translation = self.translate_malay_to_english(malay_query)
            self.logger.info(f"Translated query: {english_translation}")
            
            # Get relevant columns
            self.logger.info("Finding relevant columns")
            relevant_columns = await self.get_relevant_columns(english_translation, user_id)
            self.logger.info(f"Found {len(relevant_columns)} relevant columns")
            
            # Generate SQL query
            self.logger.info("Generating SQL query")
            schema_sql = self.create_sql_schema(self.user_schemas[user_id])
            sql_query = await self.generate_sql_queries(
                english_translation,
                schema_sql,
                relevant_columns
            )
            self.logger.info(f"Generated SQL query: {sql_query}")
            
            # Create result
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Query processed in {execution_time:.2f} seconds")
            
            result = QueryResult(
                malay_query=malay_query,
                english_translation=english_translation,
                sql_query=sql_query,
                relevant_columns=relevant_columns,
                execution_time=execution_time,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache result
            if self.cache_client:
                self.logger.info("Caching query result")
                self.cache_client.set(
                    cache_key,
                    json.dumps(result.__dict__),
                    ex=1800
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query for user {user_id}: {str(e)}")
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
            
        columns_str = ',\n    '.join(columns)
        create_statement = (
            f"CREATE TABLE {schema_json['table_name']} (\n"
            f"    {columns_str}\n"
            f");"
        )
        
        return create_statement

    def _load_feedback_history(self) -> List[QueryFeedback]:
        """Load feedback history from database"""
        try:
            feedback_records = self.db.query(Feedback).all()
            feedback_history = []
            
            for record in feedback_records:
                original_query = QueryResult(**record.original_query)
                feedback = QueryFeedback(
                    original_query=original_query,
                    corrected_sql=record.corrected_sql,
                    timestamp=record.timestamp.isoformat()
                )
                feedback_history.append(feedback)
            
            return feedback_history
        except Exception as e:
            self.logger.error(f"Error loading feedback history: {str(e)}")
            return []

    async def generate_sql_queries(self, english_query: str, schema: str, relevant_columns: Dict[str, str]) -> str:
        """Generate and rerank SQL queries"""
        self.logger.info("Generating SQL queries with reranking")
        
        # Generate initial SQL query
        sql_query = await self.generate_sql_query(english_query, schema, relevant_columns)
        self.logger.info(f"Initial SQL query: {sql_query}")
        
        # Generate multiple variations if needed
        variations = [sql_query]  # In future, could generate multiple variations
        self.logger.info(f"Generated {len(variations)} query variations")
        
        # Rerank queries using both methods
        self.logger.info("Reranking queries")
        
        # 1. Golden query comparison
        golden_ranked = self.golden_query.rerank_queries(variations)
        self.logger.info(f"Golden query similarity score: {golden_ranked[0][1] if golden_ranked else 'N/A'}")
        
        # 2. Use reranker model
        try:
            with torch.no_grad():
                pairs = [[english_query, query] for query in variations]
                
                # Prepare inputs for reranker
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                outputs = self.reranker_model(**inputs)
                scores = outputs.logits[:, self.yes_loc].cpu().numpy()
                
                # Combine with queries
                reranker_ranked = list(zip(variations, scores))
                reranker_ranked.sort(key=lambda x: x[1], reverse=True)
                
                self.logger.info(f"Reranker model score: {reranker_ranked[0][1] if reranker_ranked else 'N/A'}")
                
                # Combine both rankings (simple average of normalized scores)
                final_scores = []
                for query in variations:
                    golden_score = next(score for q, score in golden_ranked if q == query)
                    reranker_score = next(score for q, score in reranker_ranked if q == query)
                    # Normalize and average scores
                    combined_score = (golden_score + reranker_score) / 2
                    final_scores.append((query, combined_score))
                
                final_scores.sort(key=lambda x: x[1], reverse=True)
                self.logger.info(f"Combined ranking score: {final_scores[0][1]}")
                
                final_query = final_scores[0][0]
                
        except Exception as e:
            self.logger.error(f"Error in reranking, falling back to golden query ranking: {str(e)}")
            final_query = golden_ranked[0][0] if golden_ranked else sql_query
        
        self.logger.info(f"Final selected query: {final_query}")
        return final_query