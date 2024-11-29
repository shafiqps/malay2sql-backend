from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
from services.malay2sql_service import Malay2SQLService, QueryResult
from dependencies import get_malay2sql_service
from pydantic import BaseModel
from auth import get_current_user
from models.user import UserDB

router = APIRouter(
    prefix="/malay2sql",
    tags=["malay2sql"]
)

class SchemaInput(BaseModel):
    schema: Dict[str, Any]

class QueryInput(BaseModel):
    query: str

@router.post("/initialize", status_code=200)
async def initialize_schema(
    schema_input: SchemaInput,
    service: Malay2SQLService = Depends(get_malay2sql_service),
    current_user: UserDB = Depends(get_current_user) 
):
    """Initialize the schema for the Malay2SQL service"""
    try:
        service.initialize_schema(schema_input.schema)
        return {"message": "Schema initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResult)
async def process_query(
    query_input: QueryInput,
    service: Malay2SQLService = Depends(get_malay2sql_service),
    current_user: UserDB = Depends(get_current_user)  
) -> QueryResult:
    """Process a Malay language query and return SQL"""
    try:
        result = await service.process_query(query_input.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))