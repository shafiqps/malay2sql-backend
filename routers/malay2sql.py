from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import get_db
from models.user import UserDB
from routers.auth import get_current_user
from services.malay2sql_service import process_malay_query, process_schema_file

router = APIRouter(prefix="/malay2sql", tags=["malay2sql"])

@router.post("/upload-schema")
async def upload_schema(file: UploadFile = File(...), current_user: UserDB = Depends(get_current_user)):
    try:
        schema_content = await file.read()
        processed_schema = process_schema_file(schema_content)
        # Here you would typically store the processed schema for the user
        return {"message": "Schema uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/query")
async def malay_to_sql_query(query: str, current_user: UserDB = Depends(get_current_user)):
    try:
        sql_query = process_malay_query(query)
        return {"sql_query": sql_query}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))