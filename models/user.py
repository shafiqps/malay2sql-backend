from sqlalchemy import Column, Integer, String, Text
from pydantic import BaseModel, EmailStr
from database import Base

class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(Text)
    first_name = Column(String(100))
    last_name = Column(String(100))
    profile_picture_url = Column(String(255), nullable=True) 

class UserCreate(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str

class UserUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None

class UserResponse(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str
    profile_picture_url: str | None = None 

    class Config:
        from_attributes = True