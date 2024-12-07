from sqlalchemy import Column, String, DateTime, JSON
from database import Base

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(String, primary_key=True)
    original_query = Column(JSON)  # Stores QueryResult as JSON
    corrected_sql = Column(String)
    timestamp = Column(DateTime) 