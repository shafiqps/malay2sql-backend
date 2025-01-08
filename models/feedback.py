from sqlalchemy import Column, String, DateTime, JSON, Text
from database import Base

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(String(36), primary_key=True)  # UUID length
    original_query = Column(JSON)
    corrected_sql = Column(Text)
    timestamp = Column(DateTime(timezone=True))