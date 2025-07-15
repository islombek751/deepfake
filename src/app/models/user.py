import uuid
from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
from src.app.db.base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)