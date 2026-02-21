"""
Base module for SQLAlchemy models.

This file should be imported first by any model to avoid circular imports.
"""
from sqlalchemy.orm import declarative_base

Base = declarative_base()
