"""
MongoDB client initialization.
"""

from motor.motor_asyncio import AsyncIOMotorClient

import os

# Read from environment, default to localhost for development
MONGO_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "adaptive_rag")

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
