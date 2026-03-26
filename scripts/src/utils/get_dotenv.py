# get_dotenv
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    MINERU_API_KEY = os.getenv('MINERU_API_KEY')

    LLM_API_KEY = os.getenv('LLM_API_KEY')
    LLM_URL = os.getenv('LLM_URL')
    METHOD_MODEL = os.getenv('METHOD_MODEL')
    RERANK_MODEL = os.getenv('RERANK_MODEL')

    EMBEDDING_URL = os.getenv('EMBEDDING_URL')
    EMBEDDING_API_KEY = os.getenv('EMBEDDING_API')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

    DATABASE_HOST = os.getenv('DATABASE_HOST')
    DATABASE_PORT = os.getenv('DATABASE_PORT')
    DATABASE_DBNAME = os.getenv('DATABASE_DBNAME')
    DATABASE_USER = os.getenv('DATABASE_USER')
    DATABASE_PASSWORD = os.getenv('DATABASE_PASSWORD')

config = Config()
