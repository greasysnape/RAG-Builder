import os
from dotenv import load_dotenv

# .env 파일 강제 재로드
load_dotenv(override=True)

class Settings:
    """간단한 설정 클래스 (pydantic 의존성 제거)"""

    def __init__(self):
        # 임베딩 설정
        self.embedding_type = os.getenv("EMBEDDING_TYPE", "sentence_transformer")
        self.local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # OpenAI API 키 (OpenAI Embeddings 사용 시)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # 텍스트 분할 설정
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        # 벡터 데이터베이스 설정 (일반)
        self.vector_db_type = os.getenv("VECTOR_DB_TYPE", "chromadb")
        self.vector_db_directory = os.getenv("VECTOR_DB_DIRECTORY", "./vector_db")
        self.vector_store_name = os.getenv("VECTOR_STORE_NAME", "ChromaDB")

        # 하위 호환성을 위한 별칭 (deprecated)
        self.chroma_persist_directory = self.vector_db_directory
        self.CHROMA_PERSIST_DIRECTORY = self.vector_db_directory

        # 크롤링 설정
        self.max_urls_per_query = int(os.getenv("MAX_URLS_PER_QUERY", "100"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))

        # MCP 설정
        self.firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")

        # Perplexity API 설정
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "")

        # 미리 정의된 컬렉션
        self.predefined_collections = [
            "digital_forensics_tools",
            "digital_forensics_techniques",
            "investigation_procedures",
            "internal_regulations"
        ]

        # 로그 설정
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()