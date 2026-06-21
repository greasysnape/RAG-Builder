# RAG Builder

자연어 쿼리로 관련 웹 문서를 자동 수집하고 RAG 벡터 데이터베이스를 구축하는 도구
Streamlit 웹 UI로 동작

## 동작 방식

1. **URL 검색** — 자연어 쿼리를 Perplexity 또는 Firecrawl Search API로 분석해 관련 URL 수집 (쿼리당 최대 30개)
2. **문서 다운로드** — Firecrawl Scrape로 각 URL을 마크다운으로 변환해 `downloads/`에 저장. PDF는 Dolphin으로 변환(설치 시)
3. **RAG 구축** — 문서를 청크로 나눠 임베딩한 뒤 벡터 DB에 저장

로컬 문서 파일 업로드, 컬렉션 관리(생성·삭제·통계), 의미 기반 검색도 지원

## 구성

| 구성 요소 | 사용 기술 |
|-----------|-----------|
| URL 검색 | Perplexity Search API / Firecrawl Search API |
| 웹 스크래핑 | Firecrawl Scrape API |
| 임베딩 | sentence-transformers (기본 `all-MiniLM-L6-v2`, 로컬) 또는 OpenAI Embeddings |
| 벡터 DB | ChromaDB (기본) 또는 FAISS, 로컬 저장 |
| 웹 UI | Streamlit |

## 설치

Python 3.10 이상 필요.

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# 2. 의존성 설치
pip install -r requirements.txt
```

## 환경 변수 설정

프로젝트 루트에 `.env` 파일을 만들고 API 키 입력 

```env
# API 키
FIRECRAWL_API_KEY=your_firecrawl_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key

# 임베딩 (선택, 기본값 사용 가능)
EMBEDDING_TYPE=sentence_transformer
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
# OPENAI_API_KEY=your_openai_api_key   # OpenAI 임베딩 사용 시

# 텍스트 분할
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 벡터 DB
VECTOR_DB_TYPE=chromadb
VECTOR_DB_DIRECTORY=./vector_db

# 크롤링
MAX_URLS_PER_QUERY=30
REQUEST_TIMEOUT=30

# 로그
LOG_LEVEL=INFO
```

`FIRECRAWL_API_KEY`는 문서 다운로드에, `PERPLEXITY_API_KEY`는 URL 검색에 사용

## 실행

```bash
streamlit run streamlit_app.py
```

브라우저에서 `http://localhost:8501`로 접속

### 탭 구성
- **전체 워크플로우** — 검색부터 RAG 구축까지 한 번에 실행
- **Task 1: URL 검색** — URL만 검색
- **Task 2: 문서 다운로드** — 검색된 URL에서 문서 다운로드
- **Task 3: RAG 구축** — 다운로드된 문서로 벡터 DB 생성
- **정보** — 컬렉션 관리 및 시스템 정보

## 프로젝트 구조

```
RAG-Builder/
├── streamlit_app.py              # Streamlit 웹 UI (진입점)
├── requirements.txt              # Python 의존성
├── .env                          # API 키 및 환경 변수
├── rag_builder/
│   ├── main.py                   # RAGBuilder 클래스 (워크플로우 핵심)
│   ├── config/settings.py        # 환경 변수 로드
│   ├── crawler/                  # Perplexity + Firecrawl 크롤러
│   ├── converter/                # PDF → 마크다운 변환 (Dolphin)
│   ├── embedder/                 # 텍스트 분할, 임베딩, 벡터 스토어
│   ├── locales/                  # 다국어 (en / ko / fr)
│   └── utils/                    # 파일 I/O, 로깅, 사용자 설정
├── downloads/                    # 다운로드된 문서
└── vector_db/                    # 벡터 데이터베이스
```

## 라이선스

Kyuyeon Choi
