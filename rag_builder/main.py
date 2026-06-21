from typing import List, Dict, Optional, Callable
import re
from tqdm import tqdm
from .crawler.web_document_crawler import FirecrawlDocumentCrawler
from .embedder.text_splitter import TextSplitter
from .embedder.embedding_factory import EmbeddingFactory
from .embedder.vector_store_factory import VectorStoreFactory
from .utils.file_handler import FileHandler
from .converter.pdf_converter import PDFConverter
from .utils.logger import get_logger
from .config.settings import settings

logger = get_logger(__name__)

class RAGBuilder:
    def __init__(
        self,
        use_tqdm: bool = True,
        embedding_type: str = None,
        model_name: str = None,
        vector_store_type: str = None
    ):
        """
        Args:
            use_tqdm: CLI에서 tqdm 진행바 사용 여부 (Streamlit에서는 False)
            embedding_type: 임베딩 타입 ("sentence_transformer" 또는 "openai")
            model_name: 임베딩 모델명
            vector_store_type: 벡터 스토어 타입 ("chromadb" 또는 "faiss")
        """
        self.document_crawler = FirecrawlDocumentCrawler()
        self.text_splitter = TextSplitter()
        self.pdf_converter = PDFConverter()

        # 동적 임베딩 엔진 생성
        self.embedding_type = embedding_type or getattr(settings, 'embedding_type', 'sentence_transformer')
        self.model_name = model_name or settings.local_embedding_model
        self.embedding_engine = EmbeddingFactory.create_embedding(
            embedding_type=self.embedding_type,
            model_name=self.model_name
        )

        # 벡터 스토어 타입 설정
        self.vector_store_type = vector_store_type or getattr(settings, 'vector_db_type', 'chromadb')

        self.file_handler = FileHandler()
        self.use_tqdm = use_tqdm

        logger.info(f"RAGBuilder 초기화 완료 - 임베딩: {self.embedding_type}/{self.model_name}, 벡터DB: {self.vector_store_type}")

    async def find_urls_for_query(self, query: str, max_urls: int = 10, categories: List[str] = None) -> List[Dict]:
        """
        Task 1: URL 목록 찾기 (Perplexity 또는 Firecrawl Search)

        Args:
            query: 자연어 검색 쿼리
            max_urls: 최대 URL 개수
            categories: Firecrawl 검색 카테고리 (["github", "research"] 등). None이면 Perplexity 사용

        Returns:
            URL 정보 리스트
        """
        try:
            urls = await self.document_crawler.analyze_query_and_find_urls(query, max_urls, categories)
            search_type = "Firecrawl" if categories else "Perplexity"
            logger.info(f"{search_type} 검색: {len(urls)}개 URL 발견")
            return urls
        except Exception as e:
            logger.error(f"URL 검색 실패: {e}")
            return []

    async def download_documents_from_urls(
        self,
        urls: List[Dict],
        output_folder: str,
        folder_name: str = None,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Task 2: Firecrawl scrape로 문서 다운로드 및 마크다운 저장

        Args:
            urls: URL 정보 리스트 (find_urls_for_query 결과)
            output_folder: 기본 출력 폴더
            folder_name: 하위 폴더명 (예: "Memory Forensic Tools")
            overwrite: 기존 파일 덮어쓰기 여부
            progress_callback: 진행상황 콜백 함수 (current, total)

        Returns:
            저장된 파일 경로 리스트
        """
        try:
            file_paths = await self.document_crawler.scrape_urls_to_markdown(
                urls, output_folder, folder_name, overwrite,
                progress_callback=progress_callback,
                use_tqdm=self.use_tqdm
            )
            logger.info(f"{len(file_paths)}개 문서 다운로드 완료")
            return file_paths
        except Exception as e:
            logger.error(f"문서 다운로드 실패: {e}")
            return []

    async def build_rag_from_query(
        self,
        query: str,
        output_folder: str,
        collection_name: str = None,
        max_urls: int = 10
    ) -> bool:
        """
        전체 워크플로우 실행: URL 검색 → 문서 다운로드 → RAG 구축

        Args:
            query: 자연어 검색 쿼리 (예: "official documents of memory forensic tools")
            output_folder: 문서 저장 폴더
            collection_name: RAG 컬렉션명 (None이면 쿼리에서 자동 생성)
            max_urls: 최대 URL 개수

        Returns:
            성공 여부
        """
        try:
            # Task 1: URL 검색 (Perplexity)
            logger.info(f"Task 1: Perplexity Search API로 URL 검색 - '{query}'")
            urls = await self.find_urls_for_query(query, max_urls)

            if not urls:
                logger.error("검색된 URL 없음")
                return False

            # Task 2: 문서 다운로드 (Firecrawl scrape)
            logger.info(f"Task 2: Firecrawl scrape로 {len(urls)}개 문서 다운로드")
            folder_name = collection_name or query.replace(' ', '_')[:50]
            file_paths = await self.download_documents_from_urls(urls, output_folder, folder_name)

            if not file_paths:
                logger.error("다운로드된 문서 없음")
                return False

            # Task 3: RAG 데이터베이스 구축
            logger.info(f"Task 3: RAG 데이터베이스 구축 - {len(file_paths)}개 문서")
            collection_name = self._sanitize_collection_name(collection_name or folder_name)
            success = await self.build_from_documents(file_paths, collection_name)

            if success:
                logger.info(f"✓ RAG 구축 완료: {collection_name}")
                logger.info(f"  - 검색된 URL: {len(urls)}개")
                logger.info(f"  - 다운로드: {len(file_paths)}개")
                logger.info(f"  - 저장 경로: {output_folder}/{folder_name}")

            return success

        except Exception as e:
            logger.error(f"RAG 구축 실패: {e}")
            return False

    def _sanitize_collection_name(self, name: str) -> str:
        """컬렉션명을 ChromaDB 호환 형식으로 변환"""
        # ChromaDB 규칙: 3-512자, [a-zA-Z0-9._-], 시작/끝은 영숫자

        # 공백을 언더스코어로 변환
        name = name.replace(' ', '_')

        # 허용되지 않는 문자를 언더스코어로 변환
        name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)

        # 연속 언더스코어 제거
        name = re.sub(r'_+', '_', name)

        # 앞뒤 언더스코어 제거
        name = name.strip('_')

        # 빈 문자열이거나 너무 짧은 경우
        if not name or len(name) < 3:
            name = "custom_collection"

        # 길이 제한
        if len(name) > 512:
            name = name[:512]

        # 시작과 끝이 영문자 또는 숫자인지 확인
        if not re.match(r'^[a-zA-Z0-9]', name):
            name = f"col_{name}"
        if not re.match(r'.*[a-zA-Z0-9]$', name):
            name = f"{name}_col"

        return name

    async def build_from_documents(
        self,
        file_paths: List[str],
        collection_name: str = "rag_documents",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """지정된 문서 파일들로부터 RAG 데이터베이스 구축"""
        try:
            # 컬렉션명 정리
            collection_name = self._sanitize_collection_name(collection_name)
            logger.info(f"RAG 구축 시작: {len(file_paths)}개 문서, 컬렉션: {collection_name}")

            # 문서 로드 (PDF 파일은 먼저 Markdown으로 변환)
            all_documents = []
            file_iterator = tqdm(file_paths, desc="📂 문서 로딩", unit="파일") if self.use_tqdm else file_paths

            for file_path in file_iterator:
                try:
                    # PDF 파일인 경우 Markdown으로 변환
                    if file_path.lower().endswith('.pdf'):
                        logger.info(f"PDF 파일 감지: {file_path}")
                        if self.pdf_converter.is_dolphin_installed():
                            # downloads 폴더에 변환된 MD 파일 저장
                            file_path = self.pdf_converter.convert_and_replace(file_path, output_folder="./downloads")
                        else:
                            logger.warning(f"Dolphin이 설치되지 않았습니다. PDF 파일을 건너뜁니다: {file_path}")
                            logger.warning("설치 방법: https://github.com/ByteDance/Dolphin")
                            continue

                    content = self.file_handler.load_file(file_path)
                    if content:
                        filename = file_path.split('/')[-1]

                        # 파일명에서 URL 해시 추출 (형식: {hash}.md 또는 {hash}.txt)
                        url_hash = filename.replace('.md', '').replace('.txt', '').replace('.pdf', '')

                        all_documents.append({
                            "content": content,
                            "metadata": {
                                "source": file_path,
                                "filename": filename,
                                "file_type": file_path.split('.')[-1].lower(),
                                "url_hash": url_hash  # URL 해시 추가
                            }
                        })
                except Exception as e:
                    logger.error(f"문서 로드 실패 {file_path}: {e}")

            if not all_documents:
                logger.error("로드된 문서 없음")
                return False

            # 문서 처리 및 임베딩
            return await self._process_and_store_documents(all_documents, collection_name, progress_callback)

        except Exception as e:
            logger.error(f"RAG 구축 실패: {e}")
            return False

    async def _process_and_store_documents(
        self,
        documents: List[Dict],
        collection_name: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """문서 처리 및 벡터 데이터베이스 저장"""
        try:
            # 텍스트 분할
            all_chunks = []
            doc_iterator = tqdm(documents, desc="✂️ 텍스트 분할", unit="문서") if self.use_tqdm else documents

            for doc in doc_iterator:
                try:
                    chunks = self.text_splitter.split_text(
                        doc["content"],
                        metadata=doc["metadata"]
                    )
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"텍스트 분할 실패: {e}")

            if not all_chunks:
                logger.error("분할된 청크 없음")
                return False

            logger.info(f"총 {len(all_chunks)}개 청크 생성")

            # 임베딩 생성 (배치 처리)
            texts = [chunk["content"] for chunk in all_chunks]
            embeddings = self.embedding_engine.embed_texts(
                texts,
                progress_callback=progress_callback,
                use_tqdm=self.use_tqdm
            )

            # 벡터 데이터 준비 (BaseVectorStore 시그니처에 맞게)
            ids = [chunk["id"] for chunk in all_chunks]
            documents = [chunk["content"] for chunk in all_chunks]
            metadatas = [chunk["metadata"] for chunk in all_chunks]

            if not ids:
                logger.error("생성된 임베딩 없음")
                return False

            # 벡터 데이터베이스 저장 (동적 벡터 스토어 생성)
            vector_store = VectorStoreFactory.create_vector_store(
                store_type=self.vector_store_type,
                collection_name=collection_name,
                dimension=self.embedding_engine.get_dimension()
            )
            success = vector_store.add_documents(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            if success:
                logger.info(f"RAG 구축 완료: {len(all_chunks)}개 문서, {len(embeddings)}개 임베딩 → {self.vector_store_type}")
                return True
            else:
                logger.error("벡터 DB 저장 실패")
                return False

        except Exception as e:
            logger.error(f"문서 처리 실패: {e}")
            return False
