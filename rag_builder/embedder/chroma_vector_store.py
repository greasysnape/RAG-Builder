"""ChromaDB 기반 벡터 스토어"""
import chromadb
from typing import List, Dict, Optional
from .base_vector_store import BaseVectorStore
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB 기반 벡터 스토어 구현"""

    def __init__(self, collection_name: str = "rag_documents", **kwargs):
        super().__init__(collection_name)
        self.persist_directory = kwargs.get('persist_directory', settings.vector_db_directory)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """컬렉션 가져오기 또는 생성"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"기존 ChromaDB 컬렉션 로드: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Builder Documents"}
            )
            logger.info(f"새 ChromaDB 컬렉션 생성: {self.collection_name}")

        return collection

    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> bool:
        """문서를 ChromaDB에 추가 (upsert 모드: 중복 시 업데이트)"""
        try:
            # upsert 사용: 같은 ID가 있으면 업데이트, 없으면 추가
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"ChromaDB 문서 추가/업데이트 완료: {len(ids)}개")
            return True

        except Exception as e:
            logger.error(f"ChromaDB 문서 추가 실패: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """유사도 기반 문서 검색"""
        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }

            # 필터 적용 (ChromaDB where 문법)
            if filter_dict:
                query_params["where"] = filter_dict

            results = self.collection.query(**query_params)

            # 결과 포맷팅
            formatted_results = []
            if results and results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        "id": results["ids"][0][i] if results.get("ids") else None
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"ChromaDB 검색 실패: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> bool:
        """ID로 문서 삭제"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"ChromaDB 문서 삭제: {len(ids)}개")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 문서 삭제 실패: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "type": "chromadb"
            }
        except Exception as e:
            logger.error(f"ChromaDB 통계 조회 실패: {e}")
            return {}

    def delete_collection(self) -> bool:
        """컬렉션 삭제"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"ChromaDB 컬렉션 삭제: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"ChromaDB 컬렉션 삭제 실패: {e}")
            return False

    def check_duplicate_embeddings(self, file_paths: List[str]) -> Dict:
        """파일들의 중복 임베딩 체크"""
        try:
            duplicates = []
            new_files = []

            for file_path in file_paths:
                filename = file_path.split('/')[-1]

                # 파일명에서 URL 해시 추출
                url_hash = None
                if '_' in filename:
                    parts = filename.rsplit('_', 1)
                    if len(parts) == 2:
                        url_hash = parts[1].replace('.md', '').replace('.txt', '')

                if url_hash:
                    try:
                        # 첫 번째 청크 ID로 존재 여부 확인
                        chunk_id = f"{url_hash}_0"
                        results = self.collection.get(ids=[chunk_id])

                        if results and results['ids']:
                            duplicates.append({
                                "filename": filename,
                                "url_hash": url_hash,
                                "path": file_path
                            })
                        else:
                            new_files.append(file_path)
                    except:
                        new_files.append(file_path)
                else:
                    new_files.append(file_path)

            return {
                "duplicates": duplicates,
                "new_files": new_files
            }

        except Exception as e:
            logger.error(f"ChromaDB 중복 체크 실패: {e}")
            return {"duplicates": [], "new_files": file_paths}

    def get_embedded_hashes(self) -> List[str]:
        """임베딩된 모든 URL 해시 목록 가져오기"""
        try:
            all_data = self.collection.get()
            ids = all_data.get("ids", [])

            # ID에서 URL 해시 추출 (형식: {hash}_0, {hash}_1, ...)
            hashes = set()
            for doc_id in ids:
                if "_" in doc_id:
                    url_hash = doc_id.rsplit("_", 1)[0]
                    hashes.add(url_hash)

            return list(hashes)
        except Exception as e:
            logger.error(f"ChromaDB 임베딩된 해시 목록 조회 실패: {e}")
            return []

    def delete_by_url_hash(self, url_hash: str) -> bool:
        """특정 URL 해시의 모든 청크 삭제"""
        try:
            all_data = self.collection.get()
            ids_to_delete = [doc_id for doc_id in all_data.get("ids", [])
                            if doc_id.startswith(f"{url_hash}_")]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"ChromaDB URL 해시 {url_hash}의 {len(ids_to_delete)}개 청크 삭제")
                return True
            else:
                logger.warning(f"ChromaDB URL 해시 {url_hash}의 청크를 찾을 수 없음")
                return False
        except Exception as e:
            logger.error(f"ChromaDB URL 해시 삭제 실패: {e}")
            return False

    @staticmethod
    def list_all_collections() -> List[str]:
        """모든 컬렉션 목록 가져오기"""
        try:
            client = chromadb.PersistentClient(path=settings.vector_db_directory)
            collections = client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"ChromaDB 컬렉션 목록 조회 실패: {e}")
            return []

    @staticmethod
    def get_all_collections_info() -> List[Dict]:
        """모든 컬렉션의 상세 정보 가져오기"""
        try:
            client = chromadb.PersistentClient(path=settings.vector_db_directory)
            collections = client.list_collections()

            info_list = []
            for col in collections:
                try:
                    count = col.count()
                    metadata = col.metadata

                    info_list.append({
                        "name": col.name,
                        "document_count": count,
                        "metadata": metadata,
                        "id": col.id
                    })
                except Exception as e:
                    logger.error(f"ChromaDB 컬렉션 정보 조회 실패 {col.name}: {e}")

            return sorted(info_list, key=lambda x: x["document_count"], reverse=True)
        except Exception as e:
            logger.error(f"ChromaDB 전체 컬렉션 정보 조회 실패: {e}")
            return []

    @staticmethod
    def get_database_size() -> Dict:
        """ChromaDB 데이터베이스 크기 정보"""
        try:
            import os
            db_path = settings.vector_db_directory

            total_size = 0
            file_count = 0

            for dirpath, dirnames, filenames in os.walk(db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                    file_count += 1

            return {
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "file_count": file_count,
                "path": db_path,
                "type": "chromadb"
            }
        except Exception as e:
            logger.error(f"ChromaDB 크기 조회 실패: {e}")
            return {}
