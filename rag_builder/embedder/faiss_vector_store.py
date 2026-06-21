"""FAISS 기반 벡터 스토어"""
import os
import json
import numpy as np
from typing import List, Dict, Optional
from .base_vector_store import BaseVectorStore
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """FAISS 기반 벡터 스토어 구현"""

    def __init__(self, collection_name: str = "rag_documents", **kwargs):
        super().__init__(collection_name)
        self.persist_directory = kwargs.get('persist_directory', settings.vector_db_directory)
        self.dimension = kwargs.get('dimension', settings.embedding_dimension)

        # 파일 경로
        self.index_path = os.path.join(self.persist_directory, f"{collection_name}.faiss")
        self.metadata_path = os.path.join(self.persist_directory, f"{collection_name}_metadata.json")

        # FAISS 인덱스 및 메타데이터 로드
        self.index = None
        self.metadata_store = {}  # {index: {id, document, metadata}}
        self.id_to_index = {}  # {id: index}
        self.next_index = 0

        self._ensure_directory()
        self._load_or_create_index()

    def _ensure_directory(self):
        """저장 디렉토리 생성"""
        os.makedirs(self.persist_directory, exist_ok=True)

    def _load_or_create_index(self):
        """FAISS 인덱스 로드 또는 생성"""
        try:
            import faiss
        except ImportError:
            logger.error("FAISS 패키지가 설치되지 않았습니다. 'pip install faiss-cpu' 또는 'pip install faiss-gpu'를 실행하세요")
            return

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            # 기존 인덱스 로드
            try:
                self.index = faiss.read_index(self.index_path)

                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metadata_store = {int(k): v for k, v in data['metadata'].items()}
                    self.id_to_index = data['id_to_index']
                    self.next_index = data['next_index']

                logger.info(f"기존 FAISS 인덱스 로드: {self.collection_name} ({len(self.metadata_store)}개 문서)")
            except Exception as e:
                logger.error(f"FAISS 인덱스 로드 실패: {e}, 새로 생성합니다")
                self._create_new_index(faiss)
        else:
            # 새 인덱스 생성
            self._create_new_index(faiss)

    def _create_new_index(self, faiss):
        """새 FAISS 인덱스 생성"""
        # L2 거리 기반 인덱스 (코사인 유사도를 위해 정규화된 벡터 사용 시 동일)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = {}
        self.id_to_index = {}
        self.next_index = 0
        logger.info(f"새 FAISS 인덱스 생성: {self.collection_name} (차원: {self.dimension})")

    def _save_index(self):
        """FAISS 인덱스 및 메타데이터 저장"""
        try:
            import faiss

            # FAISS 인덱스 저장
            faiss.write_index(self.index, self.index_path)

            # 메타데이터 저장
            data = {
                'metadata': {str(k): v for k, v in self.metadata_store.items()},
                'id_to_index': self.id_to_index,
                'next_index': self.next_index
            }

            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"FAISS 인덱스 저장 완료: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"FAISS 인덱스 저장 실패: {e}")
            return False

    def add_documents(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
        ids: List[str]
    ) -> bool:
        """문서를 FAISS에 추가"""
        try:
            if not self.index:
                logger.error("FAISS 인덱스가 초기화되지 않았습니다")
                return False

            # 임베딩을 numpy 배열로 변환 및 정규화
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # L2 정규화 (코사인 유사도를 위함)
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms

            for i, (embedding, document, metadata, doc_id) in enumerate(zip(embeddings, documents, metadatas, ids)):
                # 기존 ID가 있으면 업데이트 (삭제 후 추가)
                if doc_id in self.id_to_index:
                    old_index = self.id_to_index[doc_id]
                    # FAISS는 삭제가 어려우므로 덮어쓰기
                    # 실제로는 새 인덱스로 추가하고 메타데이터만 업데이트
                    pass

                # 새 인덱스 할당
                current_index = self.next_index
                self.next_index += 1

                # 메타데이터 저장
                self.metadata_store[current_index] = {
                    'id': doc_id,
                    'document': document,
                    'metadata': metadata
                }

                # ID 매핑
                self.id_to_index[doc_id] = current_index

            # FAISS에 벡터 추가
            self.index.add(embeddings_array)

            # 저장
            self._save_index()

            logger.info(f"FAISS 문서 추가 완료: {len(ids)}개")
            return True

        except Exception as e:
            logger.error(f"FAISS 문서 추가 실패: {e}")
            return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """유사도 기반 문서 검색"""
        try:
            if not self.index or self.index.ntotal == 0:
                logger.warning("FAISS 인덱스가 비어있습니다")
                return []

            # 쿼리 임베딩 정규화
            query_array = np.array([query_embedding], dtype=np.float32)
            query_array = query_array / np.linalg.norm(query_array)

            # FAISS 검색
            distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))

            # 결과 포맷팅
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:  # 유효하지 않은 인덱스
                    continue

                if idx in self.metadata_store:
                    doc_data = self.metadata_store[idx]

                    # 필터 적용 (간단한 키-값 매칭)
                    if filter_dict:
                        match = all(
                            doc_data['metadata'].get(k) == v
                            for k, v in filter_dict.items()
                        )
                        if not match:
                            continue

                    results.append({
                        'document': doc_data['document'],
                        'metadata': doc_data['metadata'],
                        'distance': float(distance),
                        'id': doc_data['id']
                    })

            return results[:top_k]

        except Exception as e:
            logger.error(f"FAISS 검색 실패: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> bool:
        """ID로 문서 삭제 (메타데이터만 삭제, FAISS 인덱스는 재구축 필요)"""
        try:
            deleted_count = 0
            for doc_id in ids:
                if doc_id in self.id_to_index:
                    idx = self.id_to_index[doc_id]
                    if idx in self.metadata_store:
                        del self.metadata_store[idx]
                        del self.id_to_index[doc_id]
                        deleted_count += 1

            if deleted_count > 0:
                self._save_index()
                logger.info(f"FAISS 문서 메타데이터 삭제: {deleted_count}개 (인덱스 재구축 권장)")
                return True
            return False

        except Exception as e:
            logger.error(f"FAISS 문서 삭제 실패: {e}")
            return False

    def get_collection_stats(self) -> Dict:
        """컬렉션 통계 정보"""
        try:
            total_docs = len(self.metadata_store)
            return {
                "total_documents": total_docs,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "dimension": self.dimension,
                "faiss_index_size": self.index.ntotal if self.index else 0,
                "type": "faiss"
            }
        except Exception as e:
            logger.error(f"FAISS 통계 조회 실패: {e}")
            return {}

    def delete_collection(self) -> bool:
        """컬렉션 삭제 (파일 삭제)"""
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)

            self.index = None
            self.metadata_store = {}
            self.id_to_index = {}
            self.next_index = 0

            logger.info(f"FAISS 컬렉션 삭제: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"FAISS 컬렉션 삭제 실패: {e}")
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
                    # ID 검색
                    chunk_id = f"{url_hash}_0"
                    if chunk_id in self.id_to_index:
                        duplicates.append({
                            "filename": filename,
                            "url_hash": url_hash,
                            "path": file_path
                        })
                    else:
                        new_files.append(file_path)
                else:
                    new_files.append(file_path)

            return {
                "duplicates": duplicates,
                "new_files": new_files
            }

        except Exception as e:
            logger.error(f"FAISS 중복 체크 실패: {e}")
            return {"duplicates": [], "new_files": file_paths}

    def get_embedded_hashes(self) -> List[str]:
        """임베딩된 모든 URL 해시 목록 가져오기"""
        try:
            hashes = set()
            for doc_id in self.id_to_index.keys():
                if "_" in doc_id:
                    url_hash = doc_id.rsplit("_", 1)[0]
                    hashes.add(url_hash)
            return list(hashes)
        except Exception as e:
            logger.error(f"FAISS 임베딩된 해시 목록 조회 실패: {e}")
            return []

    def delete_by_url_hash(self, url_hash: str) -> bool:
        """특정 URL 해시의 모든 청크 삭제"""
        try:
            ids_to_delete = [doc_id for doc_id in self.id_to_index.keys()
                            if doc_id.startswith(f"{url_hash}_")]

            if ids_to_delete:
                return self.delete_by_ids(ids_to_delete)
            else:
                logger.warning(f"FAISS URL 해시 {url_hash}의 청크를 찾을 수 없음")
                return False
        except Exception as e:
            logger.error(f"FAISS URL 해시 삭제 실패: {e}")
            return False

    @staticmethod
    def list_all_collections() -> List[str]:
        """모든 컬렉션 목록 가져오기 (.faiss 파일 목록)"""
        try:
            db_path = settings.vector_db_directory
            if not os.path.exists(db_path):
                return []

            collections = []
            for filename in os.listdir(db_path):
                if filename.endswith('.faiss'):
                    collection_name = filename.replace('.faiss', '')
                    collections.append(collection_name)

            return sorted(collections)
        except Exception as e:
            logger.error(f"FAISS 컬렉션 목록 조회 실패: {e}")
            return []

    @staticmethod
    def get_all_collections_info() -> List[Dict]:
        """모든 컬렉션의 상세 정보 가져오기"""
        try:
            collections = FAISSVectorStore.list_all_collections()
            info_list = []

            for collection_name in collections:
                try:
                    # 임시 인스턴스 생성하여 정보 가져오기
                    store = FAISSVectorStore(collection_name)
                    stats = store.get_collection_stats()

                    info_list.append({
                        "name": collection_name,
                        "document_count": stats.get("total_documents", 0),
                        "metadata": {"type": "faiss", "dimension": stats.get("dimension", 0)},
                        "id": collection_name  # FAISS는 ID가 없으므로 이름 사용
                    })
                except Exception as e:
                    logger.error(f"FAISS 컬렉션 정보 조회 실패 {collection_name}: {e}")

            return sorted(info_list, key=lambda x: x["document_count"], reverse=True)
        except Exception as e:
            logger.error(f"FAISS 전체 컬렉션 정보 조회 실패: {e}")
            return []

    @staticmethod
    def get_database_size() -> Dict:
        """FAISS 데이터베이스 크기 정보"""
        try:
            db_path = settings.vector_db_directory

            total_size = 0
            file_count = 0

            if os.path.exists(db_path):
                for filename in os.listdir(db_path):
                    if filename.endswith('.faiss') or filename.endswith('_metadata.json'):
                        filepath = os.path.join(db_path, filename)
                        total_size += os.path.getsize(filepath)
                        file_count += 1

            return {
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "file_count": file_count,
                "path": db_path,
                "type": "faiss"
            }
        except Exception as e:
            logger.error(f"FAISS 크기 조회 실패: {e}")
            return {}
