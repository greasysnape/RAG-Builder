import os
import glob
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from datetime import datetime
from .logger import get_logger

logger = get_logger(__name__)

class FileManager:
    """파일 관리 유틸리티 클래스"""

    def __init__(self, base_folder: str = "./downloads"):
        self.base_folder = Path(base_folder)
        self.base_folder.mkdir(parents=True, exist_ok=True)

        # 태그 데이터 파일
        self.tags_file = self.base_folder / "tags.json"
        self.tags_data = self._load_tags()

    def get_all_folders(self) -> List[Dict]:
        """모든 하위 폴더 목록 가져오기"""
        try:
            folders = []
            for item in self.base_folder.iterdir():
                if item.is_dir():
                    file_count = len(list(item.glob("*.md")))
                    total_size = sum(f.stat().st_size for f in item.glob("*.md"))

                    folders.append({
                        "name": item.name,
                        "path": str(item),
                        "file_count": file_count,
                        "total_size": total_size,
                        "modified_time": datetime.fromtimestamp(item.stat().st_mtime)
                    })

            return sorted(folders, key=lambda x: x["modified_time"], reverse=True)
        except Exception as e:
            logger.error(f"폴더 목록 가져오기 실패: {e}")
            return []

    def get_files_in_folder(self, folder_name: str = None) -> List[Dict]:
        """특정 폴더의 파일 목록 가져오기"""
        try:
            if folder_name:
                search_path = self.base_folder / folder_name
            else:
                search_path = self.base_folder

            if not search_path.exists():
                return []

            files = []
            for file_path in search_path.glob("*.md"):
                stat_info = file_path.stat()

                # 파일에서 메타데이터 읽기
                metadata = self._extract_metadata_from_file(file_path)

                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": stat_info.st_size,
                    "size_mb": stat_info.st_size / (1024 * 1024),
                    "modified_time": datetime.fromtimestamp(stat_info.st_mtime),
                    "created_time": datetime.fromtimestamp(stat_info.st_ctime),
                    "url": metadata.get("url", ""),
                    "title": metadata.get("title", ""),
                    "url_hash": file_path.stem  # 파일명이 해시값
                })

            return sorted(files, key=lambda x: x["modified_time"], reverse=True)
        except Exception as e:
            logger.error(f"파일 목록 가져오기 실패: {e}")
            return []

    def _extract_metadata_from_file(self, file_path: Path) -> Dict:
        """마크다운 파일에서 메타데이터 추출"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1000)  # 앞부분만 읽기

                metadata = {}
                if content.startswith('---'):
                    # YAML 프론트매터 파싱
                    end_idx = content.find('---', 3)
                    if end_idx > 0:
                        frontmatter = content[3:end_idx]
                        for line in frontmatter.split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                metadata[key.strip()] = value.strip()

                return metadata
        except Exception as e:
            logger.error(f"메타데이터 추출 실패 {file_path}: {e}")
            return {}

    def read_file_content(self, file_path: str, max_length: int = 5000) -> str:
        """파일 내용 읽기 (미리보기용)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(max_length)
                if len(content) == max_length:
                    content += "\n\n... (내용이 더 있습니다)"
                return content
        except Exception as e:
            logger.error(f"파일 읽기 실패 {file_path}: {e}")
            return f"오류: {e}"

    def delete_file(self, file_path: str) -> bool:
        """파일 삭제"""
        try:
            Path(file_path).unlink()
            logger.info(f"파일 삭제: {file_path}")
            return True
        except Exception as e:
            logger.error(f"파일 삭제 실패 {file_path}: {e}")
            return False

    def delete_folder(self, folder_path: str) -> bool:
        """폴더 삭제 (폴더 내 모든 파일 포함)"""
        try:
            import shutil
            shutil.rmtree(folder_path)
            logger.info(f"폴더 삭제: {folder_path}")
            return True
        except Exception as e:
            logger.error(f"폴더 삭제 실패 {folder_path}: {e}")
            return False

    def get_all_files_recursive(self) -> List[Dict]:
        """모든 마크다운 파일 재귀적으로 가져오기"""
        try:
            files = []
            for file_path in self.base_folder.glob("**/*.md"):
                stat_info = file_path.stat()
                metadata = self._extract_metadata_from_file(file_path)

                # 상대 경로 계산
                relative_path = file_path.relative_to(self.base_folder)
                folder_name = str(relative_path.parent) if relative_path.parent != Path('.') else "root"

                files.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "folder": folder_name,
                    "size": stat_info.st_size,
                    "size_mb": stat_info.st_size / (1024 * 1024),
                    "modified_time": datetime.fromtimestamp(stat_info.st_mtime),
                    "url": metadata.get("url", ""),
                    "title": metadata.get("title", ""),
                    "url_hash": file_path.stem
                })

            return files
        except Exception as e:
            logger.error(f"전체 파일 목록 가져오기 실패: {e}")
            return []

    def find_duplicate_files(self) -> Dict:
        """중복 파일 찾기 (같은 URL 해시)"""
        try:
            files = self.get_all_files_recursive()
            hash_map = {}

            for file_info in files:
                url_hash = file_info["url_hash"]
                if url_hash not in hash_map:
                    hash_map[url_hash] = []
                hash_map[url_hash].append(file_info)

            # 중복된 것만 필터링
            duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}

            return {
                "duplicate_count": len(duplicates),
                "duplicates": duplicates,
                "total_duplicate_files": sum(len(v) for v in duplicates.values())
            }
        except Exception as e:
            logger.error(f"중복 파일 찾기 실패: {e}")
            return {"duplicate_count": 0, "duplicates": {}, "total_duplicate_files": 0}

    def get_statistics(self) -> Dict:
        """전체 통계 정보"""
        try:
            files = self.get_all_files_recursive()
            folders = self.get_all_folders()

            total_size = sum(f["size"] for f in files)

            # URL 도메인 분석
            domain_count = {}
            for file_info in files:
                url = file_info.get("url", "")
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        domain_count[domain] = domain_count.get(domain, 0) + 1
                    except:
                        pass

            # 상위 도메인
            top_domains = sorted(domain_count.items(), key=lambda x: x[1], reverse=True)[:10]

            # 날짜별 다운로드 수
            date_count = {}
            for file_info in files:
                date_str = file_info["modified_time"].strftime("%Y-%m-%d")
                date_count[date_str] = date_count.get(date_str, 0) + 1

            return {
                "total_files": len(files),
                "total_folders": len(folders),
                "total_size": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "top_domains": top_domains,
                "date_distribution": date_count,
                "avg_file_size_mb": (total_size / len(files) / (1024 * 1024)) if files else 0
            }
        except Exception as e:
            logger.error(f"통계 생성 실패: {e}")
            return {}

    def search_files(self, query: str, search_in_content: bool = False) -> List[Dict]:
        """파일 검색"""
        try:
            files = self.get_all_files_recursive()
            results = []

            query_lower = query.lower()

            for file_info in files:
                # 파일명, 제목, URL에서 검색
                if (query_lower in file_info["filename"].lower() or
                    query_lower in file_info.get("title", "").lower() or
                    query_lower in file_info.get("url", "").lower()):
                    results.append(file_info)
                    continue

                # 내용 검색 (옵션)
                if search_in_content:
                    content = self.read_file_content(file_info["path"])
                    if query_lower in content.lower():
                        results.append(file_info)

            return results
        except Exception as e:
            logger.error(f"파일 검색 실패: {e}")
            return []

    def get_orphan_files(self, embedded_hashes: List[str]) -> List[Dict]:
        """임베딩되지 않은 파일 찾기 (고아 파일)"""
        try:
            all_files = self.get_all_files_recursive()
            orphans = []

            for file_info in all_files:
                if file_info["url_hash"] not in embedded_hashes:
                    orphans.append(file_info)

            return orphans
        except Exception as e:
            logger.error(f"고아 파일 찾기 실패: {e}")
            return []

    def rename_file(self, old_path: str, new_name: str) -> bool:
        """파일 이름 변경"""
        try:
            old_path = Path(old_path)
            new_path = old_path.parent / new_name
            old_path.rename(new_path)
            logger.info(f"파일 이름 변경: {old_path} -> {new_path}")
            return True
        except Exception as e:
            logger.error(f"파일 이름 변경 실패: {e}")
            return False

    def get_folder_size(self, folder_name: str) -> int:
        """폴더 크기 계산"""
        try:
            folder_path = self.base_folder / folder_name
            total_size = sum(f.stat().st_size for f in folder_path.glob("**/*") if f.is_file())
            return total_size
        except Exception as e:
            logger.error(f"폴더 크기 계산 실패: {e}")
            return 0

    # ==================== 태그 시스템 ====================

    def _load_tags(self) -> Dict:
        """태그 데이터 로드"""
        try:
            if self.tags_file.exists():
                with open(self.tags_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"태그 데이터 로드 실패: {e}")
            return {}

    def _save_tags(self) -> bool:
        """태그 데이터 저장"""
        try:
            with open(self.tags_file, 'w', encoding='utf-8') as f:
                json.dump(self.tags_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"태그 데이터 저장 실패: {e}")
            return False

    def add_tag_to_file(self, file_path: str, tag: str) -> bool:
        """파일에 태그 추가"""
        try:
            # 파일 경로를 키로 사용
            if file_path not in self.tags_data:
                self.tags_data[file_path] = {
                    "tags": [],
                    "note": "",
                    "favorite": False,
                    "created_at": datetime.now().isoformat()
                }

            # 태그 추가 (중복 방지)
            if tag not in self.tags_data[file_path]["tags"]:
                self.tags_data[file_path]["tags"].append(tag)
                self.tags_data[file_path]["updated_at"] = datetime.now().isoformat()
                self._save_tags()
                logger.info(f"태그 추가: {file_path} -> {tag}")
                return True

            return False
        except Exception as e:
            logger.error(f"태그 추가 실패: {e}")
            return False

    def remove_tag_from_file(self, file_path: str, tag: str) -> bool:
        """파일에서 태그 제거"""
        try:
            if file_path in self.tags_data and tag in self.tags_data[file_path]["tags"]:
                self.tags_data[file_path]["tags"].remove(tag)
                self.tags_data[file_path]["updated_at"] = datetime.now().isoformat()
                self._save_tags()
                logger.info(f"태그 제거: {file_path} -> {tag}")
                return True
            return False
        except Exception as e:
            logger.error(f"태그 제거 실패: {e}")
            return False

    def get_file_tags(self, file_path: str) -> List[str]:
        """파일의 태그 목록 가져오기"""
        try:
            if file_path in self.tags_data:
                return self.tags_data[file_path].get("tags", [])
            return []
        except Exception as e:
            logger.error(f"태그 조회 실패: {e}")
            return []

    def get_all_tags(self) -> Dict[str, int]:
        """모든 태그와 사용 횟수"""
        try:
            tag_count = {}
            for file_data in self.tags_data.values():
                for tag in file_data.get("tags", []):
                    tag_count[tag] = tag_count.get(tag, 0) + 1

            # 사용 횟수 순으로 정렬
            return dict(sorted(tag_count.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"전체 태그 조회 실패: {e}")
            return {}

    def get_files_by_tag(self, tag: str) -> List[str]:
        """특정 태그가 있는 파일 목록"""
        try:
            files = []
            for file_path, file_data in self.tags_data.items():
                if tag in file_data.get("tags", []):
                    files.append(file_path)
            return files
        except Exception as e:
            logger.error(f"태그별 파일 조회 실패: {e}")
            return []

    def set_file_note(self, file_path: str, note: str) -> bool:
        """파일에 메모 추가"""
        try:
            if file_path not in self.tags_data:
                self.tags_data[file_path] = {
                    "tags": [],
                    "note": "",
                    "favorite": False,
                    "created_at": datetime.now().isoformat()
                }

            self.tags_data[file_path]["note"] = note
            self.tags_data[file_path]["updated_at"] = datetime.now().isoformat()
            self._save_tags()
            return True
        except Exception as e:
            logger.error(f"메모 저장 실패: {e}")
            return False

    def get_file_note(self, file_path: str) -> str:
        """파일의 메모 가져오기"""
        try:
            if file_path in self.tags_data:
                return self.tags_data[file_path].get("note", "")
            return ""
        except Exception as e:
            logger.error(f"메모 조회 실패: {e}")
            return ""

    def toggle_favorite(self, file_path: str) -> bool:
        """파일 즐겨찾기 토글"""
        try:
            if file_path not in self.tags_data:
                self.tags_data[file_path] = {
                    "tags": [],
                    "note": "",
                    "favorite": False,
                    "created_at": datetime.now().isoformat()
                }

            current = self.tags_data[file_path].get("favorite", False)
            self.tags_data[file_path]["favorite"] = not current
            self.tags_data[file_path]["updated_at"] = datetime.now().isoformat()
            self._save_tags()
            return True
        except Exception as e:
            logger.error(f"즐겨찾기 토글 실패: {e}")
            return False

    def is_favorite(self, file_path: str) -> bool:
        """파일이 즐겨찾기인지 확인"""
        try:
            if file_path in self.tags_data:
                return self.tags_data[file_path].get("favorite", False)
            return False
        except Exception as e:
            logger.error(f"즐겨찾기 확인 실패: {e}")
            return False

    def get_favorite_files(self) -> List[str]:
        """즐겨찾기 파일 목록"""
        try:
            favorites = []
            for file_path, file_data in self.tags_data.items():
                if file_data.get("favorite", False):
                    favorites.append(file_path)
            return favorites
        except Exception as e:
            logger.error(f"즐겨찾기 목록 조회 실패: {e}")
            return []

    def get_file_metadata(self, file_path: str) -> Dict:
        """파일의 모든 메타데이터 가져오기"""
        try:
            if file_path in self.tags_data:
                return self.tags_data[file_path]
            return {
                "tags": [],
                "note": "",
                "favorite": False
            }
        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {e}")
            return {"tags": [], "note": "", "favorite": False}
