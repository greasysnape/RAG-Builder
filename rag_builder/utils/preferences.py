"""사용자 설정 저장 및 불러오기"""
import json
from pathlib import Path
from typing import Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


class UserPreferences:
    """사용자 설정 관리 클래스"""

    def __init__(self, preferences_file: str = ".rag_builder_preferences.json"):
        """
        Args:
            preferences_file: 설정 파일명 (프로젝트 루트에 저장)
        """
        # 프로젝트 루트 디렉토리 찾기
        self.preferences_path = Path.cwd() / preferences_file
        self._defaults = {
            "language": "en",
            "embedding_type": "sentence_transformer",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db_type": "chromadb"
        }

    def load(self) -> Dict[str, Any]:
        """
        저장된 설정 불러오기

        Returns:
            설정 딕셔너리. 파일이 없으면 기본값 반환
        """
        try:
            if self.preferences_path.exists():
                with open(self.preferences_path, 'r', encoding='utf-8') as f:
                    preferences = json.load(f)
                    logger.info(f"사용자 설정 불러오기 완료: {self.preferences_path}")
                    return {**self._defaults, **preferences}  # 기본값과 병합
            else:
                logger.info("저장된 설정 없음. 기본값 사용")
                return self._defaults.copy()
        except Exception as e:
            logger.warning(f"설정 불러오기 실패 (기본값 사용): {e}")
            return self._defaults.copy()

    def save(self, preferences: Dict[str, Any]) -> bool:
        """
        설정 저장

        Args:
            preferences: 저장할 설정 딕셔너리

        Returns:
            성공 여부
        """
        try:
            # 지원되는 키만 저장
            valid_keys = set(self._defaults.keys())
            filtered_preferences = {
                k: v for k, v in preferences.items()
                if k in valid_keys
            }

            with open(self.preferences_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_preferences, f, indent=2, ensure_ascii=False)

            logger.info(f"사용자 설정 저장 완료: {self.preferences_path}")
            return True
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
            return False

    def update(self, **kwargs) -> bool:
        """
        특정 설정값만 업데이트

        Args:
            **kwargs: 업데이트할 설정 (language="en", embedding_type="openai" 등)

        Returns:
            성공 여부
        """
        try:
            # 기존 설정 불러오기
            current = self.load()

            # 업데이트
            current.update(kwargs)

            # 저장
            return self.save(current)
        except Exception as e:
            logger.error(f"설정 업데이트 실패: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        특정 설정값 가져오기

        Args:
            key: 설정 키
            default: 기본값

        Returns:
            설정값
        """
        preferences = self.load()
        return preferences.get(key, default)

    def reset(self) -> bool:
        """
        설정 초기화 (기본값으로 리셋)

        Returns:
            성공 여부
        """
        try:
            if self.preferences_path.exists():
                self.preferences_path.unlink()
                logger.info("사용자 설정 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"설정 초기화 실패: {e}")
            return False
