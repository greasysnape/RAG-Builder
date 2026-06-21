"""다국어 번역 유틸리티"""
import json
import os
from typing import Dict, Any


class Translator:
    """JSON 기반 다국어 번역 클래스"""

    SUPPORTED_LANGUAGES = {
        'en': '🇬🇧 English',
        'fr': '🇫🇷 Français',
        'ko': '🇰🇷 한국어'
    }

    def __init__(self, lang: str = 'en'):
        """
        Args:
            lang: 언어 코드 ('en', 'fr', 'kr')
        """
        self.lang = lang if lang in self.SUPPORTED_LANGUAGES else 'en'
        self.translations: Dict[str, Any] = {}
        self.load_translations()

    def load_translations(self):
        """현재 언어의 번역 파일을 로드"""
        locale_path = os.path.join(os.path.dirname(__file__), f'{self.lang}.json')

        try:
            with open(locale_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Translation file not found: {locale_path}")
            self.translations = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {locale_path}: {e}")
            self.translations = {}

    def t(self, key_path: str, **kwargs) -> str:
        """

        Args:
            key_path: 점으로 구분된 키 경로 (예: 'ui.title')
            **kwargs: 포맷팅에 사용할 변수들

        Returns:
            번역된 텍스트 (없으면 key_path 반환)
            
        """
        keys = key_path.split('.')
        value = self.translations

        # 중첩된 딕셔너리 탐색
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return key_path
            else:
                return key_path

        # 문자열 포맷팅 적용
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except (KeyError, ValueError):
                return value

        return str(value) if value is not None else key_path

    def change_language(self, lang: str):
        """
        언어 변경

        Args:
            lang: 새 언어 코드
        """
        if lang in self.SUPPORTED_LANGUAGES:
            self.lang = lang
            self.load_translations()

    def get_current_language(self) -> str:
        """현재 언어 코드 반환"""
        return self.lang

    def get_language_name(self) -> str:
        """현재 언어 이름 반환"""
        return self.SUPPORTED_LANGUAGES.get(self.lang, self.lang)

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """지원하는 모든 언어 목록 반환"""
        return cls.SUPPORTED_LANGUAGES.copy()
