import os
from pathlib import Path
from typing import List, Dict, Optional
import json
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FileHandler:
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.html'}

    def find_documents(self, directory: str) -> List[str]:
        """디렉토리에서 지원되는 문서 파일들 찾기"""
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"디렉토리가 존재하지 않습니다: {directory}")
            return []

        document_files = []
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                document_files.append(str(file_path))

        logger.info(f"발견된 문서 파일: {len(document_files)}개")
        return document_files

    def read_text_file(self, file_path: str) -> Optional[str]:
        """텍스트 파일 읽기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"파일 읽기 실패 {file_path}: {e}")
            return None

    def save_json(self, data: Dict, file_path: str) -> bool:
        """JSON 파일로 저장"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON 저장 완료: {file_path}")
            return True
        except Exception as e:
            logger.error(f"JSON 저장 실패: {e}")
            return False

    def load_json(self, file_path: str) -> Optional[Dict]:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"JSON 로드 실패: {e}")
            return None

    def create_directory(self, directory: str) -> bool:
        """디렉토리 생성"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"디렉토리 생성 실패: {e}")
            return False

    def load_file(self, file_path: str) -> Optional[str]:
        """다양한 파일 형식을 로드하여 텍스트로 반환"""
        if not os.path.exists(file_path):
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return None

        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension in ['.txt', '.md']:
                return self._load_text_file(file_path)
            elif file_extension == '.pdf':
                return self._load_pdf_file(file_path)
            elif file_extension in ['.doc', '.docx']:
                return self._load_word_file(file_path)
            elif file_extension in ['.html', '.htm']:
                return self._load_html_file(file_path)
            else:
                logger.warning(f"지원되지 않는 파일 형식: {file_extension}")
                # 텍스트 파일로 시도
                return self._load_text_file(file_path)

        except Exception as e:
            logger.error(f"파일 로드 실패 {file_path}: {e}")
            return None

    def _load_text_file(self, file_path: str) -> Optional[str]:
        """텍스트 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # UTF-8로 실패하면 다른 인코딩 시도
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    return f.read()
            except:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()

    def _load_pdf_file(self, file_path: str) -> Optional[str]:
        """PDF 파일 로드 (PyPDF2 사용)"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except ImportError:
            logger.warning("PyPDF2가 설치되지 않았습니다. pip install PyPDF2")
            return None
        except Exception as e:
            logger.error(f"PDF 파일 로드 실패: {e}")
            return None

    def _load_word_file(self, file_path: str) -> Optional[str]:
        """Word 파일 로드 (python-docx 사용)"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except ImportError:
            logger.warning("python-docx가 설치되지 않았습니다. pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"Word 파일 로드 실패: {e}")
            return None

    def _load_html_file(self, file_path: str) -> Optional[str]:
        """HTML 파일 로드 (BeautifulSoup 사용)"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                return soup.get_text()
        except ImportError:
            logger.warning("beautifulsoup4가 설치되지 않았습니다. pip install beautifulsoup4")
            # HTML 태그를 제거하지 않고 텍스트로 읽기
            return self._load_text_file(file_path)
        except Exception as e:
            logger.error(f"HTML 파일 로드 실패: {e}")
            return self._load_text_file(file_path)

    def get_file_info(self, file_path: str) -> Dict:
        """파일 정보 가져오기"""
        path = Path(file_path)
        if not path.exists():
            return {}

        return {
            "name": path.name,
            "size": path.stat().st_size,
            "extension": path.suffix,
            "modified": path.stat().st_mtime,
            "absolute_path": str(path.absolute())
        }