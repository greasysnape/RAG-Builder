"""PDF to Markdown 변환 모듈 (Dolphin 사용)"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PDFConverter:
    """Dolphin을 사용한 PDF → Markdown 변환기"""

    def __init__(self, dolphin_path: str = None):
        """
        Args:
            dolphin_path: Dolphin 레포지토리 경로 (기본값: ./Dolphin)
        """
        if dolphin_path is None:
            # 프로젝트 루트의 Dolphin 디렉토리 사용
            self.dolphin_path = Path(__file__).parent.parent.parent / "Dolphin"
        else:
            self.dolphin_path = Path(dolphin_path)

        self.demo_script = self.dolphin_path / "demo_page.py"
        self.config_path = self.dolphin_path / "config" / "Dolphin.yaml"
        self.checkpoints_path = self.dolphin_path / "checkpoints"

    def is_dolphin_installed(self) -> bool:
        """Dolphin이 설치되어 있는지 확인"""
        return (
            self.dolphin_path.exists() and
            self.demo_script.exists() and
            self.config_path.exists() and
            self.checkpoints_path.exists()
        )

    def convert_pdf_to_markdown(
        self,
        pdf_path: str,
        output_dir: str = None,
        max_batch_size: int = 4
    ) -> Optional[str]:
        """
        PDF 파일을 Markdown으로 변환

        Args:
            pdf_path: 입력 PDF 파일 경로
            output_dir: 출력 디렉토리 (None이면 임시 디렉토리 사용)
            max_batch_size: 병렬 처리 배치 크기

        Returns:
            변환된 Markdown 파일 경로 (실패 시 None)
        """
        try:
            if not self.is_dolphin_installed():
                logger.error(f"Dolphin이 설치되지 않았습니다: {self.dolphin_path}")
                if not self.dolphin_path.exists():
                    logger.error("설치 방법: cd <project_root> && git clone https://github.com/ByteDance/Dolphin.git")
                if not self.checkpoints_path.exists():
                    logger.error("모델 다운로드가 필요합니다. checkpoints 디렉토리가 없습니다.")
                return None

            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
                return None

            # 출력 디렉토리 설정
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="dolphin_output_")
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"PDF 변환 시작: {pdf_path.name}")

            # Dolphin CLI 호출 (config 기반)
            cmd = [
                "python", str(self.demo_script),
                "--config", str(self.config_path),
                "--input_path", str(pdf_path),
                "--save_dir", str(output_dir),
                "--max_batch_size", str(max_batch_size)
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.dolphin_path),
                capture_output=True,
                text=True,
                timeout=300  # 5분 타임아웃
            )

            if result.returncode != 0:
                logger.error(f"Dolphin 실행 실패 (exit code {result.returncode})")
                logger.error(f"Command: {' '.join(cmd)}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return None
            else:
                logger.info(f"Dolphin 실행 성공")
                logger.info(f"STDOUT: {result.stdout[:500]}")  # 처음 500자만

            # 변환된 Markdown 파일 찾기
            markdown_files = list(Path(output_dir).glob("**/*.md"))  # 하위 디렉토리도 검색
            logger.info(f"출력 디렉토리 내용: {list(Path(output_dir).iterdir())}")
            logger.info(f"찾은 Markdown 파일: {markdown_files}")

            if not markdown_files:
                logger.error(f"변환된 Markdown 파일을 찾을 수 없습니다. 출력 디렉토리: {output_dir}")
                return None

            # 가장 최근 파일 반환
            markdown_file = sorted(markdown_files, key=lambda x: x.stat().st_mtime)[-1]
            logger.info(f"PDF 변환 완료: {markdown_file}")

            return str(markdown_file)

        except subprocess.TimeoutExpired:
            logger.error(f"PDF 변환 타임아웃 (5분 초과): {pdf_path}")
            return None
        except Exception as e:
            logger.error(f"PDF 변환 실패: {e}", exc_info=True)
            return None

    def convert_multiple_pdfs(
        self,
        pdf_paths: List[str],
        output_dir: str
    ) -> List[str]:
        """
        여러 PDF 파일을 일괄 변환

        Args:
            pdf_paths: PDF 파일 경로 리스트
            output_dir: 출력 디렉토리

        Returns:
            변환된 Markdown 파일 경로 리스트
        """
        markdown_files = []

        for pdf_path in pdf_paths:
            md_path = self.convert_pdf_to_markdown(pdf_path, output_dir)
            if md_path:
                markdown_files.append(md_path)

        logger.info(f"총 {len(markdown_files)}/{len(pdf_paths)}개 PDF 변환 완료")
        return markdown_files

    def convert_and_replace(self, file_path: str, output_folder: str = "./downloads") -> str:
        """
        PDF 파일을 Markdown으로 변환하고 downloads 폴더에 저장

        Args:
            file_path: PDF 파일 경로
            output_folder: 출력 폴더 (기본값: ./downloads)

        Returns:
            변환된 Markdown 파일 경로 (실패 시 원본 경로 반환)
        """
        try:
            file_path = Path(file_path)

            # PDF가 아니면 원본 반환
            if file_path.suffix.lower() != '.pdf':
                return str(file_path)

            # 출력 폴더 생성
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)

            # 임시 디렉토리에 변환
            temp_output = tempfile.mkdtemp(prefix="pdf_convert_")
            converted_md = self.convert_pdf_to_markdown(str(file_path), temp_output)

            if converted_md is None:
                logger.warning(f"PDF 변환 실패, 원본 파일 유지: {file_path}")
                return str(file_path)

            # 변환된 파일을 downloads 폴더에 저장
            output_path = output_folder / f"{file_path.stem}.md"
            shutil.copy(converted_md, output_path)

            # 임시 디렉토리 정리
            shutil.rmtree(temp_output, ignore_errors=True)

            logger.info(f"PDF → MD 변환 완료: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"PDF 변환 중 오류: {e}")
            return str(file_path)
