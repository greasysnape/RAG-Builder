"""통합 웹 문서 크롤러 - Perplexity Search + Firecrawl Scrape"""
import re
import json
import time
import hashlib
import aiohttp
from typing import List, Dict, Optional, Callable
from pathlib import Path
from tqdm import tqdm
from ..config.settings import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class WebDocumentCrawler:
    """Perplexity Search + Firecrawl Scrape를 통한 웹 문서 크롤러"""

    def __init__(self):
        self.perplexity_api_key = settings.perplexity_api_key
        self.perplexity_api_url = "https://api.perplexity.ai/chat/completions"
        self.firecrawl_api_key = settings.firecrawl_api_key

    # ============================================
    # Perplexity URL 수집
    # ============================================

    async def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perplexity API를 사용하여 웹 검색 수행

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수

        Returns:
            URL 정보 리스트 [{"url": str, "title": str, "description": str, "score": float}]
        """
        try:
            if not self.perplexity_api_key:
                logger.error("Perplexity API 키가 설정되지 않았습니다")
                return []

            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }

            # Perplexity에게 URL 리스트를 반환하도록 요청하는 시스템 프롬프트
            system_prompt = """You are a search assistant. When given a query, search for relevant web pages that user has suggested. Review every web pages before returning a list of relevant URLs with their titles in JSON format, to make sure they meet the query.
Format your response as a JSON array:
[
  {"title": "Page Title", "url": "https://example.com"},
  ...
]
Return only the JSON array, no additional text."""

            payload = {
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{query} (max {max_results} results)"}
                ],
                "temperature": 0.2,
                "max_tokens": 2000,
                "return_citations": True,
                "return_images": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.perplexity_api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_perplexity_response(result, query, max_results)
                    else:
                        error_text = await response.text()
                        logger.error(f"Perplexity API HTTP {response.status}: {error_text}")
                        return []

        except Exception as e:
            logger.error(f"Perplexity 검색 실패: {e}")
            return []

    def _parse_perplexity_response(self, response: Dict, query: str, max_results: int) -> List[Dict]:
        """Perplexity API 응답 파싱"""
        try:
            urls = []

            # citations에서 URL 추출 (Perplexity의 기본 기능)
            citations = response.get("citations", [])
            if citations:
                for i, citation in enumerate(citations[:max_results]):
                    urls.append({
                        "url": citation,
                        "title": f"Document {i+1}",
                        "description": f"Related to {query}",
                        "score": 1.0 - (i * 0.05)  # 순위에 따른 점수
                    })

            # 응답 내용에서 JSON 추출 시도
            if not urls:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    try:
                        # JSON 배열 추출
                        json_match = re.search(r'\[[\s\S]*\]', content)
                        if json_match:
                            parsed_urls = json.loads(json_match.group())
                            for i, item in enumerate(parsed_urls[:max_results]):
                                if isinstance(item, dict) and "url" in item:
                                    urls.append({
                                        "url": item.get("url", ""),
                                        "title": item.get("title", f"Document {i+1}"),
                                        "description": item.get("description", f"Related to {query}")[:200],
                                        "score": item.get("score", 1.0 - (i * 0.05))
                                    })
                    except json.JSONDecodeError:
                        logger.warning("응답에서 JSON 파싱 실패")

            # 결과 요약 로그
            if len(urls) < max_results:
                logger.info(f"Perplexity 검색 완료: {len(urls)}개 URL 발견 (요청: {max_results}개)")
            else:
                logger.info(f"Perplexity 검색 완료: {len(urls)}개 URL 발견")

            return urls

        except Exception as e:
            logger.error(f"Perplexity 응답 파싱 실패: {e}")
            return []

    async def search_with_firecrawl(self, query: str, max_results: int = 10, categories: List[str] = None) -> List[Dict]:
        """
        Firecrawl Search API를 사용하여 웹 검색 수행 (카테고리 지원)

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            categories: 검색 카테고리 리스트 (예: ["github", "research"])

        Returns:
            URL 정보 리스트 [{"url": str, "title": str, "description": str}]
        """
        try:
            if not self.firecrawl_api_key:
                logger.error("Firecrawl API 키가 설정되지 않았습니다")
                return []

            api_url = "https://api.firecrawl.dev/v2/search"

            headers = {
                "Authorization": f"Bearer {self.firecrawl_api_key}",
                "Content-Type": "application/json"
            }

            # Request body 구성
            payload = {
                "query": query,
                "sources": ["web"],  # web 검색 소스 지정
                "limit": max_results
            }

            # categories 파라미터 추가 (Firecrawl 형식: ["github", "research"])
            if categories:
                payload["categories"] = categories
                logger.info(f"Firecrawl Search (카테고리: {', '.join(categories)})")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()

                        # 디버깅: API 응답 구조 로깅
                        logger.info(f"Firecrawl API 응답 구조: {json.dumps(result, indent=2, ensure_ascii=False)[:2000]}")

                        if result.get("success"):
                            urls = []
                            data = result.get("data", [])

                            # data가 리스트인 경우 (v2 API)
                            if isinstance(data, list):
                                for i, item in enumerate(data[:max_results]):
                                    urls.append({
                                        "url": item.get("url", ""),
                                        "title": item.get("title", f"Document {i+1}"),
                                        "description": item.get("description", "")[:200],
                                        "score": 1.0 - (i * 0.05)
                                    })
                            # data가 딕셔너리인 경우 (web 필드 확인)
                            elif isinstance(data, dict):
                                web_results = data.get("web", [])
                                for i, item in enumerate(web_results[:max_results]):
                                    urls.append({
                                        "url": item.get("url", ""),
                                        "title": item.get("title", f"Document {i+1}"),
                                        "description": item.get("description", "")[:200],
                                        "score": 1.0 - (i * 0.05)
                                    })

                            logger.info(f"Firecrawl Search 완료: {len(urls)}개 URL 발견")

                            # URL 목록 로깅
                            for i, url_info in enumerate(urls[:5]):  # 처음 5개만 로그
                                logger.info(f"  [{i+1}] {url_info['title'][:50]} - {url_info['url'][:80]}")

                            return urls
                        else:
                            logger.error(f"Firecrawl Search 실패: {result.get('error', 'Unknown error')}")
                            return []
                    else:
                        error_text = await response.text()
                        logger.error(f"Firecrawl API HTTP {response.status}: {error_text}")
                        return []

        except Exception as e:
            logger.error(f"Firecrawl Search 실패: {e}")
            return []

    async def analyze_query_and_find_urls(self, nl_query: str, max_urls: int = 10, categories: List[str] = None) -> List[Dict]:
        """
        자연어 쿼리로 URL 검색 (Perplexity 또는 Firecrawl)

        Args:
            nl_query: 검색 쿼리
            max_urls: 최대 URL 개수
            categories: Firecrawl 카테고리 (["github", "research"] 등). None이면 Perplexity 사용

        Returns:
            URL 정보 리스트
        """
        try:
            # 카테고리가 지정되면 Firecrawl Search, 아니면 Perplexity Search
            if categories:
                verified_urls = await self.search_with_firecrawl(nl_query, max_urls, categories)
                search_type = "Firecrawl"
            else:
                verified_urls = await self.search(nl_query, max_urls)
                search_type = "Perplexity"

            if not verified_urls:
                logger.warning(f"{search_type} 검색 결과 없음")
                return []

            logger.info(f"{search_type} 검색 완료: {len(verified_urls)}개 URL")
            return verified_urls

        except Exception as e:
            logger.error(f"URL 검색 실패: {e}")
            return []

    # ============================================
    # Firecrawl 문서 다운로드
    # ============================================

    def check_duplicate_files(
        self,
        urls: List[Dict],
        output_folder: str,
        folder_name: str = None
    ) -> List[Dict]:
        """
        중복 파일 사전 체크

        Returns:
            List[Dict]: 중복 파일 정보 [{"url": ..., "title": ..., "filename": ...}]
        """
        try:
            if folder_name:
                save_path = Path(output_folder) / self._sanitize_folder_name(folder_name)
            else:
                save_path = Path(output_folder)

            duplicates = []
            for url_info in urls:
                url = url_info.get("url", "")
                if not url:
                    continue

                # 파일명 생성 로직 (실제 다운로드와 동일)
                title = url_info.get("title", "Document")
                url_hash = hashlib.md5(url.encode()).hexdigest()
                filename = f"{url_hash}.md"
                file_path = save_path / filename

                if file_path.exists():
                    duplicates.append({
                        "url": url,
                        "title": title,
                        "filename": filename,
                        "path": str(file_path)
                    })

            return duplicates

        except Exception as e:
            logger.error(f"중복 체크 실패: {e}")
            return []

    async def scrape_urls_to_markdown(
        self,
        urls: List[Dict],
        output_folder: str,
        folder_name: str = None,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tqdm: bool = True
    ) -> List[str]:
        """
        URL 리스트를 Firecrawl scrape로 다운로드하여 마크다운 파일로 저장

        Args:
            urls: URL 정보 리스트 (analyze_query_and_find_urls 결과)
            output_folder: 기본 출력 폴더 경로
            folder_name: 하위 폴더명 (예: "Memory Forensic Tools")
            overwrite: 기존 파일 덮어쓰기 여부
            progress_callback: 진행상황 콜백 함수 (current, total)
            use_tqdm: CLI에서 tqdm 진행바 사용 여부

        Returns:
            저장된 파일 경로 리스트
        """
        try:
            # 폴더 생성
            if folder_name:
                save_path = Path(output_folder) / self._sanitize_folder_name(folder_name)
            else:
                save_path = Path(output_folder)

            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"문서 저장 경로: {save_path}")

            saved_files = []
            total_urls = len(urls)

            # tqdm 진행바 (CLI 모드)
            url_iterator = tqdm(enumerate(urls), total=total_urls, desc="📥 문서 다운로드", unit="파일") if use_tqdm else enumerate(urls)

            for i, url_info in url_iterator:
                url = url_info.get("url", "")
                if not url:
                    continue

                try:
                    # Firecrawl scrape 실행
                    scraped_data = await self._scrape_with_firecrawl(url)

                    if scraped_data and scraped_data.get("markdown"):
                        # Firecrawl의 title을 우선 사용, 없으면 url_info의 title 사용
                        firecrawl_title = scraped_data.get("title", "")
                        title = firecrawl_title or url_info.get("title", f"Document_{i+1}")

                        # URL 기반 MD5 해시 생성 (32자리)
                        url_hash = hashlib.md5(url.encode()).hexdigest()

                        # 파일명 생성: {MD5Hash}.md
                        filename = f"{url_hash}.md"
                        file_path = save_path / filename

                        # 중복 체크: 파일이 이미 존재하면 스킵 (overwrite=False인 경우)
                        if file_path.exists() and not overwrite:
                            logger.info(f"⏭️ 스킵 (이미 존재): {filename}")
                            saved_files.append(str(file_path))
                            continue
                        elif file_path.exists() and overwrite:
                            logger.info(f"🔄 덮어쓰기: {filename}")

                        # 마크다운 저장
                        markdown_content = scraped_data["markdown"]

                        # 내용이 너무 짧으면 경고
                        if len(markdown_content.strip()) < 100:
                            logger.warning(f"⚠️ 스크래핑 내용이 매우 짧음 ({len(markdown_content)} chars): {url}")
                            logger.warning(f"   제목: {title}")

                        metadata_header = f"""---
url: {url}
title: {title}
scraped_at: {time.strftime('%Y-%m-%d %H:%M:%S')}
---

"""
                        full_content = metadata_header + markdown_content

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(full_content)

                        saved_files.append(str(file_path))
                        logger.info(f"저장 완료: {filename} ({len(markdown_content)} chars)")
                    else:
                        logger.error(f"❌ 스크래핑 실패 (마크다운 없음): {url}")
                        logger.error(f"   제목: {url_info.get('title', 'Unknown')}")

                except Exception as e:
                    logger.error(f"URL 처리 실패 {url}: {e}")
                    continue

                # 진행상황 콜백 호출 (Streamlit용)
                if progress_callback:
                    progress_callback(i + 1, total_urls)

            logger.info(f"총 {len(saved_files)}개 문서 저장 완료")
            return saved_files

        except Exception as e:
            logger.error(f"문서 다운로드 실패: {e}")
            return []

    async def _scrape_with_firecrawl(self, url: str) -> Optional[Dict]:
        """Firecrawl API로 URL 스크래핑"""
        try:
            api_url = "https://api.firecrawl.dev/v1/scrape"

            headers = {
                "Authorization": f"Bearer {self.firecrawl_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
                "waitFor": 2000,
                "timeout": 30000
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()

                        if result.get("success"):
                            data = result.get("data", {})
                            markdown = data.get("markdown", "")

                            # 디버그 로그
                            logger.debug(f"Firecrawl 응답: markdown 길이={len(markdown)}, url={url}")

                            return {
                                "markdown": markdown,
                                "title": data.get("metadata", {}).get("title", ""),
                                "html": data.get("html", ""),
                                "metadata": data.get("metadata", {})
                            }
                        else:
                            error_msg = result.get("error", "알 수 없는 오류")
                            logger.error(f"Firecrawl scrape 실패 ({url}): {error_msg}")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Firecrawl API HTTP {response.status} ({url}): {error_text[:200]}")
                        return None

        except Exception as e:
            logger.error(f"Firecrawl scrape 오류 {url}: {e}")
            return None

    # ============================================
    # 통합 파이프라인 (새로운 기능)
    # ============================================

    async def search_and_download(
        self,
        query: str,
        max_urls: int = 10,
        output_folder: str = "./downloads",
        folder_name: str = None,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        use_tqdm: bool = True
    ) -> Dict[str, any]:
        """
        URL 수집부터 문서 다운로드까지 한 번에 실행하는 통합 파이프라인

        Args:
            query: 검색 쿼리
            max_urls: 최대 URL 개수
            output_folder: 저장 폴더
            folder_name: 하위 폴더명
            overwrite: 기존 파일 덮어쓰기
            progress_callback: 진행상황 콜백
            use_tqdm: CLI 진행바 사용 여부

        Returns:
            {
                "urls": [...],      # 수집된 URL 리스트
                "files": [...],     # 저장된 파일 경로 리스트
                "success": bool     # 성공 여부
            }
        """
        try:
            logger.info(f"통합 파이프라인 시작: {query}")

            # Step 1: URL 수집
            urls = await self.search(query, max_urls)
            if not urls:
                logger.warning("URL 수집 실패 또는 결과 없음")
                return {
                    "urls": [],
                    "files": [],
                    "success": False
                }

            logger.info(f"Step 1/2 완료: {len(urls)}개 URL 수집")

            # Step 2: 문서 다운로드
            files = await self.scrape_urls_to_markdown(
                urls=urls,
                output_folder=output_folder,
                folder_name=folder_name,
                overwrite=overwrite,
                progress_callback=progress_callback,
                use_tqdm=use_tqdm
            )

            logger.info(f"Step 2/2 완료: {len(files)}개 문서 다운로드")

            return {
                "urls": urls,
                "files": files,
                "success": len(files) > 0
            }

        except Exception as e:
            logger.error(f"통합 파이프라인 실패: {e}")
            return {
                "urls": [],
                "files": [],
                "success": False,
                "error": str(e)
            }

    # ============================================
    # 유틸리티 메서드
    # ============================================

    def _sanitize_folder_name(self, name: str) -> str:
        """폴더명을 안전한 형식으로 변환"""
        # 특수문자 제거 및 공백을 언더바로 변환
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = name.replace(' ', '_')
        return name[:100]  # 길이 제한

    def _sanitize_filename(self, name: str) -> str:
        """파일명을 안전한 형식으로 변환"""
        # 특수문자 제거 및 공백을 언더바로 변환
        name = re.sub(r'[<>:"/\\|?*]', '', name)
        name = name.replace(' ', '_')
        # 연속 언더바 제거
        name = re.sub(r'_+', '_', name)
        return name[:100]  # 길이 제한

# 기존 코드와의 호환성을 위해 클래스 별칭 제공
FirecrawlDocumentCrawler = WebDocumentCrawler
