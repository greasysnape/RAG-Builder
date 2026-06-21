import streamlit as st
import asyncio
import os
import time
import platform
import psutil
import glob
import datetime
import pandas as pd
import plotly.express as px
from rag_builder.main import RAGBuilder
from rag_builder.config.settings import settings
from rag_builder.embedder.embedding_factory import EmbeddingFactory
from rag_builder.embedder.vector_store_factory import VectorStoreFactory
from rag_builder.utils.file_manager import FileManager
from rag_builder.embedder.vector_store import VectorStore
from rag_builder.locales.translator import Translator
from rag_builder.utils.preferences import UserPreferences

# 파일 업로드 및 경로 관리 기능
def handle_uploaded_files(uploaded_files, session_key):
    """업로드된 파일들을 임시 폴더에 저장하고 경로 반환"""
    if not uploaded_files:
        return []

    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="rag_builder_")
    file_paths = []

    for uploaded_file in uploaded_files:
        # 임시 파일로 저장
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_file_path)

    # 세션 상태에 저장
    st.session_state[session_key] = file_paths
    return file_paths

st.set_page_config(
    page_title="RAG Builder",
    page_icon="🔥",
    layout="wide"
)

# 사용자 설정 관리자 초기화
if 'preferences' not in st.session_state:
    st.session_state.preferences = UserPreferences()

# 저장된 설정을 session_state에 캐시 (한 번만 로드)
if 'saved_prefs' not in st.session_state:
    st.session_state.saved_prefs = st.session_state.preferences.load()

# 언어 설정 초기화 (저장된 설정 우선 사용)
if 'language' not in st.session_state:
    st.session_state.language = st.session_state.saved_prefs.get('language', 'en')
if 'translator' not in st.session_state:
    st.session_state.translator = Translator(st.session_state.language)

# 번역 함수 단축어
t = st.session_state.translator.t

# 사이드바: 설정
with st.sidebar:
    st.header(t('ui.settings'))

    # 언어 선택
    st.markdown(f"### {t('ui.language')}")

    lang_options = Translator.get_supported_languages()
    selected_lang = st.selectbox(
        "",  # 레이블 없음 (헤더에 이미 표시)
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=0 if st.session_state.language == 'ko' else 1,
        key="lang_selector",
        label_visibility="collapsed"
    )

    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.session_state.translator.change_language(selected_lang)
        # 언어 설정 자동 저장
        st.session_state.preferences.update(language=selected_lang)
        # session_state의 캐시도 업데이트
        st.session_state.saved_prefs['language'] = selected_lang
        st.rerun()

    st.markdown("---")

    st.markdown(f"### {t('settings.embedding_model')}")

    embedding_type_options = {
        "sentence_transformer": "Sentence Transformers",
        "openai": "OpenAI Embeddings API"
    }

    # 저장된 설정 우선 사용 (session_state 캐시 사용)
    current_embedding_type = st.session_state.saved_prefs.get('embedding_type', getattr(settings, 'embedding_type', 'sentence_transformer'))

    selected_embedding_type = st.selectbox(
        t('settings.embedding_type'),
        options=list(embedding_type_options.keys()),
        format_func=lambda x: embedding_type_options[x],
        index=0 if current_embedding_type == "sentence_transformer" else 1,
        key="embedding_type_selector"
    )

    available_models = EmbeddingFactory.get_available_models(selected_embedding_type)

    if available_models:
        model_options = list(available_models.keys())
        # 저장된 모델 우선 사용 (session_state 캐시 사용)
        default_model = st.session_state.saved_prefs.get('embedding_model', settings.local_embedding_model)
        if default_model in model_options:
            default_index = model_options.index(default_model)
        else:
            default_index = 0

        selected_model = st.selectbox(
            t('settings.model'),
            options=model_options,
            format_func=lambda x: f"{x} ({available_models[x]['dimension']}d)",
            index=default_index,
            key="model_selector"
        )

        model_info = available_models[selected_model]
        st.caption(t('settings.dimension', dimension=model_info['dimension']))

    else:
        selected_model = settings.local_embedding_model
        st.warning(t('settings.package_not_installed', db='Model'))

    if selected_embedding_type == "openai":
        st.markdown("---")
        current_openai_key = getattr(settings, 'openai_api_key', '')
        openai_key_input = st.text_input(
            t('settings.openai_api_key'),
            value=current_openai_key if current_openai_key else "",
            type="password",
            help=t('settings.openai_api_key_help')
        )
        if openai_key_input:
            os.environ["OPENAI_API_KEY"] = openai_key_input

    st.markdown(f"### {t('settings.vector_db')}")

    vector_store_options = {
        "chromadb": "ChromaDB",
        "faiss": "FAISS"
    }

    # 저장된 설정 우선 사용 (session_state 캐시 사용)
    current_vector_store_type = st.session_state.saved_prefs.get('vector_db_type', getattr(settings, 'vector_db_type', 'chromadb'))

    selected_vector_store = st.selectbox(
        t('settings.vector_db_type'),
        options=list(vector_store_options.keys()),
        format_func=lambda x: vector_store_options[x],
        index=0 if current_vector_store_type == "chromadb" else 1,
        key="vector_store_selector"
    )

    is_available = VectorStoreFactory.is_type_available(selected_vector_store)
    if not is_available:
        st.warning(t('settings.package_not_installed', db=selected_vector_store.upper()))
        if selected_vector_store == "faiss":
            st.caption(t('settings.install_command', package='faiss-cpu'))

    st.markdown("---")

    if st.button(t('common.apply'), type="primary"):
        st.session_state.embedding_type = selected_embedding_type
        st.session_state.model_name = selected_model
        st.session_state.vector_store_type = selected_vector_store

        # 설정 자동 저장
        st.session_state.preferences.update(
            embedding_type=selected_embedding_type,
            embedding_model=selected_model,
            vector_db_type=selected_vector_store
        )

        # session_state의 캐시도 업데이트
        st.session_state.saved_prefs['embedding_type'] = selected_embedding_type
        st.session_state.saved_prefs['embedding_model'] = selected_model
        st.session_state.saved_prefs['vector_db_type'] = selected_vector_store

        st.session_state.rag_builder = RAGBuilder(
            use_tqdm=False,
            embedding_type=selected_embedding_type,
            model_name=selected_model,
            vector_store_type=selected_vector_store
        )
        st.success(t('settings.settings_applied'))
        st.caption(t('settings.embedding_info', type=embedding_type_options[selected_embedding_type], model=selected_model))
        st.caption(t('settings.vector_db_info', db=vector_store_options[selected_vector_store]))
        st.rerun()

# RAG Builder 초기화 (Streamlit 모드: use_tqdm=False)
if 'rag_builder' not in st.session_state:
    # 저장된 설정 불러오기 (session_state 캐시 사용)
    default_embedding_type = st.session_state.saved_prefs.get('embedding_type', getattr(settings, 'embedding_type', 'sentence_transformer'))
    default_embedding_model = st.session_state.saved_prefs.get('embedding_model', settings.local_embedding_model)
    default_vector_store_type = st.session_state.saved_prefs.get('vector_db_type', getattr(settings, 'vector_db_type', 'chromadb'))

    embedding_type = getattr(st.session_state, 'embedding_type', default_embedding_type)
    model_name = getattr(st.session_state, 'model_name', default_embedding_model)
    vector_store_type = getattr(st.session_state, 'vector_store_type', default_vector_store_type)

    st.session_state.rag_builder = RAGBuilder(
        use_tqdm=False,
        embedding_type=embedding_type,
        model_name=model_name,
        vector_store_type=vector_store_type
    )

# 헤더
col1, = st.columns([1])
with col1:
    st.title(t('ui.title'))
    st.caption(t('ui.subtitle'))

# 탭 구성
auto_tab, task1_tab, task2_tab, task3_tab, file_manager_tab, info_tab = st.tabs([
    t('tabs.auto_workflow'),
    t('tabs.task1'),
    t('tabs.task2'),
    t('tabs.task3'),
    t('tabs.file_manager'),
    t('tabs.info')
])

# 전체 워크플로우 자동화
with auto_tab:
    st.subheader(t('auto_workflow.title'))
    st.markdown(t('auto_workflow.description'))

    with st.form("auto_workflow_form"):
        col_auto1, col_auto2 = st.columns(2)

        with col_auto1:
            auto_query = st.text_area(
                t('auto_workflow.search_query'),
                placeholder="official documents of memory forensic tools",
                height=120
            )

            auto_max_urls = st.number_input(
                t('auto_workflow.max_urls'),
                min_value=1,
                max_value=30,
                value=5,
                step=1
            )

            # 카테고리 선택 (Firecrawl Search)
            auto_category_options = st.multiselect(
                t('task1.search_categories'),
                options=["github", "research"],
                default=[],
                help="Select categories to use Firecrawl Search. Leave empty to use Perplexity.",
                key="auto_category_select"
            )

        with col_auto2:
            auto_output_folder = st.text_input(
                t('auto_workflow.output_folder'),
                value="./downloads"
            )

            # 컬렉션 선택 (드롭다운)
            collection_options = [t('auto_workflow.new_collection')] + settings.predefined_collections
            auto_collection_select = st.selectbox(
                t('auto_workflow.collection'),
                collection_options,
                index=0
            )

            # "새 컬렉션 만들기" 선택 시에만 입력 필드 표시
            if auto_collection_select == t('auto_workflow.new_collection'):
                auto_collection_name = st.text_input(
                    t('auto_workflow.collection_name_input'),
                    placeholder="digital_forensic_tools",
                    label_visibility="collapsed"
                )
            else:
                auto_collection_name = auto_collection_select

        # 덮어쓰기 옵션
        auto_overwrite = st.checkbox(
            t('auto_workflow.overwrite_files'),
            value=False,
            help="체크하면 이미 다운로드된 파일도 다시 다운로드합니다"
        )

        auto_run_btn = st.form_submit_button(t('auto_workflow.run_workflow'), type="primary")

        if auto_run_btn and auto_query:
            st.markdown("---")
            st.info(t('auto_workflow.starting'))

            # 진행 단계 표시
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                start_time = time.time()

                # 카테고리 설정
                auto_search_categories = auto_category_options if auto_category_options else None

                # Task 1: URL 수집
                with progress_placeholder:
                    if auto_search_categories:
                        st.info(t('auto_workflow.step1_firecrawl'))
                    else:
                        st.info(t('auto_workflow.step1_perplexity'))

                urls = asyncio.run(
                    st.session_state.rag_builder.find_urls_for_query(auto_query, auto_max_urls, auto_search_categories)
                )

                if not urls:
                    st.error(t('auto_workflow.search_failed'))
                else:
                    # Task 2: 문서 다운로드
                    with progress_placeholder:
                        st.info(t('auto_workflow.step2', count=len(urls)))

                    # 진행바 생성
                    download_progress_bar = st.progress(0)
                    download_status = st.empty()

                    def download_callback(current, total):
                        download_progress_bar.progress(current / total)
                        download_status.text(t('auto_workflow.downloading', current=current, total=total))

                    folder_name = auto_collection_name or auto_query.replace(' ', '_')[:50]
                    file_paths = asyncio.run(
                        st.session_state.rag_builder.download_documents_from_urls(
                            urls, auto_output_folder, folder_name, auto_overwrite,
                            progress_callback=download_callback
                        )
                    )

                    download_progress_bar.empty()
                    download_status.empty()

                    if not file_paths:
                        st.error(t('auto_workflow.download_failed'))
                    else:
                        # Task 3: RAG 구축
                        with progress_placeholder:
                            st.info(t('auto_workflow.step3', count=len(file_paths)))

                        # 진행바 생성
                        embedding_progress_bar = st.progress(0)
                        embedding_status = st.empty()

                        def embedding_callback(current, total):
                            embedding_progress_bar.progress(current / total)
                            embedding_status.text(t('auto_workflow.embedding', current=current, total=total))

                        collection_name = auto_collection_name or folder_name
                        success = asyncio.run(
                            st.session_state.rag_builder.build_from_documents(
                                file_paths, collection_name,
                                progress_callback=embedding_callback
                            )
                        )

                        embedding_progress_bar.empty()
                        embedding_status.empty()

                        elapsed_time = time.time() - start_time

                        if success:
                            progress_placeholder.empty()
                            st.success(t('auto_workflow.success', time=elapsed_time))

                            # 결과 요약
                            st.markdown(f"### {t('auto_workflow.summary')}")
                            col_summary1, col_summary2, col_summary3 = st.columns(3)

                            with col_summary1:
                                st.caption(t('auto_workflow.search_status'))
                                st.write(t('auto_workflow.completed'))
                            with col_summary2:
                                st.caption(t('auto_workflow.download_status'))
                                st.write(t('auto_workflow.completed'))
                            with col_summary3:
                                st.caption(t('auto_workflow.rag_status'))
                                st.write(t('auto_workflow.completed'))

                            # 상세 정보
                            with st.expander(t('auto_workflow.details')):
                                st.markdown(f"""
                                - **{t('auto_workflow.query')}**: {auto_query}
                                - **{t('auto_workflow.save_path')}**: {auto_output_folder}
                                - **{t('auto_workflow.collection')}**: {auto_collection_name if auto_collection_name else t('auto_workflow.auto_generated')}
                                - **{t('auto_workflow.elapsed_time')}**: {elapsed_time:.1f}{t('auto_workflow.seconds')}
                                """)
                        else:
                            progress_placeholder.empty()
                            st.error(t('auto_workflow.rag_failed'))

            except Exception as e:
                progress_placeholder.empty()
                error_msg = str(e)
                st.error(t('auto_workflow.error', error=error_msg))

                with st.expander(t('auto_workflow.error_details')):
                    st.code(error_msg)

# Task 1: URL 수집 (Perplexity)
with task1_tab:
    st.subheader(t('task1.title'))

    # 태스크 설명
    with st.expander(t('task1.info_title'), expanded=False):
        st.markdown(t('task1.info_content'))

    with st.form("task1_form"):
        nl_query = st.text_area(
            t('task1.search_query'),
            placeholder="official documents of memory forensic tools, Volatility3 documentation",
            height=80,
            help=t('task1.search_help')
        )

        col_task1_1, col_task1_2 = st.columns(2)

        with col_task1_1:
            max_urls = st.number_input(t('task1.max_urls'), min_value=1, max_value=50, value=10, step=1)

        with col_task1_2:
            # 카테고리 선택 (Firecrawl Search)
            category_options = st.multiselect(
                t('task1.search_categories'),
                options=["github", "research"],
                default=[],
                help="Select categories to use Firecrawl Search. Leave empty to use Perplexity."
            )

        search_btn = st.form_submit_button(t('task1.search_button'), type="primary")

        if search_btn and nl_query:
            # 카테고리가 선택되면 Firecrawl, 아니면 Perplexity
            search_categories = category_options if category_options else None

            if search_categories:
                search_msg = t('task1.searching_firecrawl')
            else:
                search_msg = t('task1.searching_perplexity')

            with st.spinner(search_msg):
                try:
                    found_urls = asyncio.run(
                        st.session_state.rag_builder.find_urls_for_query(nl_query, max_urls, search_categories)
                    )

                    if found_urls:
                        st.success(t('task1.success', count=len(found_urls)))
                        st.session_state.task1_results = {
                            'query': nl_query,
                            'urls': found_urls,
                            'timestamp': time.time()
                        }

                        # URL 목록 표시
                        st.markdown(f"### {t('task1.url_list')}")
                        for i, url_info in enumerate(found_urls):
                            with st.expander(f"{i+1}. {url_info.get('title', 'No Title')[:60]}"):
                                st.markdown(f"**URL**: {url_info['url']}")
                    else:
                        st.warning(t('task1.no_results'))

                except Exception as e:
                    error_msg = str(e)
                    st.error(t('task1.search_failed', error=error_msg))

                    with st.expander(t('task1.error_details')):
                        st.code(error_msg)

    # 기존 결과 표시
    if hasattr(st.session_state, 'task1_results'):
        result = st.session_state.task1_results
        timestamp = datetime.datetime.fromtimestamp(result['timestamp'])

        st.markdown("---")
        st.info(t('task1.saved_results', count=len(result['urls']), time=timestamp.strftime('%Y-%m-%d %H:%M:%S')))

        col_t1_1, col_t1_2 = st.columns(2)
        with col_t1_1:
            st.caption(t('task1.query_caption'))
            st.write(result['query'])
        with col_t1_2:
            st.caption(t('task1.urls_caption'))
            st.write(t('task1.url_count', count=len(result['urls'])))

# Task 2: 문서 다운로드 (Firecrawl Scrape)
with task2_tab:
    st.subheader(t('task2.title'))

    # 태스크 설명
    with st.expander(t('task2.info_title'), expanded=False):
        st.markdown(t('task2.info_content'))

    # URL 소스 선택
    url_source = st.radio(t('task2.url_source'), [t('task2.use_task1'), t('task2.manual_input')], horizontal=True)

    urls_to_download = []

    if url_source == t('task2.use_task1'):
        if hasattr(st.session_state, 'task1_results'):
            urls_to_download = st.session_state.task1_results['urls']
            st.success(t('task2.urls_loaded', count=len(urls_to_download)))
        else:
            st.warning(t('task2.run_task1_first'))

    elif url_source == t('task2.manual_input'):
        manual_urls = st.text_area(
            t('task2.url_list'),
            height=100,
            placeholder="https://example.com/document1\nhttps://example.com/document2"
        )
        if manual_urls:
            url_list = [url.strip() for url in manual_urls.split('\n') if url.strip()]
            urls_to_download = [{"url": url, "title": url.split('/')[-1]} for url in url_list]
            st.success(t('task2.urls_entered', count=len(urls_to_download)))

    if urls_to_download:
        st.markdown("---")

        # URL 목록 미리보기
        with st.expander(t('task2.url_list_preview', count=len(urls_to_download))):
            for i, url_info in enumerate(urls_to_download):
                st.text(f"{i+1}. {url_info.get('title', url_info['url'])[:80]}")

        # 저장 설정
        col_save1, col_save2 = st.columns(2)
        with col_save1:
            output_folder = st.text_input(t('task2.output_folder'), value="./downloads")
        with col_save2:
            folder_name = st.text_input(t('task2.subfolder'), value="", help=t('task2.subfolder_help'))

        # 중복 파일 체크
        duplicates = st.session_state.rag_builder.document_crawler.check_duplicate_files(
            urls_to_download, output_folder, folder_name
        )

        # 중복 파일 경고 및 선택 옵션
        overwrite = False
        if duplicates:
            st.warning(t('task2.duplicate_warning', count=len(duplicates)))

            with st.expander(t('task2.duplicate_list', count=len(duplicates))):
                for dup in duplicates:
                    st.text(f"📁 {dup['filename']}")

            overwrite = st.radio(
                t('task2.duplicate_handling'),
                [t('task2.skip_duplicates'), t('task2.overwrite_duplicates')],
                help=t('task2.duplicate_help')
            ) == t('task2.overwrite_duplicates')

        if st.button(t('task2.scrape_button'), type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def task2_callback(current, total):
                progress_bar.progress(current / total)
                status_text.text(t('task2.downloading', current=current, total=total))

            try:
                # Firecrawl scrape로 마크다운 다운로드
                saved_files = asyncio.run(
                    st.session_state.rag_builder.download_documents_from_urls(
                        urls=urls_to_download,
                        output_folder=output_folder,
                        folder_name=folder_name,
                        overwrite=overwrite,
                        progress_callback=task2_callback
                    )
                )

                progress_bar.progress(1.0)

                if saved_files:
                    status_text.text(t('task2.completed', count=len(saved_files)))

                    # 결과 메트릭
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.caption(t('task2.saved_files'))
                        st.write(t('task2.file_count', count=len(saved_files)))
                    with col_metric2:
                        st.caption(t('task2.save_path'))
                        st.write(f"{output_folder}/{folder_name}")

                    # 파일 목록
                    with st.expander(t('task2.saved_file_list')):
                        for file_path in saved_files:
                            st.text(f"✓ {os.path.basename(file_path)}")

                    # Task 2 결과 저장
                    st.session_state.task2_results = {
                        'folder_path': os.path.join(output_folder, folder_name),
                        'file_paths': saved_files,
                        'timestamp': time.time()
                    }
                else:
                    status_text.text(t('task2.no_files'))

            except Exception as e:
                progress_bar.empty()
                error_msg = str(e)
                st.error(t('task2.download_failed', error=error_msg))

                with st.expander(t('task2.error_details')):
                    st.code(error_msg)

    # 기존 결과 표시
    if hasattr(st.session_state, 'task2_saved_results'):
        result = st.session_state.task2_results
        timestamp = datetime.datetime.fromtimestamp(result['timestamp'])

        st.markdown("---")
        st.success(t('task2.saved_results', count=len(result['file_paths']), time=timestamp.strftime('%Y-%m-%d %H:%M:%S')))

        col_t2_1, col_t2_2 = st.columns(2)
        with col_t2_1:
            st.caption(t('task2.files_caption'))
            st.write(t(count=len(result['file_paths'])))
        with col_t2_2:
            st.caption(t('task2.folder_path'))
            st.write(result['folder_path'])

# Task 3: RAG 구축
with task3_tab:
    st.subheader(t('task3.title'))

    # 태스크 설명
    current_vector_db = st.session_state.rag_builder.vector_store_type.upper()
    with st.expander(t('task3.info_title'), expanded=False):
        st.markdown(t('task3.info_content',
                      vector_db=current_vector_db,
                      chunk_size=settings.chunk_size,
                      chunk_overlap=settings.chunk_overlap))

    # 문서 소스 선택
    doc_source = st.radio(t('task3.doc_source'), [t('task3.use_task2'), t('task3.folder_input'), t('task3.file_upload')], horizontal=True)

    file_paths = []

    if doc_source == t('task3.use_task2'):
        if hasattr(st.session_state, 'task2_results'):
            file_paths = st.session_state.task2_results['file_paths']
            timestamp = datetime.datetime.fromtimestamp(st.session_state.task2_results['timestamp'])
            st.success(t('task3.loaded_from_task2', count=len(file_paths), time=timestamp.strftime('%H:%M:%S')))
        else:
            st.warning(t('task3.run_task2_first'))

    elif doc_source == t('task3.folder_input'):
        folder_path = st.text_input(
            t('task3.folder_path'),
            value="./downloads",
            help=t('task3.folder_help')
        )

        if folder_path and os.path.exists(folder_path):
            # 재귀적으로 모든 .md 파일 찾기
            file_paths = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)

            if file_paths:
                st.info(t('task3.files_found', count=len(file_paths)))

                with st.expander(t('task3.file_list')):
                    for fp in file_paths[:20]:  # 최대 20개만 표시
                        st.text(f"✓ {os.path.basename(fp)}")
                    if len(file_paths) > 20:
                        st.text(t('task3.and_more', count=len(file_paths) - 20))
            else:
                st.warning(t('task3.no_md_files'))
        elif folder_path:
            st.error(t('task3.folder_not_found', path=folder_path))

    elif doc_source == t('task3.file_upload'):
        uploaded_docs = st.file_uploader(
            t('task3.file_select'),
            type=['md', 'txt', 'pdf'],
            accept_multiple_files=True,
            key="task3_files_uploaded"
        )

        if uploaded_docs:
            file_paths = handle_uploaded_files(uploaded_docs, "task3_uploaded_file_list")
            st.success(t('task3.files_uploaded', count=len(uploaded_docs)))

            with st.expander(t('task3.uploaded_files_list')):
                for uploaded_file in uploaded_docs:
                    st.text(f"✓ {uploaded_file.name} ({uploaded_file.size} bytes)")

    if file_paths:
        st.markdown("---")

        # 컬렉션 설정
        collection_name = st.selectbox(
            t('task3.collection_select'),
            settings.predefined_collections + [t('task3.new_collection')]
        )

        if collection_name == t('task3.new_collection'):
            collection_name = st.text_input(t('task3.new_collection_name'), placeholder="my_collection")

        # 중복 임베딩 체크 (기존 컬렉션인 경우)
        update_mode = False
        if collection_name and collection_name != t('task3.new_collection'):
            from rag_builder.embedder.vector_store import VectorStore
            vector_store = VectorStore(collection_name)
            dup_check = vector_store.check_duplicate_embeddings(file_paths)

            if dup_check["duplicates"]:
                st.warning(t('task3.duplicate_warning', count=len(dup_check['duplicates'])))

                with st.expander(t('task3.duplicate_list', count=len(dup_check['duplicates']))):
                    for dup in dup_check["duplicates"]:
                        st.text(f"📄 {dup['filename']}")

                update_mode = st.radio(
                    t('task3.duplicate_handling'),
                    [t('task3.skip_option'), t('task3.update_option')],
                    help=t('task3.duplicate_help')
                ) == t('task3.update_option')

                if not update_mode:
                    # 중복 제외한 파일 목록으로 업데이트
                    file_paths = dup_check["new_files"]
                    st.info(t('task3.new_docs_only', count=len(file_paths)))

        # 미리보기 정보
        col_preview1, col_preview2 = st.columns(2)
        with col_preview1:
            st.caption(t('task3.document_count'))
            st.write(t('task3.count_display', count=len(file_paths)))
        with col_preview2:
            st.caption(t('task3.target_collection'))
            st.write(collection_name if collection_name else t('task3.not_selected'))

        if collection_name and len(file_paths) > 0 and st.button(t('task3.build_button'), type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def task3_callback(current, total):
                progress_bar.progress(current / total)
                status_text.text(t('task3.embedding', current=current, total=total))

            try:
                # RAG 데이터베이스 구축
                success = asyncio.run(
                    st.session_state.rag_builder.build_from_documents(
                        file_paths, collection_name,
                        progress_callback=task3_callback
                    )
                )

                progress_bar.progress(1.0)
                status_text.empty()

                if success:
                    st.success(t('task3.build_completed'))

                    # 결과 요약
                    col_result1, col_result2, col_result3 = st.columns(3)
                    with col_result1:
                        st.caption(t('task3.collection_name'))
                        st.write(collection_name)
                    with col_result2:
                        st.caption(t('task3.document_count_label'))
                        st.write(t('task3.count_display', count=len(file_paths)))
                    with col_result3:
                        st.caption(t('task3.status'))
                        st.write(t('task3.status_completed'))

                    # 벡터 DB 통계
                    try:
                        from rag_builder.embedder.vector_store import VectorStore
                        vector_store = VectorStore(collection_name)
                        stats = vector_store.get_collection_stats()

                        if stats:
                            st.info(t('task3.total_chunks', count=stats.get('total_documents', 0)))

                            # Task 3 결과 저장
                            st.session_state.task3_results = {
                                'collection_name': collection_name,
                                'document_count': len(file_paths),
                                'total_embeddings': stats.get('total_documents', 0),
                                'timestamp': time.time()
                            }
                    except Exception as e:
                        st.warning(t('task3.stats_failed', error=e))
                else:
                    st.error(t('task3.build_failed'))

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                error_msg = str(e)
                st.error(t('task3.build_failed', error=error_msg))

                with st.expander(t('task3.build_error')):
                    st.code(error_msg)

    # 기존 결과 표시
    if hasattr(st.session_state, 'task3_results'):
        result = st.session_state.task3_results
        timestamp = datetime.datetime.fromtimestamp(result['timestamp'])

        st.markdown("---")
        st.success(t('task3.recent_build', name=result['collection_name'], time=timestamp.strftime('%Y-%m-%d %H:%M:%S')))

        col_last1, col_last2, col_last3 = st.columns(3)
        with col_last1:
            st.caption(t('task3.document_count'))
            st.write(t('task3.count_display', count=result['document_count']))
        with col_last2:
            st.caption(t('task3.embedding_chunks'))
            st.write(t('task3.count_display', count=result['total_embeddings']))
        with col_last3:
            st.caption(t('task3.status'))
            st.write(t('task3.status_completed'))

# 파일 관리자 탭
with file_manager_tab:
    st.subheader(t('file_manager.title'))

    file_manager = FileManager()

    # 하위 탭: 문서 탐색기, 임베딩 현황, 중복 관리, 통계
    fm_tab1, fm_tab2, fm_tab3, fm_tab4 = st.tabs([
        t('file_manager.tabs.explorer'),
        t('file_manager.tabs.embeddings'),
        t('file_manager.tabs.duplicates'),
        t('file_manager.tabs.stats')
    ])

    # 문서 탐색기
    with fm_tab1:
        st.markdown(f"### {t('file_manager.explorer.title')}")

        # 폴더 목록
        folders = file_manager.get_all_folders()

        col_fm1, col_fm2 = st.columns([1, 2])

        with col_fm1:
            st.markdown(f"**{t('file_manager.explorer.folder_list')}**")

            if folders:
                # 전체 보기 옵션
                selected_folder = st.radio(
                    t('file_manager.explorer.folder_select'),
                    [t('file_manager.explorer.view_all')] + [f"📁 {f['name']}" for f in folders],
                    key="folder_selector"
                )

                # 폴더 정보 표시
                if selected_folder != t('file_manager.explorer.view_all'):
                    folder_name = selected_folder.replace("📁 ", "")
                    folder_info = next((f for f in folders if f["name"] == folder_name), None)

                    if folder_info:
                        st.caption(t('file_manager.explorer.folder_info'))
                        st.text(t('file_manager.explorer.file_count', count=folder_info['file_count']))
                        st.text(t('file_manager.explorer.folder_size', size=folder_info['total_size'] / (1024*1024)))
                        st.text(t('file_manager.explorer.modified', time=folder_info['modified_time'].strftime('%Y-%m-%d %H:%M')))

                        st.markdown("---")

                        # 폴더 단위 임베딩
                        st.markdown(f"**{t('file_manager.explorer.folder_embedding')}**")
                        folder_collection = st.text_input(
                            t('file_manager.explorer.collection_name'),
                            value=folder_name,
                            key="folder_embed_collection"
                        )

                        if folder_collection:
                            if st.button(t('file_manager.explorer.embed_folder', name=folder_name), key="embed_folder_btn"):
                                folder_files = file_manager.get_files_in_folder(folder_name)
                                file_paths = [f['path'] for f in folder_files]

                                if file_paths:
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    def folder_callback(current, total):
                                        progress_bar.progress(current / total)
                                        status_text.text(t('file_manager.explorer.embedding', current=current, total=total))

                                    try:
                                        success = asyncio.run(
                                            st.session_state.rag_builder.build_from_documents(
                                                file_paths,
                                                folder_collection,
                                                progress_callback=folder_callback
                                            )
                                        )

                                        progress_bar.progress(1.0)
                                        status_text.empty()

                                        if success:
                                            st.success(t('file_manager.explorer.embed_completed'))
                                        else:
                                            st.error(t('file_manager.explorer.embed_failed'))

                                    except Exception as e:
                                        progress_bar.empty()
                                        status_text.empty()
                                        st.error(t('file_manager.explorer.embed_error', error=e))
                                else:
                                    st.warning(t('file_manager.explorer.no_files'))

                        st.markdown("---")

                        # 폴더 삭제 버튼
                        if st.button(t('file_manager.explorer.delete_folder', name=folder_name), key="delete_folder_btn"):
                            if file_manager.delete_folder(folder_info['path']):
                                st.success(t('file_manager.explorer.delete_completed'))
                                st.rerun()
                            else:
                                st.error(t('file_manager.explorer.delete_failed'))
            else:
                st.info(t('file_manager.explorer.no_folders'))

        with col_fm2:
            st.markdown(f"**{t('file_manager.explorer.file_list')}**")

            # 파일 목록 가져오기
            if folders and selected_folder != t('file_manager.explorer.view_all'):
                folder_name = selected_folder.replace("📁 ", "")
                files = file_manager.get_files_in_folder(folder_name)
            else:
                files = file_manager.get_all_files_recursive()

            if files:
                st.caption(t('file_manager.explorer.total_files', count=len(files)))

                # 검색 및 필터 행
                col_search1, col_search2 = st.columns([2, 1])

                with col_search1:
                    # 검색 기능
                    search_query = st.text_input(t('common.search'), placeholder=t('file_manager.explorer.search_placeholder'), key="file_search")

                with col_search2:
                    # 태그 필터
                    all_tags = file_manager.get_all_tags()
                    if all_tags:
                        tag_options = [t('file_manager.explorer.all_tags')] + list(all_tags.keys())
                        selected_tag_filter = st.selectbox(t('file_manager.explorer.tag_filter'), tag_options, key="tag_filter")
                    else:
                        selected_tag_filter = t('file_manager.explorer.all_tags')

                # 검색어 필터링
                if search_query:
                    files = [f for f in files if
                            search_query.lower() in f["filename"].lower() or
                            search_query.lower() in f.get("title", "").lower() or
                            search_query.lower() in f.get("url", "").lower()]
                    st.caption(t('file_manager.explorer.search_results', count=len(files)))

                # 태그 필터링
                if selected_tag_filter != t('file_manager.explorer.all_tags'):
                    tagged_file_paths = set(file_manager.get_files_by_tag(selected_tag_filter))
                    files = [f for f in files if f["path"] in tagged_file_paths]
                    st.caption(t('file_manager.explorer.tag_results', tag=selected_tag_filter, count=len(files)))

                # 즐겨찾기 필터
                show_favorites_only = st.checkbox(t('file_manager.explorer.favorites_only'), key="show_favorites")
                if show_favorites_only:
                    favorite_paths = set(file_manager.get_favorite_files())
                    files = [f for f in files if f["path"] in favorite_paths]
                    st.caption(t('file_manager.explorer.favorites_count', count=len(files)))

                # 파일 정렬
                sort_option = st.selectbox(t('file_manager.explorer.sort'), [t('file_manager.explorer.sort_recent'), t('file_manager.explorer.sort_name'), t('file_manager.explorer.sort_size')], key="file_sort")

                if sort_option == t('file_manager.explorer.sort_recent'):
                    files = sorted(files, key=lambda x: x["modified_time"], reverse=True)
                elif sort_option == t('file_manager.explorer.sort_name'):
                    files = sorted(files, key=lambda x: x["filename"])
                elif sort_option == t('file_manager.explorer.sort_size'):
                    files = sorted(files, key=lambda x: x["size"], reverse=True)

                # 파일 목록 표시
                for idx, file_info in enumerate(files[:50]):  # 최대 50개만 표시
                    file_path = file_info['path']

                    # 즐겨찾기 아이콘
                    is_fav = file_manager.is_favorite(file_path)
                    fav_icon = "⭐" if is_fav else ""

                    # 태그 표시
                    file_tags = file_manager.get_file_tags(file_path)
                    tags_str = " ".join([f"🏷️{tag}" for tag in file_tags[:3]])  # 최대 3개만
                    if len(file_tags) > 3:
                        tags_str += f" +{len(file_tags)-3}"

                    with st.expander(f"{idx+1}. {fav_icon} {file_info['filename'][:50]} {tags_str}"):
                        col_file1, col_file2, col_file3 = st.columns([2, 1, 1])

                        with col_file1:
                            st.markdown(f"**제목**: {file_info.get('title', 'N/A')[:80]}")
                            st.markdown(f"**URL**: {file_info.get('url', 'N/A')[:80]}")
                            st.text(f"크기: {file_info['size_mb']:.2f} MB")
                            st.text(f"수정: {file_info['modified_time'].strftime('%Y-%m-%d %H:%M')}")

                            # 메모 표시
                            file_note = file_manager.get_file_note(file_path)
                            if file_note:
                                st.markdown(f"**메모**: {file_note[:100]}")

                        with col_file2:
                            # 미리보기 버튼
                            if st.button("👁️ 미리보기", key=f"preview_{idx}"):
                                st.session_state[f"preview_content_{idx}"] = file_manager.read_file_content(file_path)

                            # 즐겨찾기 버튼
                            fav_label = "⭐ 즐겨찾기 해제" if is_fav else "⭐ 즐겨찾기"
                            if st.button(fav_label, key=f"fav_{idx}"):
                                file_manager.toggle_favorite(file_path)
                                st.rerun()

                            # 삭제 버튼
                            if st.button("🗑️ 삭제", key=f"delete_{idx}"):
                                if file_manager.delete_file(file_path):
                                    st.success("파일 삭제 완료!")
                                    st.rerun()
                                else:
                                    st.error("파일 삭제 실패")

                        with col_file3:
                            # 태그 관리
                            st.markdown("**🏷️ 태그 관리**")

                            # 태그 추가
                            new_tag = st.text_input("태그 추가:", key=f"new_tag_{idx}", placeholder="#forensics")

                            if new_tag and st.button("➕ 추가", key=f"add_tag_{idx}"):
                                # # 제거
                                clean_tag = new_tag.strip().lstrip('#')
                                if clean_tag:
                                    if file_manager.add_tag_to_file(file_path, clean_tag):
                                        st.success(f"태그 추가: {clean_tag}")
                                        st.rerun()

                            # 기존 태그 표시 및 제거
                            if file_tags:
                                st.caption("현재 태그:")
                                for tag in file_tags:
                                    col_tag1, col_tag2 = st.columns([3, 1])
                                    with col_tag1:
                                        st.text(f"#{tag}")
                                    with col_tag2:
                                        if st.button("❌", key=f"remove_tag_{idx}_{tag}"):
                                            file_manager.remove_tag_from_file(file_path, tag)
                                            st.rerun()

                        # 메모 추가/수정
                        st.markdown("---")
                        st.markdown("**📝 메모**")
                        note_input = st.text_area(
                            "메모:",
                            value=file_note,
                            key=f"note_{idx}",
                            height=80,
                            placeholder="파일에 대한 메모를 추가하세요..."
                        )

                        if st.button("💾 메모 저장", key=f"save_note_{idx}"):
                            if file_manager.set_file_note(file_path, note_input):
                                st.success("메모 저장 완료!")
                            else:
                                st.error("메모 저장 실패")

                        # 미리보기 내용 표시
                        if f"preview_content_{idx}" in st.session_state:
                            st.markdown("---")
                            st.markdown("**미리보기:**")
                            st.code(st.session_state[f"preview_content_{idx}"], language="markdown")

                if len(files) > 50:
                    st.info(f"ℹ️ {len(files) - 50}개 파일이 더 있습니다. 검색을 사용하세요.")
            else:
                st.info("파일이 없습니다")

    # 임베딩 현황
    with fm_tab2:
        st.markdown(f"### {t('file_manager.embeddings.title')}")

        # 컬렉션 목록
        collections = VectorStore.get_all_collections_info()

        if collections:
            st.caption(t('file_manager.embeddings.total_collections', count=len(collections)))

            # 컬렉션 대시보드
            for col_info in collections:
                with st.expander(t('file_manager.embeddings.collection', name=col_info['name'], count=col_info['document_count']), expanded=False):
                    col_emb1, col_emb2, col_emb3 = st.columns(3)

                    with col_emb1:
                        st.write(t('file_manager.embeddings.chunk_count'), col_info['document_count'])

                    with col_emb2:
                        st.write(t('file_manager.embeddings.collection_id'), str(col_info['id'])[:8] + "...")

                    with col_emb3:
                        # 컬렉션 삭제 버튼
                        if st.button(t('file_manager.embeddings.delete_button'), key=f"delete_col_{col_info['name']}"):
                            try:
                                vector_store = VectorStore(col_info['name'])
                                if vector_store.delete_collection():
                                    st.success(t('file_manager.embeddings.delete_completed', name=col_info['name']))
                                    st.rerun()
                                else:
                                    st.error(t('file_manager.embeddings.delete_failed'))
                            except Exception as e:
                                st.error(t('file_manager.embeddings.delete_error', error=e))

                    # 메타데이터 표시
                    if col_info.get('metadata'):
                        st.json(col_info['metadata'])

            # 데이터베이스 크기
            st.markdown("---")
            st.markdown(f"### {t('file_manager.embeddings.storage', name=settings.vector_store_name)}")

            db_size = VectorStore.get_database_size()
            if db_size:
                col_db1, col_db2, col_db3 = st.columns(3)
                with col_db1:
                    st.caption(t('file_manager.embeddings.total_size'))
                    st.text(f"{db_size['total_size_mb']:.2f} MB")
                with col_db2:
                    st.caption(t('file_manager.embeddings.file_count'))
                    st.text(db_size['file_count'])
                with col_db3:
                    st.caption(t('file_manager.embeddings.path'))
                    st.text(db_size['path'])
        else:
            st.info(t('file_manager.embeddings.no_collections'))

    # 중복 관리
    with fm_tab3:
        st.markdown(f"### {t('file_manager.duplicates.title')}")

        col_dup1, col_dup2 = st.columns(2)

        with col_dup1:
            st.markdown("**중복 파일 검사**")
            st.caption("같은 URL에서 다운로드된 중복 파일 찾기")

            if st.button("🔍 중복 파일 검사", key="check_dup_files"):
                with st.spinner("중복 파일 검사 중..."):
                    dup_result = file_manager.find_duplicate_files()

                    # 세션에 저장
                    st.session_state.duplicate_files = dup_result

                    if dup_result["duplicate_count"] > 0:
                        st.warning(f"⚠️ {dup_result['duplicate_count']}개 URL의 중복 파일 발견!")

                        with st.expander(f"중복 파일 목록 ({dup_result['total_duplicate_files']}개 파일)"):
                            for url_hash, files in dup_result["duplicates"].items():
                                st.markdown(f"**URL 해시: {url_hash}**")
                                for f in files:
                                    st.text(f"  📁 {f['folder']}/{f['filename']} ({f['size_mb']:.2f} MB)")
                                st.markdown("---")
                    else:
                        st.success("✅ 중복 파일이 없습니다!")

            # 중복 파일 일괄 정리
            if hasattr(st.session_state, 'duplicate_files') and st.session_state.duplicate_files.get("duplicate_count", 0) > 0:
                st.markdown("---")
                st.markdown("**⚡ 일괄 정리**")

                # 정리 옵션
                cleanup_option = st.radio(
                    "정리 방식:",
                    ["최신 파일만 남기기", "수동 선택"],
                    key="dup_cleanup_option"
                )

                if cleanup_option == "최신 파일만 남기기":
                    st.caption("각 URL의 중복 파일 중 최신 파일만 남기고 나머지 삭제")

                    # 삭제할 파일 미리보기
                    total_to_delete = 0
                    for url_hash, files in st.session_state.duplicate_files["duplicates"].items():
                        # 최신 파일 제외 나머지 삭제
                        files_sorted = sorted(files, key=lambda x: x["modified_time"], reverse=True)
                        total_to_delete += len(files_sorted) - 1

                    st.info(f"ℹ️ {total_to_delete}개 파일이 삭제됩니다")

                    if st.button(f"🗑️ {total_to_delete}개 중복 파일 삭제", type="primary", key="delete_duplicates_auto"):
                        deleted_count = 0

                        with st.spinner("중복 파일 삭제 중..."):
                            for url_hash, files in st.session_state.duplicate_files["duplicates"].items():
                                # 최신 파일 제외 나머지 삭제
                                files_sorted = sorted(files, key=lambda x: x["modified_time"], reverse=True)

                                for file_info in files_sorted[1:]:  # 첫 번째(최신) 제외
                                    if file_manager.delete_file(file_info['path']):
                                        deleted_count += 1

                        if deleted_count > 0:
                            st.success(f"✅ {deleted_count}개 중복 파일 삭제 완료!")
                            del st.session_state.duplicate_files
                            st.rerun()
                        else:
                            st.error("❌ 파일 삭제 실패")

                elif cleanup_option == "수동 선택":
                    st.caption("삭제할 파일을 직접 선택하세요")
                    st.info("ℹ️ 수동 선택 기능은 추후 추가 예정입니다")

        with col_dup2:
            st.markdown("**고아 파일 검사**")
            st.caption("임베딩되지 않은 다운로드 파일 찾기")

            if st.button("🔍 고아 파일 검사", key="check_orphan_files"):
                with st.spinner("고아 파일 검사 중..."):
                    # 모든 컬렉션의 임베딩된 해시 수집
                    all_embedded_hashes = set()
                    collections = VectorStore.get_all_collections_info()

                    for col_info in collections:
                        try:
                            vector_store = VectorStore(col_info['name'])
                            hashes = vector_store.get_embedded_hashes()
                            all_embedded_hashes.update(hashes)
                        except:
                            pass

                    orphan_files = file_manager.get_orphan_files(list(all_embedded_hashes))

                    # 세션에 저장
                    st.session_state.orphan_files = orphan_files

                    if orphan_files:
                        st.warning(f"⚠️ {len(orphan_files)}개 고아 파일 발견!")

                        with st.expander(f"고아 파일 목록 ({len(orphan_files)}개)"):
                            for f in orphan_files[:30]:
                                st.text(f"📄 {f['filename']} ({f['size_mb']:.2f} MB)")
                            if len(orphan_files) > 30:
                                st.text(f"... 외 {len(orphan_files) - 30}개")
                    else:
                        st.success("✅ 모든 파일이 임베딩되었습니다!")

            # 고아 파일 일괄 임베딩
            if hasattr(st.session_state, 'orphan_files') and st.session_state.orphan_files:
                st.markdown("---")
                st.markdown("**⚡ 일괄 임베딩**")

                # 컬렉션 선택
                target_collection = st.selectbox(
                    "임베딩할 컬렉션:",
                    settings.predefined_collections + ["+ 새 컬렉션 만들기"],
                    key="orphan_collection_select"
                )

                # 새 컬렉션 입력
                if target_collection == "+ 새 컬렉션 만들기":
                    target_collection = st.text_input(
                        "새 컬렉션명:",
                        placeholder="orphan_files_collection",
                        key="orphan_new_collection"
                    )

                if target_collection and target_collection != "+ 새 컬렉션 만들기":
                    col_batch1, col_batch2 = st.columns(2)

                    with col_batch1:
                        st.caption(f"대상 파일: {len(st.session_state.orphan_files)}개")

                    with col_batch2:
                        st.caption(f"대상 컬렉션: {target_collection}")

                    if st.button(f"⚡ {len(st.session_state.orphan_files)}개 파일 일괄 임베딩", type="primary", key="batch_embed_orphans"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            # 파일 경로 추출
                            file_paths = [f['path'] for f in st.session_state.orphan_files]

                            status_text.text(f"🚀 {len(file_paths)}개 고아 파일 임베딩 시작...")

                            # 진행상황 콜백
                            def batch_callback(current, total):
                                progress_bar.progress(current / total)
                                status_text.text(f"🧠 임베딩 중... {current}/{total} 청크")

                            # RAG 구축
                            success = asyncio.run(
                                st.session_state.rag_builder.build_from_documents(
                                    file_paths,
                                    target_collection,
                                    progress_callback=batch_callback
                                )
                            )

                            progress_bar.progress(1.0)

                            if success:
                                status_text.empty()
                                st.success(f"✅ {len(file_paths)}개 고아 파일 임베딩 완료!")

                                # 통계 표시
                                try:
                                    vector_store = VectorStore(target_collection)
                                    stats = vector_store.get_collection_stats()
                                    if stats:
                                        st.info(f"📊 컬렉션 '{target_collection}': 총 {stats['total_documents']}개 청크")
                                except:
                                    pass

                                # 세션 정리
                                del st.session_state.orphan_files
                            else:
                                status_text.empty()
                                st.error("❌ 임베딩 실패")

                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"❌ 오류 발생: {e}")

    # 통계/시각화
    with fm_tab4:
        st.markdown("### 📊 통계 & 시각화")

        if st.button("📈 통계 새로고침", key="refresh_stats"):
            st.rerun()

        stats = file_manager.get_statistics()

        if stats:
            # 전체 개요
            st.markdown("**📌 전체 개요**")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("전체 파일", f"{stats['total_files']}개")
            with col_stat2:
                st.metric("전체 폴더", f"{stats['total_folders']}개")
            with col_stat3:
                st.metric("저장 용량", f"{stats['total_size_mb']:.1f} MB")
            with col_stat4:
                st.metric("평균 파일 크기", f"{stats['avg_file_size_mb']:.2f} MB")

            st.markdown("---")

            # 상위 도메인
            if stats.get("top_domains"):
                st.markdown("**🌐 상위 다운로드 도메인 (Top 10)**")

                df_domains = pd.DataFrame(stats["top_domains"], columns=["도메인", "파일 수"])

                fig = px.bar(
                    df_domains,
                    x="파일 수",
                    y="도메인",
                    orientation='h',
                    title="도메인별 다운로드 파일 수",
                    color="파일 수",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, width="stretch")

            # 날짜별 분포
            if stats.get("date_distribution"):
                st.markdown("---")
                st.markdown("**📅 날짜별 다운로드 분포**")

                df_dates = pd.DataFrame(list(stats["date_distribution"].items()), columns=["날짜", "파일 수"])
                df_dates = df_dates.sort_values("날짜")

                fig = px.line(
                    df_dates,
                    x="날짜",
                    y="파일 수",
                    title="날짜별 다운로드 추이",
                    markers=True
                )
                st.plotly_chart(fig, width="stretch")

            # 컬렉션 분포
            collections = VectorStore.get_all_collections_info()
            if collections:
                st.markdown("---")
                st.markdown("**📦 컬렉션별 임베딩 분포**")

                df_collections = pd.DataFrame([
                    {"컬렉션": c["name"], "청크 수": c["document_count"]}
                    for c in collections
                ])

                fig = px.pie(
                    df_collections,
                    names="컬렉션",
                    values="청크 수",
                    title="컬렉션별 청크 수 분포"
                )
                st.plotly_chart(fig, width="stretch")

            # 태그 통계
            all_tags = file_manager.get_all_tags()
            if all_tags:
                st.markdown("---")
                st.markdown("**🏷️ 태그 사용 통계**")

                # 상위 20개 태그만
                top_tags = dict(list(all_tags.items())[:20])

                df_tags = pd.DataFrame([
                    {"태그": f"#{tag}", "사용 횟수": count}
                    for tag, count in top_tags.items()
                ])

                fig = px.bar(
                    df_tags,
                    x="사용 횟수",
                    y="태그",
                    orientation='h',
                    title="태그 사용 빈도 (Top 20)",
                    color="사용 횟수",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, width="stretch")

                # 태그 클라우드 (텍스트)
                st.markdown("**태그 클라우드**")
                tag_cloud = " · ".join([f"#{tag}({count})" for tag, count in list(all_tags.items())[:30]])
                st.info(tag_cloud)

            # 즐겨찾기 통계
            favorite_count = len(file_manager.get_favorite_files())
            if favorite_count > 0:
                st.markdown("---")
                st.markdown("**⭐ 즐겨찾기**")
                st.metric("즐겨찾기 파일", f"{favorite_count}개")

        else:
            st.info("통계 데이터가 없습니다")

# 정보 탭
with info_tab:
    st.subheader(t('info.title'))

    item_style = """
        font-size: 14px;
        margin-bottom: 12px; /* 항목 간의 세로 간격 */
    """

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)

    with col_info1:
        st.markdown(f"<h4>{t('info.embedding_model')}</h4>", unsafe_allow_html=True)
        current_engine = st.session_state.rag_builder.embedding_engine
        model_info = current_engine.get_model_info()
        st.markdown(f"<div style='{item_style}'><b>Type</b>: {model_info['type']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>Model</b>: {model_info['model_name']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>Dimension</b>: {model_info['dimension']}</div>", unsafe_allow_html=True)

        status_text = t('info.status_available') if model_info['is_loaded'] else t('info.status_failed')
        st.markdown(f"<div style='{item_style}'><b>Status</b>: {status_text}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) # 섹션 간격 추가

    with col_info2:
        st.markdown(f"<h4>{t('info.vector_db')}</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>Type</b>: {st.session_state.rag_builder.vector_store_type}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>Path</b>: {settings.vector_db_directory}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>Chunk Size</b>: {settings.chunk_size}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>Overlap</b>: {settings.chunk_overlap}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) # 섹션 간격 추가

    with col_info3:
        st.markdown(f"<h4>{t('info.api_settings')}</h4>", unsafe_allow_html=True)
        if settings.perplexity_api_key:
            st.markdown(f"<div style='{item_style}'><b>Perplexity API: </b>{t('info.perplexity_configured')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='{item_style}'><b>Perplexity API: </b>{t('info.perplexity_not_configured')}</div>", unsafe_allow_html=True)

        if settings.firecrawl_api_key:
            st.markdown(f"<div style='{item_style}'><b>Firecrawl API: </b>{t('info.firecrawl_configured')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='{item_style}'><b>Firecrawl API: </b>{t('info.firecrawl_not_configured')}</div>", unsafe_allow_html=True)

        if settings.openai_api_key:
            st.markdown(f"<div style='{item_style}'><b>OpenAI API: </b>{t('info.openai_configured')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='{item_style}'><b>OpenAI API: </b>{t('info.openai_not_configured')}</div>", unsafe_allow_html=True)

    with col_info4:
        st.markdown(f"<h4>{t('info.system')}</h4>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>OS</b>: {platform.system()} {platform.release()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>CPU</b>: {psutil.cpu_count()}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='{item_style}'><b>RAM</b>: {psutil.virtual_memory().total // (1024**3)}GB</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True) # 섹션 간격 추가

    # 벡터 DB 통계
    st.markdown(f"### {t('info.vector_db_stats')}")

    # 컬렉션 선택
    available_collections = settings.predefined_collections
    selected_collection = st.selectbox(
        t('info.collection_select'),
        available_collections,
        help=t('info.collection_help')
    )

    if st.button(t('info.refresh_stats')):
        try:
            from rag_builder.embedder.vector_store import VectorStore
            vector_store = VectorStore(selected_collection)
            stats = vector_store.get_collection_stats()

            if stats:
                col_stat1, col_stat2, col_stat3 = st.columns(3)

                with col_stat1:
                    st.caption(t('info.collection_name_label'))
                    st.write(stats.get('collection_name_label', 'N/A'))

                with col_stat2:
                    st.caption(t('info.total_chunks'))
                    st.write(stats.get('total_documents', 0))

                with col_stat3:
                    st.caption(t('info.storage_path'))
                    st.write(f"{settings.vector_db_directory}")

                st.success(t('info.stats_updated', name=selected_collection))
            else:
                st.warning(t('info.collection_empty', name=selected_collection))

        except Exception as e:
            error_msg = str(e)
            st.error(t('info.stats_failed', error=error_msg))

            with st.expander(t('info.error_details')):
                st.code(error_msg)
