# Ollama_PDF_RAG_multi.py - 다중 PDF 지원 개선 버전
import gradio as gr
import ollama
import os
import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from config import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 캐시 디렉토리 생성
os.makedirs(config.vector_cache_dir, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """파일 해시 생성 (MD5)"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"파일 해시 생성 실패: {e}")
        raise

def validate_pdf_file(file_path: str) -> bool:
    """PDF 파일 검증"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("❌ PDF 파일만 지원됩니다.")
    
    # 파일 크기 체크
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > config.max_file_size_mb:
        raise ValueError(f"❌ 파일 크기가 너무 큽니다. (최대 {config.max_file_size_mb}MB, 현재: {file_size_mb:.2f}MB)")
    
    return True

def get_cache_path(file_hash: str) -> str:
    """캐시 경로 반환"""
    return os.path.join(config.vector_cache_dir, file_hash)

def load_pdf_documents(file_path: str, progress=None) -> List[Document]:
    """PDF 문서 로드"""
    try:
        if progress:
            progress(0.1, desc=f"📄 PDF 파일 로딩 중: {os.path.basename(file_path)}")
        
        validate_pdf_file(file_path)
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        
        if not docs:
            raise ValueError(f"❌ PDF에서 텍스트를 추출할 수 없습니다: {os.path.basename(file_path)}")
        
        # 메타데이터에 파일명 추가
        for doc in docs:
            doc.metadata['source_file'] = os.path.basename(file_path)
            doc.metadata['file_path'] = file_path
        
        logger.info(f"PDF 로드 완료: {os.path.basename(file_path)}, 페이지 수: {len(docs)}")
        return docs
    
    except Exception as e:
        logger.error(f"PDF 로드 실패: {e}")
        raise

def split_documents(docs: List[Document], progress=None) -> List[Document]:
    """문서 분할"""
    if progress:
        progress(0.3, desc="✂️ 텍스트 분할 중...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    splits = text_splitter.split_documents(docs)
    logger.info(f"문서 분할 완료: {len(splits)}개 청크 생성")
    return splits

def get_or_create_vectorstore_for_file(file_path: str, embeddings, progress=None) -> Optional[Chroma]:
    """단일 파일용 벡터 스토어 가져오기 또는 생성"""
    file_hash = get_file_hash(file_path)
    cache_path = get_cache_path(file_hash)
    
    # 기존 벡터 스토어 확인
    if os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "chroma.sqlite3")):
        try:
            if progress:
                progress(0.5, desc=f"♻️ 캐시된 벡터 스토어 로드 중: {os.path.basename(file_path)}")
            
            vectorstore = Chroma(
                persist_directory=cache_path,
                embedding_function=embeddings
            )
            logger.info(f"캐시된 벡터 스토어 로드: {os.path.basename(file_path)}")
            return vectorstore
        except Exception as e:
            logger.warning(f"캐시 로드 실패, 새로 생성: {e}")
    
    # 새로 생성
    if progress:
        progress(0.2, desc=f"📚 PDF 처리 중: {os.path.basename(file_path)}")
    
    docs = load_pdf_documents(file_path, progress)
    splits = split_documents(docs, progress)
    
    if progress:
        progress(0.6, desc=f"🔢 벡터 임베딩 생성 중: {os.path.basename(file_path)}")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=cache_path
    )
    vectorstore.persist()
    
    logger.info(f"벡터 스토어 생성 완료: {os.path.basename(file_path)}")
    return vectorstore

def process_multiple_pdfs(file_paths: List[str], progress=None) -> Chroma:
    """다중 PDF 처리 및 통합 벡터 스토어 생성"""
    if not file_paths:
        raise ValueError("❌ PDF 파일이 선택되지 않았습니다.")
    
    # 임베딩 모델 초기화
    embeddings = OllamaEmbeddings(model=config.embedding_model)
    
    # 모든 파일의 해시를 결합하여 통합 캐시 키 생성
    combined_hash = hashlib.md5()
    for file_path in sorted(file_paths):  # 정렬하여 순서 일관성 유지
        file_hash = get_file_hash(file_path)
        combined_hash.update(file_hash.encode())
    
    combined_hash_str = combined_hash.hexdigest()
    combined_cache_path = get_cache_path(f"combined_{combined_hash_str}")
    
    # 통합 벡터 스토어가 이미 존재하는지 확인
    if os.path.exists(combined_cache_path) and os.path.exists(os.path.join(combined_cache_path, "chroma.sqlite3")):
        try:
            if progress:
                progress(0.8, desc="♻️ 통합 벡터 스토어 로드 중...")
            
            vectorstore = Chroma(
                persist_directory=combined_cache_path,
                embedding_function=embeddings
            )
            logger.info("통합 벡터 스토어 캐시 로드 완료")
            return vectorstore
        except Exception as e:
            logger.warning(f"통합 캐시 로드 실패, 새로 생성: {e}")
    
    # 각 PDF 처리 및 통합
    all_splits = []
    total_files = len(file_paths)
    
    for idx, file_path in enumerate(file_paths):
        if progress:
            progress(
                (idx / total_files) * 0.7,
                desc=f"📚 PDF 처리 중 ({idx + 1}/{total_files}): {os.path.basename(file_path)}"
            )
        
        try:
            # 개별 파일 벡터 스토어 가져오기 또는 생성
            file_vectorstore = get_or_create_vectorstore_for_file(file_path, embeddings, progress)
            
            # 모든 문서 가져오기
            all_docs = file_vectorstore.get()
            for doc_id, doc_text, metadata in zip(
                all_docs['ids'],
                all_docs['documents'],
                all_docs['metadatas']
            ):
                doc = Document(page_content=doc_text, metadata=metadata)
                all_splits.append(doc)
        
        except Exception as e:
            logger.error(f"파일 처리 실패: {os.path.basename(file_path)}, 오류: {e}")
            raise ValueError(f"❌ 파일 처리 실패: {os.path.basename(file_path)}\n오류: {str(e)}")
    
    if not all_splits:
        raise ValueError("❌ 처리된 문서가 없습니다.")
    
    # 통합 벡터 스토어 생성
    if progress:
        progress(0.8, desc="🔗 통합 벡터 스토어 생성 중...")
    
    combined_vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=combined_cache_path
    )
    combined_vectorstore.persist()
    
    logger.info(f"통합 벡터 스토어 생성 완료: {len(all_splits)}개 문서, {total_files}개 파일")
    
    if progress:
        progress(1.0, desc="✅ 완료!")
    
    return combined_vectorstore

def format_docs(docs: List[Document]) -> str:
    """문서 포맷팅 (출처 정보 포함)"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source_file', 'Unknown')
        content = doc.page_content
        formatted.append(f"[출처: {source}]\n{content}")
    return "\n\n---\n\n".join(formatted)

def check_ollama_connection() -> bool:
    """Ollama 연결 확인"""
    try:
        models = ollama.list()
        logger.info("Ollama 연결 확인 완료")
        return True
    except Exception as e:
        logger.error(f"Ollama 연결 실패: {e}")
        return False

def rag_chain(files: List, question: str, progress=gr.Progress()) -> str:
    """RAG 체인 동작 (다중 PDF 지원)"""
    try:
        # 입력 검증
        if not question or not question.strip():
            return "❌ 질문을 입력해주세요."
        
        if not files:
            return "❌ PDF 파일을 업로드해주세요."
        
        # 파일 경로 추출
        file_paths = []
        if isinstance(files, list):
            file_paths = [f.name if hasattr(f, 'name') else str(f) for f in files if f]
        elif hasattr(files, 'name'):
            file_paths = [files.name]
        else:
            file_paths = [str(files)]
        
        # Ollama 연결 확인
        if not check_ollama_connection():
            return "❌ Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
        
        progress(0.0, desc="🚀 시작...")
        
        # 다중 PDF 처리
        vectorstore = process_multiple_pdfs(file_paths, progress)
        retriever = vectorstore.as_retriever(search_kwargs={"k": config.search_k})
        
        progress(0.85, desc="🔍 관련 문서 검색 중...")
        
        # 문서 검색
        retrieved_docs = retriever.invoke(question)
        
        if not retrieved_docs:
            return "❌ 관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해보세요."
        
        # 컨텍스트 구성
        context = format_docs(retrieved_docs)
        logger.info(f"검색된 문서 수: {len(retrieved_docs)}")
        
        # 프롬프트 구성
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

문서 내용:
{context}

질문: {question}

위 문서들의 내용을 바탕으로 정확하고 상세하게 답변해주세요. 답변은 한국어로 작성하고, 이모지를 적절히 사용해주세요."""
        
        progress(0.9, desc="🤖 AI 답변 생성 중...")
        
        # LLM 답변 생성
        response = ollama.chat(
            model=config.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": config.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        answer = response['message']['content']
        
        # 검색된 문서 출처 정보 추가
        sources = set([doc.metadata.get('source_file', 'Unknown') for doc in retrieved_docs])
        source_info = f"\n\n📚 참고 문서: {', '.join(sources)}"
        
        return answer + source_info
    
    except FileNotFoundError as e:
        logger.error(f"파일 오류: {e}")
        return f"❌ {str(e)}"
    except ValueError as e:
        logger.error(f"값 오류: {e}")
        return f"❌ {str(e)}"
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}", exc_info=True)
        return f"❌ 오류 발생: {str(e)}\n\n자세한 내용은 로그를 확인해주세요."

# Gradio 인터페이스
def create_interface():
    """Gradio 인터페이스 생성"""
    with gr.Blocks(title="다중 PDF RAG 시스템") as iface:
        gr.Markdown("""
        # 📚 다중 PDF 기반 질문 응답 시스템
        
        여러 PDF 파일을 업로드하고 질문을 입력하면, 모든 문서를 통합하여 답변해드립니다.
        
        **주요 기능:**
        - ✅ 다중 PDF 파일 동시 처리
        - ✅ 자동 캐시 재사용 (빠른 재질문)
        - ✅ 진행 상황 실시간 표시
        - ✅ 문서 출처 표시
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="PDF 파일 업로드 (다중 선택 가능)",
                    file_count="multiple",
                    file_types=[".pdf"],
                    height=200
                )
                
                question_input = gr.Textbox(
                    label="질문을 입력하세요",
                    placeholder="예: 이 문서들의 주요 내용은 무엇인가요?",
                    lines=3
                )
                
                submit_btn = gr.Button("🚀 질문하기", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="답변",
                    lines=15,
                    interactive=False
                )
        
        # 예시
        gr.Examples(
            examples=[
                ["이 문서들의 주요 주제는 무엇인가요?"],
                ["모든 문서에서 공통으로 언급되는 내용은 무엇인가요?"],
                ["각 문서의 핵심 요약을 제공해주세요."]
            ],
            inputs=question_input
        )
        
        # 이벤트 핸들러
        submit_btn.click(
            fn=rag_chain,
            inputs=[file_input, question_input],
            outputs=output
        )
        
        question_input.submit(
            fn=rag_chain,
            inputs=[file_input, question_input],
            outputs=output
        )
    
    return iface

if __name__ == "__main__":
    # Ollama 연결 확인
    if not check_ollama_connection():
        print("⚠️ 경고: Ollama 서버에 연결할 수 없습니다.")
        print("Ollama가 실행 중인지 확인해주세요: ollama serve")
    
    print("\n" + "="*60)
    print("🚀 다중 PDF RAG 시스템이 시작되었습니다!")
    print("="*60)
    print("📱 웹 브라우저에서 다음 주소로 접속하세요:")
    print("   → http://localhost:7860")
    print("   → http://127.0.0.1:7860")
    print("="*60 + "\n")
    
    iface = create_interface()
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

