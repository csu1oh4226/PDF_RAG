# config.py - 설정 관리 파일
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """RAG 시스템 설정"""
    # 벡터 스토어 설정
    vector_cache_dir: str = "chroma_pdf_cache"
    
    # 텍스트 분할 설정
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # 임베딩 모델 설정
    embedding_model: str = "mxbai-embed-large"
    
    # LLM 모델 설정
    llm_model: str = "llama3"
    
    # 검색 설정
    search_k: int = 4  # 검색할 문서 개수
    
    # 시스템 프롬프트
    system_prompt: str = "You are a helpful assistant. Read the PDF content and answer the question. Translate the answer in Korean with emoji."
    
    # 파일 크기 제한 (MB)
    max_file_size_mb: int = 100

# 전역 설정 인스턴스
config = RAGConfig()
