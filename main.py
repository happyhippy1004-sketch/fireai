"""
소방 현장지휘 AI 참모 - 백엔드 서버
FastAPI + ChromaDB + Claude API (RAG 구조)

실행: python -m uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import anthropic
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber
import hashlib
import os
import io
from pathlib import Path

app = FastAPI(title="소방 AI 참모 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 설정 ──────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "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")
CHROMA_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

# ── 초기화 ─────────────────────────────────────────
client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_or_create_collection(
    name="fire_manuals",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)


# ── 유틸 함수 ──────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def search_manuals(query: str, n_results=TOP_K):
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "source": results["metadatas"][0][i].get("source", "알 수 없음"),
            "relevance": round(1 - results["distances"][0][i], 3)
        })
    return chunks


SYSTEM_PROMPT = """당신은 소방 현장지휘관을 지원하는 AI 참모입니다.
아래 [참조 매뉴얼] 내용을 최우선으로 활용하여 답변하세요.

규칙:
1. 매뉴얼에 관련 내용이 있으면 반드시 인용하고 출처를 명시하세요
2. 매뉴얼에 없는 내용은 소방 일반 지식으로 보완하되 "(일반 지식)" 표시
3. 현장에서 즉시 적용 가능한 간결한 언어 사용
4. 중요 순서는 번호로 구조화
5. 안전 경고, 확인사항, 즉시조치 아이콘 활용
6. 한국어로 답변"""


# ── API 엔드포인트 ──────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    manual_found: bool


@app.get("/")
def root():
    return {
        "status": "소방 AI 참모 서버 정상 가동",
        "manuals": collection.count()
    }


@app.get("/reload-manuals")
def reload_manuals():
    """C:\\FireAI\\manuals 폴더의 PDF 자동 등록"""
    manual_dir = Path("./manuals")

    if not manual_dir.exists():
        manual_dir.mkdir()
        return {"message": "manuals 폴더를 새로 만들었습니다. PDF를 넣고 다시 시도하세요.", "results": []}

    pdf_files = list(manual_dir.glob("*.pdf"))

    if not pdf_files:
        return {"message": "PDF 파일이 없습니다. C:\\FireAI\\manuals 폴더에 PDF를 넣으세요.", "results": []}

    results = []
    for pdf_file in pdf_files:
        try:
            pdf_bytes = pdf_file.read_bytes()
            file_hash = hashlib.md5(pdf_bytes).hexdigest()[:8]
            text = extract_text_from_pdf(pdf_bytes)

            if not text.strip():
                results.append({
                    "file": pdf_file.name,
                    "status": "텍스트 추출 실패 (스캔본은 OCR 필요)"
                })
                continue

            chunks = split_into_chunks(text)

            try:
                collection.delete(where={"file_hash": file_hash})
            except:
                pass

            ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": pdf_file.name,
                    "chunk_index": i,
                    "file_hash": file_hash
                }
                for i in range(len(chunks))
            ]

            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )

            results.append({
                "file": pdf_file.name,
                "chunks": len(chunks),
                "status": "등록완료"
            })

        except Exception as e:
            results.append({
                "file": pdf_file.name,
                "status": f"오류: {str(e)}"
            })

    return {
        "results": results,
        "total": len(results),
        "total_chunks": collection.count()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    sources = search_manuals(req.message)
    manual_found = len(sources) > 0 and sources[0]["relevance"] > 0.5

    context = ""
    if sources:
        context = "\n\n[참조 매뉴얼]\n"
        for i, s in enumerate(sources, 1):
            context += f"\n{i}. 출처: {s['source']} (관련도: {s['relevance']})\n{s['text']}\n"

    messages = req.history + [
        {"role": "user", "content": context + "\n\n질문: " + req.message}
    ]

    response = client_anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    return ChatResponse(
        answer=response.content[0].text,
        sources=sources[:3],
        manual_found=manual_found
    )


@app.get("/manuals")
def list_manuals():
    try:
        all_items = collection.get(include=["metadatas"])
        seen = {}
        for meta in all_items["metadatas"]:
            src = meta.get("source", "")
            if src not in seen:
                seen[src] = {
                    "filename": src,
                    "chunks": 0,
                    "hash": meta.get("file_hash", "")
                }
            seen[src]["chunks"] += 1
        return {
            "manuals": list(seen.values()),
            "total_chunks": collection.count()
        }
    except:
        return {"manuals": [], "total_chunks": 0}


@app.get("/export-cache")
def export_cache():
    all_items = collection.get(include=["documents", "metadatas"])
    cache = {
        "version": "1.0",
        "chunks": [
            {
                "id": all_items["ids"][i],
                "text": all_items["documents"][i],
                "source": all_items["metadatas"][i].get("source", ""),
            }
            for i in range(len(all_items["ids"]))
        ]
    }
    return cache
