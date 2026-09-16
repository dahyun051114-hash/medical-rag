"""
14주차: 의학용어 RAG 검색 데모 (Streamlit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UI:   st.text_input / st.button / st.expander
파이프라인: 동의어 확장 → Gemini 임베딩 → hybrid_search → LLM 응답
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os, sys, time, json, re
import streamlit as st
from pathlib import Path

# ── API 키 로드 ──────────────────────────────────────────────────────
def load_secrets():
    try:
        return (
            st.secrets["GEMINI_API_KEY"],
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_SERVICE_KEY"],
        )
    except Exception:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=r"C:\Users\이다현\.env")
        return (
            os.getenv("GEMINI_API_KEY"),
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_SERVICE_KEY"),
        )

GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY = load_secrets()

_SYN_PATH = Path(__file__).parent / "synonyms.json"
with open(_SYN_PATH, encoding="utf-8") as _f:
    SYNONYMS = json.load(_f)

from supabase import create_client
from google import genai
from google.genai import types
from query_expander import expand_query

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client   = genai.Client(api_key=GEMINI_API_KEY)

EMBED_MODEL = "gemini-embedding-001"
FLASH_MODEL = "gemini-2.5-flash"


# ── Supabase 캐시 ────────────────────────────────────────────────────
def cache_get(query: str):
    try:
        res = supabase.table("rag_cache").select("answer").eq("query", query).execute()
        if res.data:
            return res.data[0]["answer"]
    except Exception:
        pass
    return None

def cache_set(query: str, answer: str):
    try:
        supabase.table("rag_cache").upsert({"query": query, "answer": answer}).execute()
    except Exception:
        pass


# ── 파이프라인 함수들 ────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    for attempt in range(3):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            if "429" in str(e) or "RATE" in str(e).upper():
                time.sleep(10 * (attempt + 1))
            else:
                raise
    raise RuntimeError("임베딩 생성 실패 (rate limit)")


def hybrid_search(query: str, embedding: list, match_count: int = 15) -> list:
    result = supabase.rpc("hybrid_search", {
        "query_text":      query,
        "query_embedding": embedding,
        "match_count":     match_count,
        "vector_weight":   0.7,
        "fts_weight":      0.3,
        "source_filter":   None
    }).execute()
    return result.data or []


def expanded_rrf_search(query: str, match_count: int = 10) -> list:
    expanded = expand_query(query)
    search_queries = list(dict.fromkeys(expanded))[:4]
    rrf_scores, rrf_data = {}, {}
    for eq in search_queries:
        try:
            emb     = get_embedding(eq)
            results = hybrid_search(eq, emb, match_count)
            for rank, item in enumerate(results):
                cid = item["id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
                rrf_data[cid]   = item
            time.sleep(0.3)
        except Exception:
            continue
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
    return [rrf_data[cid] for cid in sorted_ids[:match_count]]


def build_prompt(query: str, chunks: list) -> str:
    context = "\n\n".join([
        f"[자료 {i+1}]\n{c['chunk_text']}"
        for i, c in enumerate(chunks)
    ])
    return f"""당신은 의학용어 전문 교육 도우미입니다.

[규칙]
1. 반드시 아래 참고 자료의 내용만 사용하여 답하세요.
2. 출처 표시([자료 N] 등)는 절대 하지 마세요.
3. 영문 의학 용어가 나오면 반드시 한국어 명칭도 함께 표기하세요.
4. 자료에 없는 추가적인 의학 지식이나 배경 지식은 포함하지 마세요.
5. 자료에 질문과 직접 관련된 내용이 부분적으로라도 있으면, 그 내용을 조합하여 최대한 답변하세요.
6. 자료에 전혀 관련 내용이 없을 때만 "제공된 자료에서 해당 정보를 찾을 수 없습니다."라고 답하세요.

===== 참고 자료 =====
{context}
====================

질문: {query}

답변:"""


def generate_answer(prompt: str) -> str:
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
        )
    )
    text = response.text.strip()
    text = re.sub(r'\[자료\s*\d+\]', '', text).strip()
    return text


def run_pipeline(query: str) -> dict:
    cached = cache_get(query)
    if cached:
        return {"answer": cached, "chunks": [], "expanded": []}
    expanded = expand_query(query)[:3]
    chunks   = expanded_rrf_search(query)
    if not chunks:
        return {"answer": None, "chunks": [], "expanded": expanded}
    prompt = build_prompt(query, chunks)
    answer = generate_answer(prompt)
    cache_set(query, answer)
    return {"answer": answer, "chunks": chunks, "expanded": expanded}


def run_direct(query: str) -> str:
    prompt = f"""당신은 의학용어 전문 교육 도우미입니다.
아래 질문에 대해 알고 있는 의학 지식을 바탕으로 답변해주세요.
영문 의학 용어가 나오면 반드시 한국어 명칭도 함께 표기하세요.

질문: {query}

답변:"""
    response = client.models.generate_content(
        model=FLASH_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2048,
        )
    )
    return response.text.strip()


# ── Streamlit UI ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="의학용어 검색",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 의학용어 검색 시스템")
st.caption("의학용어 학습 도우미 | Gemini + Supabase")

st.divider()

rag_mode = st.toggle("✨ 스마트 검색 모드", value=True)
if rag_mode:
    st.caption("✨ 스마트 검색 모드 활성화")
else:
    st.caption("💬 일반 모드")

query = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 고혈압이란 무엇인가요? / What is hypertension?",
    max_chars=200,
)

search_btn = st.button("🔍 검색", type="primary", use_container_width=True)

if search_btn:
    if not query.strip():
        st.warning("질문을 입력해주세요!")
    else:
        with st.spinner("검색 중... 잠시만 기다려주세요 🔄"):
            try:
                t0 = time.time()

                if rag_mode:
                    result   = run_pipeline(query.strip())
                    elapsed  = round(time.time() - t0, 1)
                    answer   = result.get("answer")
                    expanded = result.get("expanded", [])

                    if answer is None:
                        st.info("📭 제공된 자료에서 해당 정보를 찾을 수 없습니다.")
                    else:
                        st.success("✅ 답변")
                        st.markdown(answer)
                        st.caption(f"⏱️ 응답 시간: {elapsed}초")
                        if len(expanded) > 1:
                            with st.expander("🔤 동의어 확장 결과 보기"):
                                st.write("입력 질문에서 아래 용어들로 검색을 확장했습니다:")
                                st.code(", ".join(expanded))
                else:
                    answer  = run_direct(query.strip())
                    elapsed = round(time.time() - t0, 1)
                    st.success("✅ 답변")
                    st.markdown(answer)
                    st.caption(f"⏱️ 응답 시간: {elapsed}초")

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    st.error("⚠️ API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
                elif "timeout" in err.lower() or "deadline" in err.lower():
                    st.error("⏰ 요청 시간이 초과됐습니다. 다시 시도해주세요.")
                elif "SUPABASE" in err.upper() or "connection" in err.lower():
                    st.error("🔌 데이터베이스 연결에 실패했습니다. 잠시 후 다시 시도해주세요.")
                else:
                    st.error(f"❌ 오류가 발생했습니다: {err}")

st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "의학용어 학습 도우미 | 본 서비스는 교육 목적으로 제작되었습니다."
    "</div>",
    unsafe_allow_html=True,
)
