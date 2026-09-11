"""
8주차: 쿼리 동의어 확장 함수
- synonyms.json을 로드하여 쿼리에서 의학 용어를 확장
- 한국어 조사 처리: 2글자 이상인 term이 쿼리에 포함되면 매칭
- 단글자 오매칭 방지 (예: "이" = tooth 가 "염증이" 에 매칭되는 오류 방지)
- 짧은 약어 오매칭 방지 (예: "mi"가 "abdomino", "hemi" 등에 매칭되는 오류 방지)
"""

import json
import os


def load_synonyms(path: str = None) -> dict:
    """synonyms.json 로드"""
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "synonyms.json"),
            os.path.join(base, "..", "8주차", "synonyms.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_query(query: str, synonyms: dict = None) -> list[str]:
    """
    쿼리에서 의학 용어를 찾아 동의어 목록을 포함한 확장 쿼리 반환.

    매칭 규칙:
    - term이 2글자 이상이고, term이 query 안에 포함될 때만 매칭
      (예: "염증" in "염증이 뭐야?" → True, 한국어 조사 처리)
    - 1글자 term: 완전 일치만 허용
    - query in term 방향 검사는 제외 (MI → abdomino 오매칭 방지)

    Returns:
        원본 쿼리 + 모든 관련 동의어를 포함한 리스트
    """
    if synonyms is None:
        synonyms = load_synonyms()

    query_lower = query.lower().strip()
    results = [query]  # 원본 쿼리 포함

    for key, synonyms_list in synonyms.items():
        all_terms = [key] + synonyms_list
        matched = False

        for term in all_terms:
            term_lower = term.lower().strip()

            if len(term_lower) == 0:
                continue

            if len(term_lower) == 1:
                # 단글자: 완전 일치만
                if term_lower == query_lower:
                    matched = True
                    break
            else:
                # 2글자 이상: term이 query 안에 포함되면 매칭
                # (한국어 조사 처리: "염증" in "염증이 뭐야?" → True)
                # query in term 방향은 제외 (MI → abdomino 오매칭 방지)
                if term_lower in query_lower:
                    matched = True
                    break

        if matched:
            for term in all_terms:
                if term not in results:
                    results.append(term)

    return results


# 테스트
if __name__ == "__main__":
    test_queries = [
        "고혈압",
        "HTN",
        "MI",
        "CVA",
        "염증이 뭐야?",
        "악성 종양이란?",
    ]
    for q in test_queries:
        expanded = expand_query(q)
        print(f"'{q}' → {expanded}")
