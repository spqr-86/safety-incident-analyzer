# Evidence Assessment Node (V8 RAG Pattern)

Паттерн для оценки полноты retrieved evidence перед генерацией ответа.
Источник: брейншторм сессия 2026-03-28, проблема safety-incident-analyzer.

## Проблема которую решает

Router классифицирует simple/complex до retrieval — не видит документы. 
Proxy-метрика top_score не говорит о полноте ответа.

## Решение

Сдвинуть ключевое решение post-retrieval: initial_retrieve → evidence_assess → route.

## Три ветки routing

- `fast_answer`: top_score >= 0.72 + coverage_ratio >= 0.85 + нет missing required facets
- `expand_local`: есть опора, но crossref_hits/перечисления/исключения → нужны соседние пункты  
- `decompose_global`: слабый coverage → LLM-декомпозиция на подвопросы

## Ключевые сигналы (все детерминированные, без LLM)

```python
ENUM_PATTERNS = [r"\bкто\b", r"\bкакие\b", r"\bкатегори[яи]\b", r"\bв каких случаях\b", ...]
EXCEPTION_PATTERNS = [r"\bисключени[ея]\b", r"\bза исключением\b", r"\bне требуется\b", ...]
CROSSREF_PATTERNS = [r"\bпункт[а-я]*\s+\d+", r"\bв соответствии с\b", ...]
LIST_PATTERNS = [r"^\s*\d+\)", r"\bа также\b", r"\bв том числе\b", ...]

dispersion_score = (unique_sections / total) * (1 - max_doc_ratio)
```

## Subfacets (LLM optional, rule-based fallback)

Для "кто проходит обучение по программе А":
- subject_group (required)
- program_type (required)
- conditions (required)
- exceptions (optional)

## Метрики для eval set

- `fact_recall`: сколько gold-атомов найдено retrieval-ом
- `answer_completeness`: сколько gold-атомов вошло в ответ
- `unsupported_rate`: атомы ответа не подтверждённые evidence
- `false_sufficiency_rate`: evaluate сказал "достаточно" при completeness < 0.8

## LangGraph wiring

```python
def route_after_evidence_assess(state: GraphState) -> NextNode:
    return cast(NextNode, state["route_decision"])

builder.add_conditional_edges(
    "evidence_assess",
    route_after_evidence_assess,
    {"fast_answer": "fast_answer", "expand_local": "expand_local", "decompose_global": "decompose_global"},
)
```

## V8 Pipeline

```
query → intent_gate → initial_hybrid_retrieve → evidence_assess
                                                 ↓
                                     fast_answer | expand_local | decompose_global
                                                 ↓
                                         completeness_check → generate
```

## Что НЕ готово в этом коде

- `_estimate_facet_coverage`: token matching, не semantic — пропустит парафразы
- `expand_local`: только собирает section_id-якоря, не реализует саму экспансию
- `_extract_subfacets_with_gemini`: pseudocode, нужна реальная Gemini schema
