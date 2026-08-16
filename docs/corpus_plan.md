# Corpus Generation Plan — MVT-CS Main Experiment

Status: DRAFT — awaiting approval before generation begins.  
Prepared: 2026-08-04

---

## 0. Prerequisites

- D5 diagnostic PASS ✓ (2026-08-04)
- M0 probe BF16 throughput: **pending** (running; numbers inserted after probe completes)
- Approval of this document required before any `gen-prefixes` or `run-d5` call on main corpus.

---

## 1. Models

두 모델은 서로 다른 base architecture 계열이어야 한다. Qwen 계열 두 개는 안 된다.

| ID | HF path | Base arch | Thinking mode | BF16 size |
|----|---------|-----------|--------------|-----------|
| M1 | `Qwen/Qwen3-8B` | Qwen3 (Qwen) | Native (`<think>…</think>`) | ~16 GB |
| M2 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Llama 3.1 (Meta) | Native (`<think>…</think>`) | ~16 GB |

**M2 선택 근거:** DeepSeek-R1-Distill-Llama-8B는 Llama 3.1 8B 기반으로 DeepSeek-R1에서 증류되었다. 네이티브 thinking 모드(동일한 `<think>` 토큰 구조), 단일 24 GB 카드에서 BF16 적재 가능, Qwen과 아키텍처 계열이 완전히 다름.

**M2 사전 확인 필요 사항 (생성 전):**
1. `<think>` / `</think>` 토큰 ID가 M1과 다를 수 있음 → tokenizer 에서 확인 후 `diag_logits.py`에 M2 전용 상수 추가
2. 답변 cue(`"\n</think>\n\nThe answer is **"`) 이후 top-1 토큰이 bare letter인지 M2에서도 별도 검증 (D5 cue 검증 절차 동일하게 적용)
3. M0 probe (BF16) 를 M2로도 실행 → 3090a/3090b 로짓 일치 확인

---

## 2. Datasets

모두 객관식(MCQ) 형식. D5 검증 절차(answer cue → bare letter rank-1)와 동일하게 적용 가능.

| ID | Dataset | HF path | Split | Items | Choices | Domain |
|----|---------|---------|-------|------:|---------|--------|
| DS1 | MMLU-Pro | `TIGER-Lab/MMLU-Pro` | test | 1 000 | 10 | Mixed STEM/humanities |
| DS2 | ARC-Challenge | `allenai/ai2_arc` (ARC-Challenge) | test | 1 000 | 4 | Science (elem–high school) |
| DS3 | GPQA-Diamond | `Idavidrein/gpqa` | gpqa_diamond | 198 | 4 | Graduate science |
| DS4 | MedQA (USMLE) | `bigbio/med_qa` (source=usmle) | test | 500 | 4–5 | Medical/clinical |

**DS1:** 12k+ test items 존재. 1000개를 동일한 seed 로 stratified sample (10개 subject 균등). D5의 300개는 별도 diagnostic용이며 main corpus에 포함하지 않는다 (항목 겹침 방지).  
**DS3:** 전수 사용 (198개 전부). 소규모이나 난이도가 높아 budget 효과가 뚜렷할 것으로 예상.  
**DS4:** 5지선다가 섞여 있음. OPTION_IDS에서 E(토큰 34)까지 유효; 5지선다 문항은 n_options=5 로 기록.

**모델당 합계:** 1 000 + 1 000 + 198 + 500 = **2 698 항목**  
**전체 corpus:** 2 × 2 698 = **5 396 항목**  
**Budget grid:** [256, 512, 1024, 2048, 4096, 8192] (6개, BUDGET_GRID 불변)  
**Prefix 수:** 5 396 × 6 = **32 376 prefix slots**

---

## 3. GPU 할당

**원칙:** (모델 × 데이터셋) 셀 하나를 단일 카드에 고정. 아키텍처 혼용 불가 (CLAUDE.md rule 4).

| Model | Dataset | GPU | Arch | 근거 |
|-------|---------|-----|------|------|
| M1 Qwen3-8B | DS1 MMLU-Pro | 4090 (GPU 0) | sm_89 | 4090 최고 처리량; prefix 캐시 히트율 높음 |
| M1 Qwen3-8B | DS2 ARC-Challenge | 4090 (GPU 0) | sm_89 | 동일 카드 연속 실행 (KV 캐시 재활용 극대화) |
| M1 Qwen3-8B | DS3 GPQA-Diamond | 4090 (GPU 0) | sm_89 | 198항목, 짧음 |
| M1 Qwen3-8B | DS4 MedQA | 4090 (GPU 0) | sm_89 | 모든 M1 셀을 4090 단독에 고정 |
| M2 DeepSeek-Llama | DS1 MMLU-Pro | 3090a (GPU 1) | sm_86 | D5-b 통과 → 3090a/3090b Ampere 아키 일치 확인 |
| M2 DeepSeek-Llama | DS2 ARC-Challenge | 3090a (GPU 1) | sm_86 | 동일 Ampere 카드 |
| M2 DeepSeek-Llama | DS3 GPQA-Diamond | 3090b (GPU 2) | sm_86 | 3090a가 DS2로 바쁠 때 병렬화 가능 |
| M2 DeepSeek-Llama | DS4 MedQA | 3090b (GPU 2) | sm_86 | 동일 Ampere 아키 (D5-b 기준 동일 모델) |

**주의:** M1과 M2 결과를 절대 같은 아키텍처 비교 분석에 혼용하지 않는다. 각 (M×DS) 셀은 단일 GPU tag 로 명시적으로 레이블.

---

## 4. 예상 소요 시간 및 디스크 용량

### 소요 시간 (BF16 probe 실측값, 2026-08-04)

M0 probe 실측 (`probe_*_bf16.json`, N=24 항목):

| GPU | s/item | traj tok/s | probe overhead | 2698항목 단독 | 8100항목 단독 |
|-----|-------:|----------:|:--------------:|--------------:|--------------:|
| 4090  (sm_89) | 9.19 | 424.4 | 7.0% | 6.9 h | 20.7 h |
| 3090a (sm_86) | 11.74 | 328.2 | 8.5% | 8.8 h | 26.4 h |
| 3090b (sm_86) | 10.00 | 385.0 | 8.4% | 7.5 h | 22.5 h |

BF16 동시 시퀀스 (8192-tok 기준): **~4개** (KV 1.12 GiB/seq, 가용 VRAM ~4.7 GiB after weights 15.3 GiB on 4090).  
리그 합산 처리량: **1058 items/hour** (세 GPU 병렬 시 3.40 s/item wall).

**병렬화 시나리오 (2-model plan):**
- M1 (4090): 2698항목 → **6.9 h**
- M2 (3090a + 3090b 분할): 1349항목씩 → 3090a=4.4 h, 3090b=3.8 h → M2 합계 **4.4 h**
- 벽시계 시간 = max(6.9, 4.4) = **6.9 h**

단, M2를 3090a/3090b로 분할 시 각 (model × dataset) 셀을 단일 카드에 고정해야 함 (rule 4). DS별로 카드를 배정해 분할 가능 (D5-b 통과 → 두 카드 결과 동일).

**모델 수 결정 기준:** 8100항목 단독 실행이 20–40 h 구간임 (3090a=26.4 h, 4090=20.7 h). 리그 병렬 실행은 7.7 h. 사용자 판단에 따라 3번째 모델 추가 가능.

### 디스크 용량 추정

| 파일 유형 | 모델당 크기 | 전체 |
|-----------|------------|------|
| prefix JSONL (token ID 배열) | ~280 MB × 2 | ~560 MB |
| result JSON (option_probs, metadata) | ~3 MB × 2 | ~6 MB |
| 생성 trajectory 원본 (캐시용) | ~500 MB × 2 | ~1 GB |
| **합계** | | **~1.6 GB** |

---

## 5. 증분 저장 및 재개 방식

`gen-prefixes` 는 이미 다음을 구현함 (commit 9f035d3 기준):

- **config_hash 가드:** `model + max_model_len + D5_BUDGETS + sampling` 의 SHA-256[:16]. 설정이 다른 기존 파일에 이어쓰기 시도 시 즉시 중단.
- **JSONL 증분 체크포인트:** 항목 하나 완료마다 JSONL 에 append. 중단 후 재실행 시 이미 완료된 항목 건너뜀.
- **하트비트:** 5분마다 완료 항목 수 출력.

`run-d5` (diag_logits.py) 의 main corpus 버전은 동일한 패턴을 적용할 것:
- 결과 JSONL 에 question_id + budget 조합 단위로 append
- 재실행 시 완료된 (question_id, budget) 쌍 건너뜀
- 출력 파일명 형식: `data/{model_tag}_{dataset_tag}_{gpu_tag}.jsonl`

---

## 6. 생성 전 사전 검증 항목

생성 시작 전 반드시 완료해야 하는 항목. 전부 통과 전까지 `gen-prefixes` 실행 금지.

### 6.1 모델 검증 (M2 신규)
- [ ] M2 tokenizer 로드 → `<think>` / `</think>` 토큰 ID 기록
- [ ] M2 answer cue 검증: 30개 MMLU-Pro 항목으로 `"\n</think>\n\nThe answer is **"` 이후 top-1 토큰이 bare letter인지 확인 (top-1 rate ≥ 95% 요구)
- [ ] M2 M0 probe 실행 (3090a, 3090b) → 로짓 일치 확인

### 6.2 프롬프트 길이 전수 검사
각 데이터셋의 **전체** 항목에 대해 (prompt_ids + cue_ids + B_MAX) ≤ max_model_len 확인.  
단 한 항목이라도 초과 시 해당 항목 제외 또는 max_model_len 조정 후 재검증.  
(D5에서 item 54가 초과하여 max_model_len을 8792→동적 계산으로 수정한 선례 있음)

구체적으로:
```
for each item i in dataset:
    for each budget b in BUDGET_GRID:
        assert len(prompt_ids[i]) + b + len(cue_ids) <= max_model_len
```

### 6.3 데이터 수집 검증
- [ ] 각 데이터셋 HF hub 에서 정상 다운로드 확인
- [ ] MMLU-Pro 1000개 stratified sample seed 고정 (seed=42, 10 subject 균등)
- [ ] GPQA-Diamond 전수 198개 확인 (중복 없음)
- [ ] MedQA 5지선다 항목 비율 확인 → n_options 필드 정확히 기록
- [ ] 각 데이터셋 answer_index 추출 로직 단위 테스트 (3개 항목 수동 검증)

### 6.4 config_hash 일관성
- [ ] M1 prefix 파일의 config_hash 가 M2와 독립적으로 생성되는지 확인 (모델명이 hash에 포함됨)
- [ ] BUDGET_GRID 변경 없음 (현재 [256, 512, 1024, 2048, 4096, 8192])
- [ ] SAMPLING 변경 없음 (temperature=0.6, top_p=0.95, seed=1234)

### 6.5 출력 경로 및 파일명 규칙
```
data/
  prefixes_m1_ds1_4090.jsonl      # gen-prefixes 출력
  prefixes_m1_ds2_4090.jsonl
  ...
  results_m1_ds1_4090.jsonl       # run (diag_logits 상당) 출력
  results_m1_ds2_4090.jsonl
  ...
```

---

## 7. 결과 저장 형식

D5의 `data/d5_{gpu_tag}.json` 형식을 그대로 계승. 각 레코드:

```json
{
  "question_id": "ds1_0042",
  "dataset": "mmlu_pro",
  "model": "Qwen/Qwen3-8B",
  "gpu_tag": "4090",
  "budget": 2048,
  "answer_index": 2,
  "n_options": 10,
  "option_probs": [0.02, 0.01, 0.91, ...],
  "raw_logprobs": [-3.91, -4.60, -0.09, ...],
  "top1_token_id": 34,
  "top1_is_option": true
}
```

---

## 8. 미결 사항 (승인 후 처리)

1. **M2 answer cue 검증** — M2가 동일한 cue에서 bare letter를 top-1으로 예측하는지 확인 필요. Llama 기반이라 chat template 및 thinking format이 다를 수 있음.
2. **BF16 probe 숫자** — probe 완료 후 섹션 4 표 채움.
3. **DS1 stratified sample 코드** — MMLU-Pro 10 subject 균등 샘플링 로직 구현 필요.
4. **run 스크립트 main corpus 버전** — `diag_logits.py`의 `run-d5` 명령을 일반화하거나 별도 스크립트 작성 필요. JSONL 증분 저장 포함.
