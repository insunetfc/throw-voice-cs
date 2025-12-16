#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_korean_variants_v2.py  —  general-purpose *limited* Korean utterance variant generator.

Features
- Expands ALL seeds (questions + statements) with light, theme-aware paraphrases.
- Optional fillers, tiny typo variants, and light dialect sprinkles (off/on).
- Trailing punctuation variants are selectable: --punct-mode none|minimal|rich.
- Groups outputs by the SAME canonical key your Lambda uses (NFKC→trim→lower→collapse spaces→strip .?!).
- Caps emitted variants per canonical base with --limit-per-base (default: 3).
- SIGPIPE-safe (won't crash when piped to `head`).

Usage examples
  python gen_korean_variants_v2.py --seeds seeds.txt --limit-per-base 3 --punct-mode none --no-typos --no-dialects
  echo -e "수수료가 얼마인가요?\n회의 중이에요." | python gen_korean_variants_v2.py --punct-mode minimal | head -n 20
"""
import sys, argparse, re, unicodedata, signal
from typing import List, Dict, Set

# --- handle downstream pipe closures (e.g., `| head`) cleanly ---
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# -------------------- Normalization helpers --------------------
def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_basic(s: str) -> str:
    """NFKC + trim + collapse spaces (case preserved)."""
    return normalize_whitespace(nfkc(s))

def canonical_key_for_lambda(s: str) -> str:
    """
    Mirror your Lambda's normalize_utt():
      NFKC → strip → lower → collapse spaces → strip trailing .?!
    """
    s = nfkc(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.!\?]+$", "", s)
    return s

def strip_trailing_punct(s: str) -> str:
    return re.sub(r"[.!?…~]+$", "", s).strip()

# preferred punctuation order for sorting
ORDER_ENDINGS = ["", ".", "?", "…", "!", "!!", "~"]

def order_key(v: str) -> int:
    m = re.search(r"[.!?…~]+$", v)
    token = m.group(0) if m else ""
    return ORDER_ENDINGS.index(token) if token in ORDER_ENDINGS else len(ORDER_ENDINGS)

# -------------------- Controls --------------------
PUNCT_ENDINGS = {
    "none":     [""],
    "minimal":  ["", ".", "?"],
    "rich":     ["", ".", "?", "…", "!", "!!", "~"],
}

def get_endings(mode: str) -> List[str]:
    return PUNCT_ENDINGS.get(mode, PUNCT_ENDINGS["minimal"])

# Lightweight fillers (kept conservative for production cleanliness)
FILLERS = ["", "혹시 ", "죄송하지만 ", "실례지만 ", "음 ", "아 ", "어 ", "잠깐만 ", "지금은 "]

# Very light typo variants (kept rare/harmless)
TYPOS = {
    r"바빠요": ["바뻐요", "바빠여"],
    r"괜찮":  ["괜챃", "괜찬"],  # only if stem contains '괜찮'
}

# Dialect tokens inserted AFTER stripping punctuation; we don't chain extra '요'
DIALECT_TOKENS = ["데이", "라요", "심더", "하모"]

# Theme-aware expansions (regex → list of paraphrases)
THEME_EXPANSIONS = [
    (r"(수수료|비용|요금)", [
        "수수료가 정확히 얼마인지 궁금해요",
        "추가 비용은 없는지 알려주세요",
        "최종 부담액이 얼마인지요",
    ]),
    (r"(왜|이유).*(7%|퍼센트|%)|7%|퍼센트|%", [
        "왜 7%인지 근거가 있을까요",
        "7% 산정 기준이 어떻게 되나요",
    ]),
    (r"(가입|가능|거절|조건)", [
        "가입 가능 여부가 궁금합니다",
        "어떤 조건이면 가입할 수 있나요",
        "사고 이력이 있어도 괜찮나요",
    ]),
    (r"(검증|허가|등록|금융감독원|사기|믿을|신뢰|정식 업체|허가증|후기|사례)", [
        "정식 등록업체 맞는지요",
        "허가증이나 등록 정보 볼 수 있나요",
        "실제 이용 후기나 사례가 있을까요",
    ]),
    (r"(온라인|오프라인|다이렉트|카카오톡|카톡|문자|이메일|홈페이지)", [
        "온라인/오프라인 모두 가능한가요",
        "문자나 카톡으로도 안내 받을 수 있나요",
        "이메일로 자료 받을 수 있을까요",
        "홈페이지 주소 알려주실 수 있나요",
    ]),
    (r"(연락처|명함|담당|누가|어디|회사 이름|누구세요)", [
        "담당자 연락처/명함 부탁드립니다",
        "어느 회사에서 연락 주신 건가요",
        "담당자분 성함과 직책을 알 수 있을까요",
    ]),
    (r"(언제|몇\s*분|시간|나중에|다시 전화|스케줄)", [
        "가능하신 시간대 알려주시면 맞춰 연락드릴게요",
        "몇 분 정도 통화 걸릴까요",
        "나중에 다시 연락 드려도 될까요",
    ]),
    (r"(필요 없|관심 없|안 할래|그만 연락|스팸|싫|귀찮|화가|신고)", [
        "이번에는 패스할게요",
        "관심이 없어 연락은 여기까지 부탁드려요",
        "연락 중단 요청드립니다",
    ]),
    (r"(회의|상담|운전|바빠|통화.*어려|시간이 안|곤란)", [
        "지금은 통화가 어려워요",
        "지금 바빠서 나중에 부탁드립니다",
        "회의 중이라 잠시 힘들어요",
    ]),
    (r"(견적|계약|처리|지급|언제 받을|증명서|세금)", [
        "견적/계약 일정과 처리 소요가 궁금해요",
        "수수료 지급 시점과 증빙 발급 가능 여부 알려주세요",
        "세금 처리는 어떻게 되는지요",
    ]),
]

# -------------------- Variant generators --------------------
def expand_theme(seed: str) -> List[str]:
    out: Set[str] = set()
    for patt, phrases in THEME_EXPANSIONS:
        if re.search(patt, seed):
            out.update(phrases)
    return list(out)

def apply_typos_once(s: str, enable: bool) -> List[str]:
    outs = [s]
    if not enable:
        return outs
    for patt, repls in TYPOS.items():
        if re.search(patt, s):
            for r in repls:
                outs.append(re.sub(patt, r, s, count=1))
    # dedupe preserving order
    seen: Set[str] = set()
    ordered = []
    for x in outs:
        if x not in seen:
            ordered.append(x)
            seen.add(x)
    return ordered

def sprinkle_dialect_clean(s: str, enable: bool) -> List[str]:
    if not enable:
        return [s]
    base = strip_trailing_punct(s)
    forms = [s]  # keep original
    for tok in DIALECT_TOKENS:
        forms.append(f"{base} {tok}")
    return forms

def generate_variants(seed: str,
                      limit_per_base: int,
                      endings: List[str],
                      typos: bool = True,
                      dialects: bool = True) -> List[str]:
    base = normalize_basic(seed)
    bases: Set[str] = set([base])

    # Theme-based paraphrases
    bases.update(expand_theme(base))

    # Short/yes-no style: add a couple steady stems
    if re.search(r"^(네|예|아니오|아뇨|아닙니다|괜찮|가능|불가|됩니다|안돼)", base):
        bases.update(["네, 알겠습니다", "아닙니다", "괜찮습니다"])

    out: Set[str] = set()
    for b in bases:
        b = normalize_basic(b)
        stems = apply_typos_once(b, typos)
        tmp: Set[str] = set()
        for s in stems:
            for form in sprinkle_dialect_clean(s, dialects):
                tmp.add(normalize_basic(form))

        for stem in tmp:
            for pref in FILLERS:
                core = normalize_basic(pref + stem)
                # de-dupe repeated '요' near end
                core = re.sub(r"(요){2,}(\s|$)", r"요\2", core)
                for end in endings:
                    v = normalize_basic(core + end)
                    if 2 <= len(v) <= 80:
                        out.add(v)

    # Group by canonical key (matches Lambda hashing)
    grouped: Dict[str, List[str]] = {}
    for s in out:
        key = canonical_key_for_lambda(s)
        grouped.setdefault(key, []).append(s)

    final: List[str] = []
    for _, variants in grouped.items():
        variants_sorted = sorted(variants, key=order_key)
        final.extend(variants_sorted[:limit_per_base])

    # Ensure original seed (normalized) is present
    final = list(dict.fromkeys([base] + final))
    return final

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", help="Path to newline-delimited seeds; if omitted, read stdin")
    ap.add_argument("--limit-per-base", type=int, default=3)
    ap.add_argument("--no-typos", action="store_true")
    ap.add_argument("--no-dialects", action="store_true")
    ap.add_argument("--punct-mode", choices=["none", "minimal", "rich"], default="minimal",
                    help="How many punctuation variants to generate per base")
    args = ap.parse_args()

    # Load seeds
    if args.seeds:
        with open(args.seeds, "r", encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f]
    else:
        lines = [l.rstrip("\n") for l in sys.stdin]

    endings = get_endings(args.punct_mode)
    emitted: Set[str] = set()

    try:
        for line in lines:
            seed = normalize_basic(line)
            if not seed:
                continue
            variants = generate_variants(
                seed,
                limit_per_base=args.limit_per_base,
                endings=endings,
                typos=not args.no_typos,
                dialects=not args.no_dialects,
            )
            for v in variants:
                if v not in emitted:
                    print(v)
                    emitted.add(v)
    except BrokenPipeError:
        # downstream closed (e.g., head) — exit quietly
        pass

if __name__ == "__main__":
    main()
