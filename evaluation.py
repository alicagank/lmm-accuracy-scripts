# Evaluating the Accuracy of Speech to Text Technologies in Turkish-Accented English
# Presented at the 19th Student Conference of Linguistics, İstanbul University, Türkiye, 3 April 2026
# Nuran Orhan & Ali Çağan Kaya

from pathlib import Path
import re
import pandas as pd

REFERENCE_TEXT = (
    "Please call Stella. Ask her to bring these things with her from the store: "
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack "
    "for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. "
    "She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
)

TRANSCRIPTS_DIR = Path("/Users/alicagank/Desktop/orhan-kaya/output")
TRANSCRIPT_GLOB = "*.txt"
OUT_DIR = Path("/Users/alicagank/Desktop/orhan-kaya/metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")

_NUMBER_WORDS = {"three", "five", "six"}

def _num_norm_token(t: str) -> str:
    if not t:
        return t
    if t.isdigit():
        return "num"
    if t in _NUMBER_WORDS:
        return "num"
    return t

def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s

def split_sentences(raw_text: str):
    return [p.strip() for p in SENT_SPLIT_RE.split(raw_text.strip()) if p.strip()]

def _normalize_and_tokenize_with_numbers(s: str):
    s_norm = normalize_text(s)
    toks = [t for t in s_norm.split() if t]
    toks = [_num_norm_token(t) for t in toks]
    return toks

def sentence_token_maps(text: str):
    sents = split_sentences(text)
    tokens, tok2sent, sent_offsets = [], [], []
    offset = 0
    for si, s in enumerate(sents):
        sent_offsets.append(offset)
        ts = _normalize_and_tokenize_with_numbers(s)
        tokens.extend(ts)
        tok2sent.extend([si] * len(ts))
        offset += len(ts)
    return tokens, tok2sent, sent_offsets

REF_TOKENS, REF_TOK2SENT, REF_SENT_OFFSETS = sentence_token_maps(REFERENCE_TEXT)
REF_SENTENCES_RAW = split_sentences(REFERENCE_TEXT)

def _align_sequences(ref_seq, hyp_seq):
    n, m = len(ref_seq), len(hyp_seq)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = i
        bt[i][0] = ("deletion", i-1, 0)
    for j in range(1, m+1):
        dp[0][j] = j
        bt[0][j] = ("insertion", 0, j-1)

    for i in range(1, n+1):
        ri = ref_seq[i-1]
        for j in range(1, m+1):
            hj = hyp_seq[j-1]
            cost = 0 if ri == hj else 1

            best_cost = dp[i-1][j-1] + cost
            best = ("equal" if cost == 0 else "substitution", i-1, j-1)

            cand_cost = dp[i-1][j] + 1
            if cand_cost < best_cost:
                best_cost = cand_cost
                best = ("deletion", i-1, j)

            cand_cost = dp[i][j-1] + 1
            if cand_cost < best_cost:
                best_cost = cand_cost
                best = ("insertion", i, j-1)

            dp[i][j] = best_cost
            bt[i][j] = best

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        op, pi, pj = bt[i][j]
        if op in ("equal", "substitution"):
            ops.append((op, ref_seq[pi], hyp_seq[pj], pi, pj))
            i, j = pi, pj
        elif op == "deletion":
            ops.append((op, ref_seq[pi], None, pi, None))
            i, j = pi, pj
        elif op == "insertion":
            ops.append((op, None, hyp_seq[pj], None, pj))
            i, j = pi, pj
        else:
            raise RuntimeError("Unknown op in backtrace")
    ops.reverse()
    return ops

def word_alignment_details(ref_text: str, hyp_text: str):
    ref_tokens = _normalize_and_tokenize_with_numbers(ref_text)
    hyp_tokens = _normalize_and_tokenize_with_numbers(hyp_text)

    ops = _align_sequences(ref_tokens, hyp_tokens)

    subs = sum(1 for op, *_ in ops if op == "substitution")
    dels = sum(1 for op, *_ in ops if op == "deletion")
    ins  = sum(1 for op, *_ in ops if op == "insertion")
    hits = sum(1 for op, *_ in ops if op == "equal")
    ref_len = len(ref_tokens)
    wer_val = (subs + dels + ins) / max(1, ref_len)

    summary = {"hits": hits, "subs": subs, "dels": dels, "ins": ins, "ref_len": ref_len, "wer": wer_val}
    return summary, ops, ref_tokens, hyp_tokens

def char_alignment_details(ref_text: str, hyp_text: str):
    ref_n = normalize_text(ref_text).replace(" ", "")
    hyp_n = normalize_text(hyp_text).replace(" ", "")
    ref_chars = list(ref_n)
    hyp_chars = list(hyp_n)

    ops = _align_sequences(ref_chars, hyp_chars)
    subs = sum(1 for op, *_ in ops if op == "substitution")
    dels = sum(1 for op, *_ in ops if op == "deletion")
    ins  = sum(1 for op, *_ in ops if op == "insertion")
    hits = sum(1 for op, *_ in ops if op == "equal")
    ref_len = len(ref_chars)
    cer_val = (subs + dels + ins) / max(1, ref_len)

    return {"hits_c": hits, "subs_c": subs, "dels_c": dels, "ins_c": ins,
            "ref_len_c": ref_len, "cer": cer_val}

def per_sentence_metrics(ref_sents_raw, hyp_text: str):
    hyp_sents_raw = split_sentences(hyp_text)
    n = min(len(ref_sents_raw), len(hyp_sents_raw))
    rows = []
    for i in range(n):
        ref_i = ref_sents_raw[i]
        hyp_i = hyp_sents_raw[i]
        w_sum, _, _, _ = word_alignment_details(ref_i, hyp_i)
        c_sum = char_alignment_details(ref_i, hyp_i)
        rows.append({
            "sentence_index": i,
            "ref_sentence": ref_i,
            "hyp_sentence": hyp_i,
            "wer": w_sum["wer"],
            "subs": w_sum["subs"], "dels": w_sum["dels"], "ins": w_sum["ins"], "ref_len_words": w_sum["ref_len"],
            "cer": c_sum["cer"]
        })
    return pd.DataFrame(rows)

def _unified_change_row(speaker, op, r_tok, h_tok, i_ref, j_hyp, ref_tokens, hyp_tokens, HYP_TOK2SENT):
    ref_sent_ix = REF_TOK2SENT[i_ref] if i_ref is not None and i_ref < len(REF_TOK2SENT) else -1
    hyp_sent_ix = HYP_TOK2SENT[j_hyp] if j_hyp is not None and j_hyp < len(HYP_TOK2SENT) else -1

    def ctx(tokens, idx):
        if idx is None or idx < 0 or idx >= len(tokens):
            return "", ""
        prev_ = tokens[idx-1] if idx-1 >= 0 else ""
        next_ = tokens[idx+1] if idx+1 < len(tokens) else ""
        return prev_, next_

    ref_prev, ref_next = ctx(ref_tokens, i_ref)
    hyp_prev, hyp_next = ctx(hyp_tokens, j_hyp)

    return {
        "speaker": speaker,
        "change_type": op,
        "sub_ref": r_tok if op == "substitution" else "",
        "sub_hyp": h_tok if op == "substitution" else "",
        "del_token": r_tok if op == "deletion" else "",
        "ins_token": h_tok if op == "insertion" else "",
        "ref_token_index": i_ref if i_ref is not None else -1,
        "hyp_token_index": j_hyp if j_hyp is not None else -1,
        "ref_sentence_index": ref_sent_ix,
        "hyp_sentence_index": hyp_sent_ix,
        "ref_prev": ref_prev, "ref_next": ref_next,
        "hyp_prev": hyp_prev, "hyp_next": hyp_next,
    }

def _collect_transcripts():
    return [p for p in sorted(TRANSCRIPTS_DIR.glob(TRANSCRIPT_GLOB)) if p.is_file()]

def main():
    transcript_paths = _collect_transcripts()
    if not transcript_paths:
        print(f"No transcripts found at: {TRANSCRIPTS_DIR}/{TRANSCRIPT_GLOB}")
        return

    speaker_rows = []
    global_changes = []

    for p in transcript_paths:
        speaker = p.stem
        hyp_raw = p.read_text(encoding="utf-8", errors="ignore").strip()

        w_sum, ops, ref_tokens, hyp_tokens = word_alignment_details(REFERENCE_TEXT, hyp_raw)
        c_sum = char_alignment_details(REFERENCE_TEXT, hyp_raw)
        speaker_rows.append({
            "speaker": speaker,
            "wer": w_sum["wer"],
            "subs": w_sum["subs"],
            "dels": w_sum["dels"],
            "ins":  w_sum["ins"],
            "ref_len_words": w_sum["ref_len"],
            "cer": c_sum["cer"],
            "subs_c": c_sum["subs_c"],
            "dels_c": c_sum["dels_c"],
            "ins_c":  c_sum["ins_c"],
            "ref_len_chars": c_sum["ref_len_c"],
        })

        _, HYP_TOK2SENT, _ = sentence_token_maps(hyp_raw)

        per_speaker_changes = []
        for op, r_tok, h_tok, i_ref, j_hyp in ops:
            if op == "equal":
                continue
            row = _unified_change_row(
                speaker, op, r_tok, h_tok, i_ref, j_hyp,
                ref_tokens, hyp_tokens, HYP_TOK2SENT
            )
            per_speaker_changes.append(row)
            global_changes.append(row)

        df_changes = pd.DataFrame(per_speaker_changes, columns=[
            "speaker","change_type","sub_ref","sub_hyp","del_token","ins_token",
            "ref_token_index","hyp_token_index","ref_sentence_index","hyp_sentence_index",
            "ref_prev","ref_next","hyp_prev","hyp_next"
        ])
        df_changes.to_csv(OUT_DIR / f"{speaker}_word_changes.csv", index=False, encoding="utf-8")

        df_sent = per_sentence_metrics(REF_SENTENCES_RAW, hyp_raw)
        df_sent.insert(0, "speaker", speaker)
        df_sent.to_csv(OUT_DIR / f"{speaker}_sentence_metrics.csv", index=False, encoding="utf-8")

    pd.DataFrame(speaker_rows).sort_values("speaker").to_csv(
        OUT_DIR / "speaker_summary_metrics.csv", index=False, encoding="utf-8"
    )

    df_all_changes = pd.DataFrame(global_changes, columns=[
        "speaker","change_type","sub_ref","sub_hyp","del_token","ins_token",
        "ref_token_index","hyp_token_index","ref_sentence_index","hyp_sentence_index",
        "ref_prev","ref_next","hyp_prev","hyp_next"
    ])
    df_all_changes.to_csv(OUT_DIR / "all_word_changes.csv", index=False, encoding="utf-8")

    subs = (
        df_all_changes.loc[df_all_changes["change_type"] == "substitution"]
        .groupby(["sub_ref", "sub_hyp"], dropna=False)
        .agg(count=("change_type", "size"), n_speakers=("speaker", "nunique"))
        .reset_index()
        .rename(columns={"sub_ref": "ref_token", "sub_hyp": "hyp_token"})
        .assign(change_type="substitution")
    )
    dels = (
        df_all_changes.loc[df_all_changes["change_type"] == "deletion"]
        .groupby(["del_token"], dropna=False)
        .agg(count=("change_type", "size"), n_speakers=("speaker", "nunique"))
        .reset_index()
        .rename(columns={"del_token": "ref_token"})
        .assign(change_type="deletion", hyp_token="")
    )
    ins = (
        df_all_changes.loc[df_all_changes["change_type"] == "insertion"]
        .groupby(["ins_token"], dropna=False)
        .agg(count=("change_type", "size"), n_speakers=("speaker", "nunique"))
        .reset_index()
        .rename(columns={"ins_token": "hyp_token"})
        .assign(change_type="insertion", ref_token="")
    )
    patterns = pd.concat([subs, dels, ins], ignore_index=True)[
        ["change_type", "ref_token", "hyp_token", "count", "n_speakers"]
    ].sort_values(["change_type", "count"], ascending=[True, False])
    patterns.to_csv(OUT_DIR / "all_word_change_patterns.csv", index=False, encoding="utf-8")

    print("Saved to:", OUT_DIR)
    print("- speaker_summary_metrics.csv")
    print("- <speaker>_sentence_metrics.csv")
    print("- <speaker>_word_changes.csv")
    print("- all_word_changes.csv")
    print("- all_word_change_patterns.csv")

if __name__ == "__main__":
    main()
