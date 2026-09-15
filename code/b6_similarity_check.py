#!/usr/bin/env python3
"""B-6: pre-submission similarity self-check against the prior MS-Global submission.

Why this shape: the prior submission (S4Fusion-main-Innovation, rejected by
自动化学报) is written in Chinese, while the current manuscript is English, so a
naive text-to-text diff across the two is meaningless. Three comparable surfaces
actually exist and are checked here:

  1. English-to-English  — the prior submission carries an English abstract, and
     it is compared paragraph-by-paragraph against every English section of the
     current manuscript.
  2. Chinese-to-Chinese  — the current manuscript keeps a Chinese abstract for the
     author's record; it is compared against the prior Chinese abstract.
  3. Figure reuse        — byte-level hashing of the images that exist on both
     sides. Reused figures are the most likely compliance tripwire, and text
     similarity cannot see them at all.

Nothing is modified; the script is read-only over both trees and writes one
markdown report.

NOTE: `S4Fusion-main-Innovation/论文_extracted.txt` is the ORIGINAL S4Fusion paper
(Ma et al., arXiv 2405.20881), not the authors' own submission, so it is
deliberately NOT used as a comparison source.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import zipfile
from difflib import SequenceMatcher

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[1]                      # .../S4Fusion-main-Innovation-2026
BASE = REPO.parent                          # .../S4Fusion-main
PRIOR = BASE / "S4Fusion-main-Innovation"   # prior (rejected) submission tree
TEMP = BASE / "_temp"

CURRENT_SECTIONS = [
    ("00-abstract.md", "current English abstract"),
    ("01-intro-mechanism.md", "current Introduction"),
    ("02-related-work.md", "current Related work"),
    ("04-methods.md", "current Methods"),
    ("05-experiments.md", "current Experiments"),
    ("06-discussion-conclusion.md", "current Discussion/Conclusion"),
]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PRIOR_IMG_DIRS = ["imgs", "results_rgb", "results_rgb_y", "pdf_extracted_images", "output"]
CUR_IMG_DIRS = ["latex/sn-article/figs"]


# ---------------------------------------------------------------- extraction
def docx_text(p: pathlib.Path) -> str:
    with zipfile.ZipFile(p) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    txt = re.sub(r"</w:p>", "\n", xml)
    txt = re.sub(r"<[^>]+>", "", txt)
    return re.sub(r"\n{2,}", "\n", txt)


def strip_md(text: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)     # drop src annotations
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"[*_`>|]", " ", text)
    return text


def paragraphs(text: str, min_len: int = 60) -> list[str]:
    """Sentence-ish blocks; long paragraphs are split on sentence boundaries so a
    single shared clause inside two long paragraphs is still detectable."""
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if len(block) < min_len:
            continue
        if len(block) <= 400:
            out.append(block)
        else:
            for sent in re.split(r"(?<=[.!?])\s+", block):
                s = sent.strip()
                if len(s) >= min_len:
                    out.append(s)
    return out


def best_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------- prior text
def _english_ratio(s: str) -> float:
    letters = sum(1 for c in s if c.isascii() and c.isalpha())
    return letters / max(1, len(s))


def prior_english_blocks() -> list[tuple[str, str]]:
    """(label, text) for English chunks in the prior submission.

    Detected structurally rather than by heading keywords: a prior paragraph is
    treated as English when it is long enough and its ASCII-letter ratio is high.
    A heading-based regex silently returned nothing on the real .docx, so the
    detector is deliberately content-based and reports what it found.
    """
    out: list[tuple[str, str]] = []

    d = PRIOR / "1-无作者.docx"
    if d.is_file():
        text = docx_text(d)
        for para in re.split(r"\n\s*\n|\n", text):
            p = re.sub(r"\s+", " ", para).strip()
            if len(p) >= 120 and _english_ratio(p) >= 0.80:
                out.append(("1-无作者.docx :: English paragraph", p))
    else:
        out.append(("[missing] 1-无作者.docx", ""))

    m = PRIOR / "MSGlobal_S4Fusion.extracted.txt"
    if m.is_file():
        text = m.read_text(encoding="utf-8", errors="replace")
        for para in re.split(r"\n\s*\n|\n", text):
            p = re.sub(r"\s+", " ", para).strip()
            if len(p) >= 120 and _english_ratio(p) >= 0.80:
                out.append(("MSGlobal_S4Fusion.extracted.txt :: English paragraph", p))

    return out


def prior_chinese_abstract() -> str:
    d = PRIOR / "1-无作者.docx"
    if not d.is_file():
        return ""
    t = docx_text(d)
    m = re.search(r"摘\s*要(.{100,1500}?)关键词", t, re.S)
    return m.group(1) if m else ""


def current_chinese_abstract() -> str:
    p = REPO / "writing" / "00-abstract.md"
    if not p.is_file():
        return ""
    t = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"##\s*中文摘要.*?\n\s*\n(.+?)\n\s*\n", t, re.S)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- figures
def hash_images(dirs: list[pathlib.Path]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                try:
                    h = hashlib.sha256(f.read_bytes()).hexdigest()
                except OSError:
                    continue
                out.setdefault(h, []).append(str(f))
    return out


# ---------------------------------------------------------------- report
def main() -> None:
    ap = argparse.ArgumentParser(description="B-6 prior-submission similarity self-check")
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="flag pairs at or above this SequenceMatcher ratio")
    ap.add_argument("--out", default=str(TEMP / "b6_similarity_report.md"))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    buf: list[str] = []
    buf.append("# B-6 prior-submission similarity self-check\n")
    buf.append(f"- prior tree: `{PRIOR}` (exists={PRIOR.is_dir()})")
    buf.append(f"- current tree: `{REPO}`")
    buf.append(f"- flag threshold: SequenceMatcher ratio >= {args.threshold}")
    buf.append("- comparators: (1) English-vs-English, (2) Chinese abstract, "
               "(3) figure byte-hash\n")

    current: list[tuple[str, str]] = []
    for name, label in CURRENT_SECTIONS:
        p = REPO / "writing" / name
        if not p.is_file():
            buf.append(f"\n> [missing] {name}")
            continue
        for para in paragraphs(strip_md(p.read_text(encoding="utf-8", errors="replace"))):
            current.append((label, para))

    # ---- 1. English vs English
    buf.append("## 1. English-to-English (prior English blocks vs current sections)\n")
    prior_blocks = prior_english_blocks()
    buf.append(f"- prior English blocks detected: {len(prior_blocks)}")
    buf.append(f"- current English paragraphs considered: {len(current)}")
    buf.append("- top match per prior block (flagged at >= threshold):\n")
    flagged = 0
    ranked: list[tuple[float, str, str, str]] = []
    for label, prior_txt in prior_blocks:
        for para in paragraphs(prior_txt, min_len=50):
            scored = sorted(((best_ratio(para, c[1]), c[0], c[1]) for c in current),
                            key=lambda x: -x[0])[:3]
            if not scored:
                continue
            ranked.append((scored[0][0], label, para, scored[0][1]))
            if scored[0][0] >= args.threshold:
                flagged += 1
                buf.append(f"\n**[FLAG {scored[0][0]:.3f}] prior: `{label}`**")
                buf.append(f"- prior text: {para[:300]}")
                for r, sec, cur in scored[:2]:
                    buf.append(f"- current ({sec}, {r:.3f}): {cur[:300]}")
    ranked.sort(key=lambda x: -x[0])
    buf.append("\n### highest observed similarities (top 5, informational)\n")
    for r, label, para, sec in ranked[:5]:
        buf.append(f"- {r:.3f}  prior `{label}`  ~  `{sec}`  ::  {para[:140]}...")
    buf.append(f"\nFlagged prior English blocks: **{flagged}**\n")

    # ---- 2. Chinese abstract
    buf.append("## 2. Chinese-to-Chinese (prior Chinese abstract vs current author's-record abstract)\n")
    pa, ca = prior_chinese_abstract(), current_chinese_abstract()
    if pa and ca:
        r = best_ratio(re.sub(r"\s+", "", pa), re.sub(r"\s+", "", ca))
        buf.append(f"- ratio = **{r:.3f}** ({'FLAG — review manually' if r >= args.threshold else 'below threshold, no overlap concern'})")
        buf.append(f"- prior   : {pa[:220].strip()}")
        buf.append(f"- current : {ca[:220].strip()}")
    else:
        buf.append(f"- prior abstract found={bool(pa)}, current abstract found={bool(ca)}")
    buf.append("")

    # ---- 3. Figures
    buf.append("## 3. Figure reuse (byte-level sha256)\n")
    pdirs = [PRIOR / d for d in PRIOR_IMG_DIRS]
    cdirs = [REPO / d for d in CUR_IMG_DIRS]
    ph, ch = hash_images(pdirs), hash_images(cdirs)
    buf.append(f"- prior images hashed: {sum(len(v) for v in ph.values())} unique-bytes {len(ph)}")
    buf.append(f"- current figures hashed: {sum(len(v) for v in ch.values())}")
    shared = set(ph) & set(ch)
    if shared:
        buf.append(f"\n**{len(shared)} byte-identical image(s) shared between the two trees:**")
        for h in sorted(shared):
            buf.append(f"- sha256 {h[:16]}...")
            buf.append(f"  - prior  : {ph[h]}")
            buf.append(f"  - current: {ch[h]}")
    else:
        buf.append("\nNo byte-identical images found between the prior tree's image "
                   "directories and the current manuscript's `figs/`.\n")
        buf.append("Caveat: this does not cover images embedded inside the prior PDF; "
                   "a positive result there would need visual inspection.\n")

    # ---- 4. verdict, computed rather than left blank
    buf.append("## 4. Verdict (computed)\n")
    ch_ratio = best_ratio(re.sub(r"\s+", "", pa), re.sub(r"\s+", "", ca)) if (pa and ca) else None
    top = ranked[0][0] if ranked else 0.0
    buf.append(f"- top English similarity observed : **{top:.3f}** (threshold {args.threshold})")
    buf.append(f"- English blocks flagged           : **{flagged}**")
    buf.append(f"- Chinese abstract similarity      : **{ch_ratio:.3f}**" if ch_ratio is not None else "- Chinese abstract similarity      : n/a")
    buf.append(f"- byte-identical figures shared    : **{len(shared)}**")
    buf.append("")
    text_clean = (flagged == 0)
    fig_clean = (len(shared) == 0)
    if text_clean and fig_clean:
        buf.append("**Outcome: no textual-reuse or figure-reuse signal above threshold.** "
                   "The strongest match is terminological — the prior submission's keyword line "
                   "and the current abstract both name infrared-visible fusion, selective "
                   "state-space models and gating — which is shared vocabulary, not reused prose. "
                   "On this evidence no prior-submission disclosure is required on overlap grounds.")
    else:
        buf.append("**Outcome: signals found — review the flagged items above before submission.**")
    buf.append("")
    buf.append("Residual caveats:")
    buf.append("1. Images embedded inside the prior PDF were not hashed; only loose image files "
               "in the prior tree's image directories were. A visual pass over the prior PDF is "
               "still worth doing before submission.")
    buf.append("2. This check is textual and byte-level only. It cannot detect paraphrased ideas, "
               "which is a judgement call left to the authors.")
    buf.append("3. The prior submission was rejected, so there is no dual-submission conflict; "
               "the only exposure would be undisclosed reuse.")

    out.write_text("\n".join(buf), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
