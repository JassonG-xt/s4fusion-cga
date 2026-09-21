#!/usr/bin/env python3
"""Last-round minor revision (2026-09-21) applied to latex/sn-article/main.tex.

Content-anchored, not line-numbered: every entry is one exact substring that
must occur EXACTLY ONCE in the file. A duplicate or a miss aborts the whole run
before anything is written. After the replacements the paired-structure counts
(\\begin{env} for every environment in the manuscript) are compared before and
after; any change aborts too.

Usage:
    python apply_minor_revision_20260921.py --check      # report only
    python apply_minor_revision_20260921.py              # apply
    python apply_minor_revision_20260921.py <path.tex>   # apply to a copy
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "latex" / "sn-article" / "main.tex"

ENVS = ("figure", "figure*", "table", "table*", "tabular", "enumerate",
        "itemize", "equation", "align", "algorithm", "algorithmic")

R: list[tuple[str, str]] = []


def add(old: str, new: str) -> None:
    R.append((old, new))


# =========================================================== M1/M2/S2 == §3.4
add(
    r"""over its ground-truth box, computed from the input pair alone; objects strictly above the 66th percentile of conflict scores form the high-conflict subset. The comparison is per-object and paired: recovered objects (missed by the baseline, detected with CGA) and lost objects (the converse) form the discordant pairs, assessed by an exact binomial (McNemar-style) test; the pass criterion is $p < 0.05$ together with recovered $-$ lost $\geq 3$. Because""",
    r"""over its ground-truth box, computed from the input pair alone; objects strictly above the 66th percentile of conflict scores form the high-conflict subset. The comparison is paired: recovered objects (missed by the baseline, detected with CGA) and lost objects (the converse) form the discordant pairs, and the pass criterion is $p < 0.05$ together with recovered $-$ lost $\geq 3$, with $p$ read from the test appropriate to the analysis layer in which it is evaluated. Because""",
)

add(
    r"""so the analysis is layered explicitly, and the layer that carries a claim is fixed in advance:""",
    r"""so the analysis is layered explicitly. The original protocol specified an object-level paired endpoint, and that endpoint and its criterion were fixed before the arbitration re-test. After the dependence structure was identified, a post-hoc image-clustered re-analysis was added as a robustness analysis and is reported separately from the pre-specified endpoint. Because multiple objects share the same image-level fused output and detection error, this revision reads the post-hoc image-clustered analysis as the primary interpretation of dependence-aware evidence, and retains the original object-level exact binomial as a sensitivity analysis that is not used alone to support replication. The three layers are named once here and used with those names throughout:""",
)

add(
    r"""\item \textbf{Confirmatory (cluster-aware).} The image is the unit of observation. For each seed we report (i) an image-level net-sign test on the per-image difference in recovered-minus-lost high-conflict objects, (ii) an exact cluster sign-flip test on the image scores, and (iii) a percentile bootstrap over images ($B = 2000$, fixed seed) for the recall difference. A seed counts as supporting the hypothesis only if the image-level test rejects at $0.05$ \emph{and} the bootstrap interval excludes zero; the primary criterion of $p<0.05$ with recovered $-$ lost $\geq 3$ is then read \emph{within} that layer.""",
    r"""\item \textbf{Post-hoc cluster-aware robustness analysis} --- the reading the detection verdict rests on, on the fixed 300-image test set (Table~\ref{tab:cluster}). The image is the unit of observation. For each seed we report (i) an image-level net-sign test on the per-image difference in recovered-minus-lost high-conflict objects, (ii) an exact cluster sign-flip test on the image scores, and (iii) a percentile bootstrap for the recall difference. The bootstrap resamples the $181$ images that carry the subset with replacement ($B = 2000$, fixed seed), but holds the denominator fixed at the $878$ observed high-conflict objects; its estimand is therefore the \emph{object-weighted} recall difference of the two fixed arms over the fixed image set, not an image-average recall difference and not a population quantity. Fixing the denominator is deliberate: the arms are compared on one and the same already-annotated object set, so letting the resampled object count enter the denominator would inject a between-image object-count variability that this endpoint does not have. A seed counts as supporting the hypothesis only if the image-level test rejects at $0.05$ \emph{and} the bootstrap interval excludes zero; the criterion of $p<0.05$ with recovered $-$ lost $\geq 3$ is then read \emph{within} that layer. This analysis was added after the dependence structure was identified --- that is, after the re-test outcome it describes was already in hand --- so it is a post-hoc robustness analysis of stored predictions and not a pre-registered test; Table~\ref{tab:timeline} records its date on that footing.""",
)

add(
    r"""\item \textbf{Sensitivity (object-level).} The exact binomial (McNemar-style) test is retained and reported beside every cluster-aware result, because the protocol fixed it before the cluster structure was modelled. It is a sensitivity analysis: it can show that a difference exists in aggregate, but it cannot support a replication claim on its own, and the paper does not use it as one.""",
    r"""\item \textbf{Pre-specified sensitivity endpoint (object level).} The exact binomial (McNemar-style) test on the paired recovered/lost objects is retained and reported beside every cluster-aware result, because the protocol fixed it before the cluster structure was modelled. It is a sensitivity analysis: it can show that a difference exists in aggregate, but it cannot support a replication claim on its own, and the paper does not use it as one. The two object-level tables are read only in this capacity --- Table~\ref{tab:h5} under the single-baseline convention and Table~\ref{tab:h5e32} under the budget-matched one --- as sensitivity material and as a record of the original protocol's own endpoint. Neither carries the verdict; that is Table~\ref{tab:cluster}.""",
)

add(
    r"""\item \textbf{Descriptive (pooled).} The pooled across-seed statistic""",
    r"""\item \textbf{Descriptive pooled analysis.} The pooled across-seed statistic""",
)

# ==================================================================== S1 == §3.2
add(
    r"""and apply that commitment through a zero-initialized gate only in conflict regions.""",
    r"""and apply that commitment through a zero-initialized gate. The gate is designed to act selectively through a learned support map, and whether this conflict-specific localization is necessary is not assumed but tested: the uniform and shuffled-map controls of Section~\ref{sec:exp:controls} do not support it, so the localization is reported here as a design motivation under test rather than as an established property.""",
)

# ============================================================== M1/M2 == §4.1
add(
    r"""image-clustered bootstrap and permutation tests as the confirmatory layer for the detection endpoints, and exact binomial (McNemar-style) tests on recovered/lost objects retained as a sensitivity analysis.""",
    r"""image-clustered bootstrap and permutation tests as the post-hoc cluster-aware robustness analysis of the detection endpoints, and exact binomial (McNemar-style) tests on recovered/lost objects retained as the pre-specified sensitivity endpoint. The three layers are named in Section~\ref{sec:methods:design} and used with those names everywhere: ``post-hoc cluster-aware robustness analysis'' for the image-clustered reading, ``pre-specified sensitivity endpoint'' for the object-level McNemar-style test, and ``descriptive pooled analysis'' for the across-seed aggregates.""",
)

# ============================================================== M2 == Table 4/5
add(
    r"""\textbf{Descriptive, not confirmatory}: the three seeds share the same 300 test images""",
    r"""\textbf{Descriptive pooled analysis, not a replication test}: the three seeds share the same 300 test images""",
)

# ============================================================== M2 == Table 6
add(
    r"""\caption{Primary detection analysis, budget-matched convention, with the image as the unit of observation. ``obj.'' is the object-level discordant count and its exact binomial $p$ (sensitivity analysis); ``img.'' is the image-level net-sign test; ``flip'' is the exact cluster sign-flip test on the per-image scores; the interval is a percentile bootstrap over images ($B = 2000$, fixed seed) for the high-conflict recall difference. A seed is counted as supporting the hypothesis only if the image-level test and the bootstrap interval agree.""",
    r"""\caption{Post-hoc cluster-aware robustness analysis of the detection endpoint --- the table from which the detection verdict is read --- under the budget-matched convention, with the image as the unit of observation. ``obj.'' is the object-level discordant count with its exact binomial $p$, i.e.\ the pre-specified sensitivity endpoint and not the basis of the verdict; ``img.'' is the image-level net-sign test; ``flip'' is the exact cluster sign-flip test on the per-image scores; the interval is a percentile bootstrap that resamples the $181$ images carrying the subset with replacement while holding the denominator fixed at the $878$ observed high-conflict objects, so its estimand is the object-weighted high-conflict recall difference of the two fixed arms on the fixed image set ($B = 2000$, fixed seed). The analysis is post-hoc: it was added on 20 September 2026, after the dependence structure was identified and after the re-test outcome it describes had been obtained, so it is a robustness analysis of stored predictions and not a pre-registered test (Table~\ref{tab:timeline}). A seed is counted as supporting the hypothesis only if the image-level test and the bootstrap interval agree.""",
)

# ============================================================== M2 == Table 8
add(
    r"""\caption{H5 primary endpoint under the \emph{budget-matched} base convention: CGA vs.\ each seed's own 32-epoch baseline (B0\_e32);""",
    r"""\caption{H5 endpoint in its object-level, pre-specified sensitivity form under the \emph{budget-matched} base convention: CGA vs.\ each seed's own 32-epoch baseline (B0\_e32); read as sensitivity and as a record of the original protocol's own endpoint, never as the basis of the verdict, which is Table~\ref{tab:cluster};""",
)

add(
    r"""\caption{Sensitivity analysis, \emph{single-baseline} convention:""",
    r"""\caption{Pre-specified sensitivity endpoint, \emph{single-baseline} convention:""",
)

add(
    r"""Both are retained as the sensitivity layer of Section~\ref{sec:methods:design}: they are informative in aggregate, and neither is used here as the basis of a replication claim.""",
    r"""Both are retained as the pre-specified sensitivity layer of Section~\ref{sec:methods:design}: they are informative in aggregate, and neither is used here as the basis of a replication claim --- that is read from Table~\ref{tab:cluster}.""",
)

# ============================================================== M1/M2 == §4.4
add(
    r"""\paragraph{H5 --- CGA conflict-gated arbitration improves high-conflict detection --- criterion met on one of three seeds, and only that one seed survives the cluster-aware analysis.}""",
    r"""\paragraph{H5 --- CGA conflict-gated arbitration improves high-conflict detection --- criterion met on one of three seeds, and only that one seed survives the post-hoc cluster-aware analysis.}""",
)

add(
    r"""The confirmatory analysis treats the \emph{image} as the unit of observation, as Section~\ref{sec:methods:design} specifies,""",
    r"""The post-hoc cluster-aware analysis treats the \emph{image} as the unit of observation, as Section~\ref{sec:methods:design} specifies,""",
)

add(
    r"""The cluster-aware result is unambiguous and it is weaker than the aggregate one: \emph{one} seed of three keeps a positive, cluster-robust effect.""",
    r"""The post-hoc cluster-aware result is a marginal image-level result, and it is weaker than the aggregate one: \emph{one} seed of three keeps a positive result under that analysis.""",
)

# ============================================================== S3 == §4.4
add(
    r"""Both checks are object-level, so the within-image dependence Table~\ref{tab:cluster} models applies to them too, and both are reported as sensitivity rather than confirmatory.""",
    r"""Both checks are object-level, so the within-image dependence Table~\ref{tab:cluster} models applies to them too, and both are reported as sensitivity material rather than as the primary reading.""",
)

# ============================================================== S3 == §4.5
add(
    r"""This measurement comes from the seed-42 arm and is not part of the confirmatory layer;""",
    r"""This measurement comes from the seed-42 arm and is not part of the post-hoc cluster-aware robustness analysis either;""",
)

# ============================================================== S4 == Table 11
add(
    r"""``img.'' the image-level net-sign test over the $181$ images carrying the subset; ``mAP50'' is the evaluator's all-object mAP50 (the per-image statistic of Section~\ref{sec:exp:verdicts} is given in the text) and ``prec.''/``FP'' are all-object precision and false positives per image, computed with one estimator for all three arms.""",
    r"""``img.'' the image-level net-sign test over the $181$ images carrying the subset; ``hc recall'' is the fraction of the $878$ high-conflict objects detected; ``mAP50'' is the evaluator's all-object mAP50 over the $300$ images (the per-image statistic of Section~\ref{sec:exp:verdicts} is given in the text); and ``prec.''/``FP'' are all-object precision and false positives \emph{per image}, all four computed with one estimator for all three arms.""",
)

# ============================================================== M4 == §4.9
add(
    r"""We state it that way because it is what is true, and the detector environment is recorded as the weakest link rather than left implicit.""",
    r"""We state it that way because it is what is true, and the detector environment is recorded as the weakest link rather than left implicit. The version of record for these artifacts is the final review release \texttt{v1.1-final-review-20260921} (21 September 2026), which supersedes the earlier \texttt{v1.0} snapshot and is the version the release manifest hashes; it covers the analyses added after the \texttt{v1.0} freeze --- the budget-matched baselines, the extra feature-space-arm seeds and inference-time controls, the selector-distribution summaries and the cluster-aware re-analysis --- while \texttt{v1.0} remains available and covers the fusion-quality evidence and the content-control detection tables only.""",
)

add(
    r"""the cluster-aware detection analysis of Table~\ref{tab:cluster} by \texttt{cluster\_audit\_final.py}, which prints the object$\to$image$\to$seed mapping, the object- and image-level tests and the bootstrap intervals, and which asserts the frozen population ($2583$ objects, $878$ high-conflict objects over $181$ images) so that it fails loudly rather than silently analysing a different set;""",
    r"""the cluster-aware detection analysis of Table~\ref{tab:cluster} by \texttt{cluster\_audit\_final.py}, which prints the object$\to$image$\to$seed mapping, the object- and image-level tests and the bootstrap intervals --- resampling the $181$ images with replacement while holding the denominator fixed at the $878$ high-conflict objects, the convention defined in Section~\ref{sec:methods:design} --- and which asserts the frozen population ($2583$ objects, $878$ high-conflict objects over $181$ images) so that it fails loudly rather than silently analysing a different set;""",
)

# ============================================================== M4 == Table 13
add(
    r"""Cluster-aware re-analysis (this revision) & 2026-09-20 & --- (post-hoc analysis of stored predictions) \\""",
    r"""Cluster-aware re-analysis (this revision) & 2026-09-20 & --- (post-hoc analysis of stored predictions) \\
Final review release \texttt{v1.1-final-review-20260921} & 2026-09-21 & --- (a snapshot of the released repository; supersedes \texttt{v1.0}) \\""",
)

# ============================================================== M1 == Abstract
add(
    r"""and a cluster-aware re-analysis that treats the image as the unit of observation leaves that one seed alone, so what survives is a seed-specific, non-replicated effect rather than a replicated gain;""",
    r"""and a post-hoc cluster-aware re-analysis that treats the image as the unit of observation leaves the same single seed positive, so what survives is a seed-specific, non-replicated effect rather than a replicated gain;""",
)

# ============================================================== M1 == §1 intro
add(
    r"""the budget-matched convention clears it on one seed of three, and re-analysing the endpoint with the \emph{image} as the unit leaves that one seed alone,""",
    r"""the budget-matched convention clears it on one seed of three, and a post-hoc re-analysis that takes the \emph{image} as the unit leaves that one seed alone,""",
)

add(
    r"""replication counted per seed, and a cluster-aware re-analysis that treats the image rather than the object as the unit of observation""",
    r"""replication counted per seed, and a post-hoc cluster-aware re-analysis that treats the image rather than the object as the unit of observation""",
)

# ============================================================== M1 == §5/§6/§7
add(
    r"""and the confirmatory cluster-aware analysis leaves that same single seed, on a marginal image-level test (Table~\ref{tab:cluster}).""",
    r"""and the post-hoc cluster-aware re-analysis leaves that same single seed positive, on a marginal image-level test (Table~\ref{tab:cluster}).""",
)

add(
    r"""and under the confirmatory cluster-aware analysis the same single seed survives, on an image-level test that is itself marginal.""",
    r"""and under the post-hoc cluster-aware re-analysis the same single seed survives, on an image-level test that is itself marginal.""",
)

add(
    r"""a cluster-aware re-analysis leaves that same single seed on a marginal image-level test,""",
    r"""a post-hoc cluster-aware re-analysis leaves the same single seed positive on a marginal image-level test,""",
)

# ================================================================== M3 == declarations
DECLARATIONS = r"""
\bmhead{Data availability}The M3FD benchmark used in this study is publicly available from the TarDAL project \cite{liu2022tardal}; the source datasets are not redistributed here and must be obtained from their original providers. No new data were created.

The public release accompanying this article is the final review release, tagged \texttt{v1.1-final-review-20260921} at \url{https://github.com/JassonG-xt/s4fusion-cga/releases/tag/v1.1-final-review-20260921} and frozen on \textbf{21 September 2026}. It is the version intended to cover every number reported here: the per-image fusion-quality CSVs behind the main comparison, the per-seed and pooled cross-seed tables and the colour audit; the per-image and per-object detection tables behind the learned/uniform/shuffled content-control analysis; the budget-matched 32-epoch baselines and their per-seed comparisons; the additional feature-space-arm seeds and their inference-time controls; the selector-distribution summaries; the object$\to$image$\to$seed mapping; the manifest and checkpoint hashes; and the cluster-aware audit script whose output reproduces Tables~\ref{tab:cluster} and~\ref{tab:controls} from the stored per-object detection vectors. It supersedes the earlier tag \texttt{v1.0} (15 September 2026), which remains available and should be read as covering the fusion-quality evidence and the content-control detection tables only: \texttt{v1.0} was frozen before the budget-matched runs (16 September 2026), the additional feature-space-arm seeds and the inference-time controls (19 September 2026) and the cluster-aware re-analysis (20 September 2026) were produced, and it is therefore not a snapshot of every number in this article.

\bmhead{Code availability}The scripts named in Section~\ref{sec:exp:repro} are released in the same final review release (\texttt{v1.1-final-review-20260921}) described under Data availability, under the licences recorded in its \texttt{LICENSE} file (MIT for code, CC-BY-4.0 for documentation, ledgers and tabular results). One scope limit is stated rather than implied: the detector-side library stack is inherited from the DCEvo release and is \emph{not} pinned in the released artifacts, so the detection endpoint is reproducible by recomputing the statistics from the released intermediate predictions, not by re-running the detector and expecting bit-identical output. The fusion-side environment \emph{is} pinned (\texttt{requirements-brss.txt}, torch 2.1.0+cu118, Python 3.11), and the detector-side environment as observed is recorded in the release's environment notes together with the commands that regenerate each table and figure.

The protocol records are archived rather than summarised. The frozen results list is \texttt{GATE1\_FREEZE.md} (13 September 2026) and the pre-registered decision rules are \texttt{GATE1\_PREREG\_B3\_20260914.md} (14 September 2026), both in the release's \texttt{ledgers/} area; the wording-and-evidence amendment of 15 September 2026 (\texttt{GATE1\_PREREG\_B3\_V2\_20260915.md}), the matched-budget record of 15--16 September 2026 (\texttt{GATE1\_UNFREEZE\_20260915.md}) and the amendment of 14 September 2026 (\texttt{GATE1\_AMEND\_20260914.md}) are retained in the author's working repository and are available to the editor and the reviewers on request. What those records do and do not fix is set out item by item in Table~\ref{tab:timeline}: the criterion, the gate thresholds and the reporting rules for the arbitration re-test were fixed before that re-test was run, while the architecture, the training recipe, the seed-42 checkpoint and the choice of testbed pre-date them, and the cluster-aware re-analysis post-dates the outcome it describes. This study is therefore a later-stage prospective re-test with a staged lock, not a full pre-registration, and the ordering claimed here --- each criterion fixed before the outcome it governs --- is checkable against the archived files' dates and their internal record of the runs.

\bmhead{Author contribution}Xiaotong Gao: conceptualization, method design, experiments, analysis, writing, final verification of every reported number and statement, and responsibility for the content and integrity of the article.

\bmhead{Funding}No funding was received for this work.

\bmhead{Competing interests}The author declares no competing interests.

\bmhead{Ethics approval}Not applicable; this study uses a publicly released benchmark and contains no human participants or personal data.

\bmhead{Use of AI tools}In preparing this manuscript, the author used AI-based assistance for six tasks: literature verification (cross-checking reference identities, the DOI metadata of the two closest published neighbours, and the closest-neighbour characterizations cited in Section~\ref{sec:related}, where the concurrent preprint was verified at abstract level only and Table~\ref{tab:positioning} records that difference); drafting assistance for the research design and protocol text, including the layering of the primary criterion, the stricter gates C1--C3, the decision rules and the wording rules for reporting each outcome tier; the decision to reframe the study as a controlled falsification and mechanism audit after the arbitration arm failed to replicate; statistical re-analysis scripting, including the cluster-aware object$\to$image$\to$seed analysis, the non-inferiority boundary report and the effect-size and interval computations; implementation assistance on analysis and figure code; and language polishing.

The design-related participation is disclosed explicitly because it is the part most easily under-reported. Three boundaries apply to it. First, every criterion, threshold and decision rule was reviewed and confirmed by the author and frozen before the corresponding experiment was run, and the frozen records are those named under Code availability; the post-hoc cluster-aware re-analysis of Section~\ref{sec:methods:design} is labelled as post-hoc precisely because it was not part of that frozen set. Second, the \emph{constraints} that produced the negative findings reported here were fixed by those records rather than chosen at write-up time: the per-seed replication requirement, the inclusive $-1\%$ non-inferiority margin, the requirement to report both base conventions as non-comparable, and the requirement to report failed gates were all registered beforehand. Third, no AI system retrained a model, ran the detector, or generated a number that appears in this article; all experimental numbers were produced by the frozen training and evaluation pipeline, and every re-analysis reported in the revision was computed by a script archived with the submission and re-run against the stored intermediate results. Every AI-assisted statement and number was verified against those records by the author, who reviewed the final text in full, approved each numeric claim and each scoping statement, and takes full responsibility for the content and integrity of the article.
"""

add(r"""\backmatter

%%=============================================================
%% Bibliography (sn-mathphys-num, numeric citations)""",
    r"""\backmatter
""" + DECLARATIONS + r"""
%%=============================================================
%% Bibliography (sn-mathphys-num, numeric citations)""")


# ------------------------------------------------------------------- engine
def env_counts(text: str) -> dict[str, int]:
    return {e: text.count("\\begin{" + e + "}") for e in ENVS}


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    check_only = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    path = Path(argv[0]) if argv else MAIN
    text = path.read_text(encoding="utf-8")

    bad = []
    for old, _new in R:
        n = text.count(old)
        if n != 1:
            bad.append((n, old[:110]))
    if bad:
        print("ANCHOR FAILURES (count, anchor):")
        for n, a in bad:
            print(f"  {n}x  {a!r}")
        print(f"\n{len(bad)} of {len(R)} anchors failed -> nothing written")
        return 1
    print(f"all {len(R)} anchors unique: OK")

    if check_only:
        return 0

    before = env_counts(text)
    out = text
    for old, new in R:
        out = out.replace(old, new, 1)
    after = env_counts(out)
    diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    if diff:
        print("ENVIRONMENT COUNTS CHANGED -> nothing written")
        for k, (b, a) in diff.items():
            print(f"  {k}: {b} -> {a}")
        return 2

    if path == MAIN:
        snap = MAIN.with_suffix(".tex.pre-20260921")
        if not snap.exists():
            shutil.copy2(MAIN, snap)
            print("snapshot written:", snap.name)
    path.write_text(out, encoding="utf-8", newline="\n")
    print(f"applied {len(R)} replacements -> {path}")
    print("environment counts preserved:", before == after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
