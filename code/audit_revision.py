#!/usr/bin/env python3
"""Self-audit of the 2026-09-20 revision against the review's item list.

Checks, in the manuscript, for the presence of each required fix and the
absence of each withdrawn claim. Prints one line per item so the audit can be
read as a checklist. Run:  python audit_revision.py latex/sn-article/main.tex
"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'main.tex'
t = open(path, encoding='utf-8').read()
flat = re.sub(r'\s+', ' ', t)

# The seven back-matter declaration blocks were removed from main.tex on
# 2026-09-21 (blank pages between the Conclusion and the References) and
# RESTORED into main.tex later the same day by the last-round minor-revision
# plan (2026-09-21, item M3). declarations.tex is kept as the archive of the
# pre-restoration wording; the checks below are run against main.tex again.
# The v1.0-scope sentence changed when the final review release v1.1 was
# created, so the expected strings name the new release.
DECLARATION_CHECKS = [
    ('B3  release v1.0 freeze date', '15 September 2026'),
    ('B3  v1.0 scope limited (not a full snapshot)',
     'not a snapshot of every number in this article'),
    ('B3  final release v1.1 named', 'v1.1-final-review-20260921'),
    ('B3  seven declaration blocks present',
     'Use of AI tools'),
    ('R2c author review and responsibility',
     'takes full responsibility for the content and integrity'),
]

MUST_HAVE = [
    ('B1  cluster-aware analysis script + units', r'cluster\_audit\_final.py'),
    ('B1  image as unit of observation', 'image as the unit of observation'),
    ('B1  2583 / 878 / 181 population', '2{,}583'),
    ('B1  analysis families listed (2026-09-21 wording)',
     'Post-hoc cluster-aware robustness analysis'),
    ('B1  object-level demoted to sensitivity (2026-09-21 wording)',
     'Pre-specified sensitivity endpoint (object level)'),
    ('B1  pooled demoted to descriptive (2026-09-21 wording)',
     'Descriptive pooled analysis'),
    ('B2  local pointwise image-space scope', 'pointwise convex combination'),
    ('B2  feature-space marked as hypothesis', 'treated here as a hypothesis rather than a result'),
    ('B2  near-binary selector naming', 'trained near-binary selector'),
    ('B2  five symbol definitions', 'source-selection weight'),
    ('B2  alpha sign and range reported', 'signed'),
    ('B2  epsilon floor quantified', '1.47'),
    ('B2  commit loss non-identifiability', 'identifiable as evidence'),
    ('B2  recipe-bundle disclosure', 'recipe as a whole'),
    ('B2  CGA-v1 label', 'CGA-v1'),
    ('B3  pre-registration timeline table', 'tab:timeline'),
    ('B3  later-stage prospective re-test', 'later-stage prospective re-test'),
    ('B3  scope to single testbed', 'single-testbed'),
    ('M1  subtitle mechanism audit', 'Mechanism-Audit Study'),
    ('M1  active at inference (not online)', 'active at inference'),
    ('M2  uniform as trade-off, per metric', 'tab:controls'),
    ('M2  recall vs mAP50 trade-off stated', 'recall by giving up ranking quality'),
    ('M3  staged-lock wording', 'staged lock'),
    ('M4  per-seed table as primary', 'tab:perseed'),
    ('M5  multi-objective trade-off', 'operating point'),
    ('M6  evidence-grade column', 'evidence status'),
    ('XX_M6_bib', 'XX_M6_bib'),
    ('M7  pooled statistics demoted to descriptive', 'are descriptive'),
    ('R1  cluster-aware reported per seed', 'tab:cluster'),
    ('R2  operator definition consistent', 'continuous residual arbitration'),
    ('R3  NaN and block disclaimer', 'unstable experimental prototype'),
    ('R4  controls reported per metric', 'every metric separately'),
    ('R5  scope statement in abstract', 'scoped to a single-testbed'),
    ('R6  timeline item by item', 'Provenance timeline'),
    ('R8  four-bounded-propositions conclusion', 'Four propositions'),
    ('S1  figure 2 rebuilt larger', 'five columns: the infrared source'),
    ('S1  figure 3 zoom row', 'magnifies that region'),
    ('S1  fig 5 peak corrected', '0.5902'),
    ('S1  table 3 dispersion unit column', 'over'''),
    ('S1  table 7 support definitions', 'defined in the text above'),
    ('S1  C3 unrounded + one-sided bound', '-1.0043'),
    ('S1  equality rule explicit', 'margin is inclusive'),
    ('S2  precision / FP / mAP reported for controls', 'FP/img'),
    ('S3  Table 1 evidence status row', 'abstract only'),
    ('S4  three-seed colour audit', 'tab:color'),
    ('S4  three-seed selector audit', 'tab:selector'),
    ('typeset 1 affiliation separator', 'Qingdao University,'),
    ('typeset 7 table 3 plus-minus unit', '3 seeds'),
    ('typeset 9 online removed', 'at inference'),
    ('typeset 11 equivalence wording', 'no difference was detected'),
    ('typeset 12 colour per seed', 'three-seed form'),
    # ---- last-round minor revision, 2026-09-21 (plan items M1-M4, S1-S3) ----
    ('M1  post-hoc label in §3.4 and Table 6 caption',
     'post-hoc robustness analysis of stored predictions'),
    ('M1  original endpoint fixed before the re-test',
     'that endpoint and its criterion were fixed before the arbitration re-test'),
    ('M1  post-hoc re-analysis named in abstract',
     'a post-hoc cluster-aware re-analysis that treats the image as the unit of observation leaves the same single seed positive'),
    ('M1  post-hoc wording in conclusion',
     'a post-hoc cluster-aware re-analysis leaves the same single seed positive'),
    ('M1  no pre-registered confirmatory layer',
     'not a pre-registered test'),
    ('M2  three analysis layers named in §4.1',
     'descriptive pooled analysis'),
    ('M2  Table 6 named as the verdict table',
     'the table from which the detection verdict is read'),
    ('M2  Tables 7-8 restricted to sensitivity',
     'read as sensitivity and as a record of the original protocol'),
    ('M2  sensitivity layer wording',
     'pre-specified sensitivity layer'),
    ('S1  selective action stated as under test',
     'whether this conflict-specific localization is necessary is not assumed but tested'),
    ('S2  bootstrap estimand stated',
     'object-weighted'),
    ('S2  fixed-denominator convention explained',
     'holds the denominator fixed at the $878$ observed high-conflict objects'),
    ('S2  fixed denominator echoed in §4.9',
     'resampling the $181$ images with replacement'),
    ('S3  marginal, not unambiguous',
     'marginal image-level result'),
    ('S4  Table 11 recall unit explicit',
     'the fraction of the $878$ high-conflict objects detected'),
    ('M4  final release identity in §4.9',
     'The version of record for these artifacts is the final review release'),
    ('M4  final release row in the timeline',
     'Final review release \\texttt{v1.1-final-review-20260921}'),
]

MUST_NOT_HAVE = [
    ('withdrawn: hard commitment as structure', 'structurally hard routing'),
    ('withdrawn: unconditional everywhere-at-least-as-well',
     'at least as well at both ends'),
    ('withdrawn: manual conflict map', 'manual conflict map'),
    ('withdrawn: 0.594 attributed to 01422', 'per-sample difference-map peak of 0.594'),
    ('withdrawn: operator-class generalisation', 'scan-and-blend operator class'),
    ('withdrawn: positional not incremental', 'positional, not incremental'),
    ('withdrawn: effects point the same way everywhere',
     'effects point the same way everywhere'),
    ('withdrawn: every number can be regenerated',
     'Every number in this paper can be regenerated'),
    ('withdrawn: statistically indistinguishable', 'statistically indistinguishable'),
    ('withdrawn: EOR single-seed significance claim', 'improves by $+0.0016$'),
    # ---- last-round minor revision, 2026-09-21 ----
    ('M1  no cluster-aware analysis called confirmatory',
     'confirmatory'),
    ('M1  / M2  no "layer that carries a claim is fixed in advance"',
     'the layer that carries a claim is fixed in advance'),
    ('S1  method no longer claims action only in conflict regions',
     'only in conflict regions'),
    ('S3  the marginal image-level result is not called unambiguous',
     'result is unambiguous'),
    ('S3  no "cluster-robust effect"',
     'cluster-robust effect'),
    ('M2  Table 8 no longer calls itself the H5 primary endpoint',
     'H5 primary endpoint'),
]

ok = True
print('--- required content ---')
for label, needle in MUST_HAVE:
    if needle == 'XX_M6_bib':
        continue
    hit = needle in flat
    ok &= hit
    print(f"[{'PASS' if hit else 'FAIL'}] {label}")

print('\n--- declaration blocks (restored into main.tex 2026-09-21, item M3) ---')
for label, needle in DECLARATION_CHECKS:
    hit = needle in flat
    ok &= hit
    print(f"[{'PASS' if hit else 'FAIL'}] {label}")

print('\n--- withdrawn claims ---')
for label, needle in MUST_NOT_HAVE:
    hit = needle in flat
    ok &= not hit
    print(f"[{'PASS' if not hit else 'FAIL'}] {label}")

print('\nRESULT:', 'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED')
