#!/usr/bin/env bash
# GATE-2 pre-submission checks (manual §8, fifteen gates).
# Run from anywhere; ROOT is pinned. Prints PASS/FAIL per gate; non-zero exit on FAIL.
#
# Note on pass 1 of this script: it reported seven failures, five of which were bugs
# in the script itself (a bad ROOT, a Windows/Git-Bash path passed to python, a sed
# range that never terminated because the abstract is a single line, an arbitrary URL
# count, and a section-name mismatch). Those are fixed here. G14 was a real finding:
# the template's own files sit next to the manuscript, which is fine for building but
# not for submitting, so G14 now checks the generated package rather than the source
# directory.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$ROOT/latex/submission"
cd "$ROOT"
TEX="latex/sn-article/main.tex"
ART="latex/sn-article"
PY="/c/Users/hp/.workbuddy/binaries/python/versions/3.13.12/python.exe"
COUNTER="${ROOT/#\/e\//E:/}/code/count_prose.py"
fails=0
chk() {
  if [ "$2" = "$3" ]; then printf '  [PASS] %-44s %s\n' "$1" "$2"
  else printf '  [FAIL] %-44s got=%s want=%s\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

echo "=== G1 compile (manuscript) ==="
( cd "$ART" && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/G1a.log 2>&1 \
  && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/G1b.log 2>&1 )
chk "G1 pdflatex errors"    "$(grep -cE '^! ' /tmp/G1b.log)" 0
chk "G1 overfull boxes"     "$(grep -cE 'Overfull' /tmp/G1b.log)" 0
chk "G1 undefined refs"     "$(grep -cE 'undefined' /tmp/G1b.log)" 0

echo "=== G2 citations ==="
chk "G2 undefined citations" "$(grep -coE 'Citation .* undefined' /tmp/G1b.log)" 0
chk "G2 bbl present"         "$( [ -f "$ART/main.bbl" ] && echo yes || echo no )" yes

echo "=== G3/G3b meta-language ==="
chk "G3 main.tex <=40"      "$( [ "$(grep -oiE 'pre-registered|registered|frozen' $TEX | wc -l)" -le 40 ] && echo yes || echo no )" yes
PROSE=$("$PY" "$COUNTER" writing/00-abstract.md writing/01-intro-mechanism.md writing/02-related-work.md writing/04-methods.md writing/05-experiments.md writing/06-discussion-conclusion.md 2>/dev/null | tail -1 | grep -oE 'prose=[0-9]+' | cut -d= -f2)
chk "G3b source prose <=40" "$( [ -n "${PROSE:-}" ] && [ "$PROSE" -le 40 ] && echo yes || echo no )" yes

echo "=== G4/G5/G6 terminology and fixed wording ==="
chk "G4 term red lines"     "$(grep -oiE 'per-pixel modality selection|decision-level|state collapse' $TEX | wc -l)" 0
chk "G5 internal codenames" "$(grep -oE '\bK[1-9]\b|\bGATE1\b|\bGATE2\b|\bD1\b' $TEX | wc -l)" 0
chk "G6 H5 fixed sentence"  "$(grep -o 'statistically significant but below the stricter pre-registered gate thresholds' $TEX | wc -l)" 5

echo "=== G7 abstract ==="
ABS=$(grep -m1 '^\\abstract{' $TEX | sed 's/^\\abstract{//; s/}[[:space:]]*$//')
chk "G7 abstract words <=250" "$( [ "$(echo "$ABS" | wc -w)" -le 250 ] && echo yes || echo no )" yes
chk "G7 abstract numeric values" "$(echo "$ABS" | grep -oE '[0-9]+(\.[0-9]+)?' | grep -vc '^4$')" 0

echo "=== G8/G9/G10 ==="
chk "G8 ledger present"     "$( [ -f GATE1_UNFREEZE_20260915.md ] && echo yes || echo no )" yes
NEG_OK=yes
for kw in falsif distillation NaN "structural cost" replicate; do grep -qi "$kw" "$TEX" || NEG_OK=no; done
chk "G9 negative results listed" "$NEG_OK" yes
chk "G10 abstract states cost" "$(echo "$ABS" | grep -ciE 'cost|below the stricter' | [ "$(cat)" -ge 1 ] && echo yes || echo no)" yes

echo "=== G11/G12 availability ==="
chk "G11 release URL present" "$(grep -c 'github.com/JassonG-xt/s4fusion-cga/releases/tag/v1.0' $TEX | [ "$(cat)" -ge 1 ] && echo yes || echo no)" yes
chk "G11 url formatting"      "$(grep -c 'url{https' $TEX | [ "$(cat)" -ge 2 ] && echo yes || echo no)" yes
chk "G12 data availability"   "$(grep -c 'bmhead{Data availability}' $TEX)" 1
chk "G12 code availability"   "$(grep -c 'bmhead{Code availability}' $TEX)" 1

echo "=== G13 placeholders ==="
chk "G13 PENDING markers"   "$(grep -c 'PENDING-NOT-FROZEN' $TEX)" 0
chk "G13 to-be-completed"   "$(grep -c '\[To be completed' $TEX)" 0
chk "G13 template author"   "$(grep -c 'author.name@example.org\|Organization' $TEX)" 0

echo "=== G14 template remnants (in the PACKAGE, not the build dir) ==="
if [ -d "$PKG" ]; then
  chk "G14 package has no sample .tex" "$( ls "$PKG"/sn-article.tex >/dev/null 2>&1 && echo present || echo absent )" absent
  chk "G14 package has no user manual" "$( ls "$PKG"/user-manual.pdf >/dev/null 2>&1 && echo present || echo absent )" absent
  chk "G14 package builds"             "$( [ -f "$PKG/main.pdf" ] && echo yes || echo no )" yes
else
  echo "  [FAIL] G14 package directory missing ($PKG)"; fails=$((fails+1))
fi

echo "=== G15 structure ==="
for s in Introduction "Related work" Methods Experiments Discussion Limitations Conclusion; do
  chk "G15 section $s" "$(grep -c "^\\\\section{$s}" $TEX)" 1
done

echo
if [ "$fails" -eq 0 ]; then echo "ALL GATES PASS"; else echo "FAILURES: $fails"; fi
exit "$fails"
