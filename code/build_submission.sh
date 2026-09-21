#!/usr/bin/env bash
# Build the submission package: only what the editors need to compile and read the
# manuscript, and nothing from the journal template's own sample files.
#
# The template ships sn-article.tex, its compiled PDF, fig.eps and the other .bst
# variants for demonstration. They build alongside the manuscript without harm but
# must not be submitted. empty.eps IS kept: the class loads it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/latex/sn-article"
PKG="$ROOT/latex/submission"

mkdir -p "$PKG/figs"
for f in main.tex refs.bib main.bbl sn-jnl.cls sn-mathphys-num.bst empty.eps; do
  cp "$SRC/$f" "$PKG/"
done
cp "$SRC"/figs/* "$PKG/figs/"

( cd "$PKG" && for _ in 1 2 3; do
    pdflatex -interaction=nonstopmode -halt-on-error main.tex >>/tmp/pkg1.log 2>&1
  done )

# Three passes, not two. A \ref inside an unnumbered sectioning command (the
# "already reported in Section~\ref{...}" paragraph headings) resolves only on
# the pass AFTER the one that writes the .aux, and hyperref's outline is built
# from the .out written by the previous pass. Two passes leave one bookmark
# rendering as "Section ??". Verified 2026-09-20: 3 passes -> 66 outline
# entries, 0 containing "??"; 2 passes -> 1 containing "??".

# Drop build byproducts from the package: .aux/.out/.log are noise for an editor.
# main.bbl is deliberately KEPT so the manuscript compiles without running bibtex.
rm -f "$PKG/main.aux" "$PKG/main.out" "$PKG/main.log" "$PKG/main.blg"

echo "package built at $PKG"
printf 'errors:   %s\n' "$(grep -cE '^! ' /tmp/pkg1.log)"
printf 'overfull: %s\n' "$(grep -cE 'Overfull' /tmp/pkg1.log)"
printf 'pages:    %s\n' "$(grep -oE 'main.pdf \([0-9]+ pages' /tmp/pkg1.log | tail -1 | grep -oE '[0-9]+')"
