# LaTeX Fix: Chapters No Longer Force a New Page + Numeric Appendix Numbering

Context: `thesis/final_version/main.tex` (report class). Two related requests:
1. Chapters should not always start on a fresh page.
2. Appendix chapters should be numbered "Anexa 1", "Anexa 2" (Arabic) instead of "Anexa A", "Anexa B" (`appendix` package default).

All changes live in `thesis/final_version/main.tex`. This took three iterations to get right — the first two attempts looked fine in isolated spot-checks but broke on closer inspection. Documenting the failed attempts too, since the reasons they failed are non-obvious.

## Final working changes

### 1. `\include` → `\input` for every chapter file

```latex
\input{1-introducere}
\input{2-preliminarii}
\input{3-methodology}
\input{4-results}
\input{5-explainability}
\input{6-concluzii}

\appendix
\renewcommand{\thechapter}{\arabic{chapter}}
\input{7-anexe}
```

**Why this was necessary at all:** `\include` unconditionally wraps the included file in `\clearpage ... \clearpage`, independent of whatever `\chapter` itself does — this is how `\include`/`\includeonly` work together (so partial recompilation doesn't shift page numbers). Patching `\chapter`'s own page-break logic (see below) is *not* enough on its own; `\include` was still forcing a page break around every chapter regardless. `\input` has no such side effect.

### 2. Patch `\chapter` to replace the page break with `\par`, not remove it outright

```latex
\makeatletter

% Împiedică \chapter să înceapă întotdeauna pe o pagină nouă. Se înlocuiește
% saltul de pagină cu \par (nu se elimină complet), altfel titlul de capitol
% se lipește de sfârșitul paragrafului anterior, pe aceeași linie.
\patchcmd{\chapter}{\if@openright\cleardoublepage\else\clearpage\fi}{\par}{}{%
  \typeout{Nu s-a putut elimina saltul de pagină din \string\chapter.}%
}
```

Requires `\usepackage{etoolbox}` (for `\patchcmd`) — added near the other package loads.

**First attempt (wrong):** replaced the `\if@openright\cleardoublepage\else\clearpage\fi` with nothing (`{}`). This did stop the forced page break, but left TeX in horizontal mode (mid-paragraph) whenever a chapter happened to fall mid-page. The chapter-heading code (`\@makechapterhead`) doesn't itself call `\par` first — it always relied on `\clearpage` to already be in vertical mode. Result: "Capitolul 2" got typeset as a continuation of the *previous paragraph's last line*, in giant bold font, visibly breaking that line's spacing too. Fix: replace the page break with `\par` instead of deleting it — this cleanly ends the previous paragraph without forcing a new page.

### 3. Shrink the chapter-heading vertical spacing

```latex
% Reduce spațiul vertical din titlul de capitol (implicit gândit pentru
% începutul unei pagini noi -- \vspace*{50pt} înainte, \vskip 40pt după --
% ceea ce lasă un gol mare, nepotrivit, în mijlocul paginii acum că
% \chapter nu mai forțează o pagină nouă). Se patchează valorile de
% spațiere direct în interiorul macro-urilor existente, fără a le
% rescrie complet.
\patchcmd{\@makechapterhead}{\vspace*{50\p@}}{\vspace*{10\p@}}{}{%
  \typeout{Nu s-a putut reduce \string\vspace* de dinaintea titlului de capitol (numerotat).}%
}
\patchcmd{\@makechapterhead}{\vskip 20\p@}{\vskip 6\p@}{}{%
  \typeout{Nu s-a putut reduce \string\vskip dintre numărul și titlul capitolului.}%
}
\patchcmd{\@makechapterhead}{\vskip 40\p@}{\vskip 15\p@}{}{%
  \typeout{Nu s-a putut reduce \string\vskip de după titlul de capitol (numerotat).}%
}
\patchcmd{\@makeschapterhead}{\vspace*{50\p@}}{\vspace*{10\p@}}{}{%
  \typeout{Nu s-a putut reduce \string\vspace* de dinaintea titlului de capitol (nenumerotat).}%
}
\patchcmd{\@makeschapterhead}{\vskip 40\p@}{\vskip 15\p@}{}{%
  \typeout{Nu s-a putut reduce \string\vskip de după titlul de capitol (nenumerotat).}%
}

\end{document}  % (i.e. these patches sit before \begin{document}, after \makeatletter)
```

**Why needed:** `report.cls`'s default `\@makechapterhead`/`\@makeschapterhead` use `\vspace*{50pt}` before the heading and `\vskip 20pt`/`\vskip 40pt` after — sized on the assumption the heading always starts at the top of a fresh page. Once chapters can land mid-page (change #2), that spacing reads as a large, broken-looking gap in the middle of a page.

**Second attempt (wrong):** fully rewrote `\@makechapterhead`/`\@makeschapterhead` from scratch via `\renewcommand` with the reduced spacing values baked in, copy-pasting the body of the original macros. This compiled with `! Undefined control sequence.` / `! Extra \fi.` errors on every chapter. Root cause not fully diagnosed (suspected interaction with `hyperref`'s `nameref.sty`, which does `\let\NR@chapter\@chapter` at load time and calls back into `\@makechapterhead` — the errors surfaced specifically when the retyped macro body was invoked). Fix: don't reconstruct the macros; use `\patchcmd` to substitute only the specific `\vspace*{...}`/`\vskip ...` tokens *inside* the existing macro bodies, leaving everything else (including whatever hyperref/nameref rely on) untouched.

### 4. Arabic numbering for appendix chapters

```latex
\appendix
\renewcommand{\thechapter}{\arabic{chapter}}
\input{7-anexe}
```

`\appendix` (from the `appendix` package / LaTeX kernel) switches `\thechapter` to `\Alph{chapter}` (A, B, C...) by default. Overriding `\thechapter` back to `\arabic{chapter}` right after `\appendix` gives "Anexa 1", "Anexa 2" instead of "Anexa A", "Anexa B" — the localized "Anexa" label itself comes from `polyglossia`'s Romanian `\appendixname`/`\@chapapp`, untouched by this change.

## Verification method

Since visual regressions here are easy to miss from source alone (the bug in attempt #1 was invisible in the LaTeX source — it only showed up as glued-together text in the rendered PDF), each iteration was checked by:
1. Deleting all `.aux`/`.toc`/`.bbl`/etc. and recompiling `xelatex` three times from scratch (stale `.aux` files from interrupted runs previously caused spurious `! Extra }, or forgotten \endgroup.` errors unrelated to the actual source changes — always rule this out first by cleaning and recompiling before treating a compile error as real).
2. `grep`-ing the log for `undefined`, `! `, and any custom `\typeout` failure markers from the patches above (each `\patchcmd` here has a `\typeout` fallback that fires if the pattern wasn't found — meaning the intended text moved/changed and the patch silently no-opped).
3. Using `pdftotext -layout` to locate the page number of specific chapter headings, then `pdftoppm -png -f N -l N` to render just that page and visually inspect it (via the Read tool) — both a case where the heading lands at the top of a page (should look identical to the old behavior) and a case where it lands mid-page (the actual thing being fixed) were checked.

## Gotcha for future edits

Any further changes to chapter-heading formatting in this document should account for the fact that `\@makechapterhead`/`\@makeschapterhead` are being patched (not redefined) in the preamble of `main.tex`. If report.cls's default macro body text ever changes (e.g. switching document classes, or a class update changes exact spacing literals like `50\p@`), the `\patchcmd` calls will silently no-op (with a `\typeout` warning visible in the compile log, not a hard error) rather than fail loudly — always check the log for the "Nu s-a putut..." messages after any change near this area.
