# Second page-reduction pass: 23 → 20 content pages

## Goal and result

Target: the main content (Introducere through Concluzii) should occupy at most 20
pages, with the Conclusion finishing around page 25 (content starts on page 6).

| Stage | Concluzii starts | Anexe starts | Content pages (Intro→Concluzii) |
|---|---|---|---|
| Before this pass | page 27 | page 29 | 23 |
| After this pass | page 25 | page 26 | **20** |

Verified by compiling with `xelatex` (2 passes) and reading `main.toc`. No
undefined references, no LaTeX errors. Three pre-existing/minor "Overfull hbox"
warnings remain (all under 1pt except one ~0.6cm one in an annex figure caption),
consistent with what was already present before this pass — cosmetic only, not
addressed since fixing them isn't needed to hit the page target.

All changes are in `thesis/shortened_final_version/`. Per your instructions,
**`1-introducere.tex` and `3-methodology.tex` were left untouched by me** after
you took over editing those two chapters yourself partway through.

## What was applied, in the order requested (cheapest/safest first)

### 1. Wording trims (no facts, numbers, or citations lost)

Applied throughout `2-preliminarii.tex` (§2.1, §2.3 dataset descriptions),
`4-results.tex` (all four "Discuție" blocks: Credit Card, Yahoo A1/A3/A4 summary,
CIC-IDS2017 general comparison, per-attack breakdown, final discussion), and
`5-explainability.tex` (SHAP intro, Gini section, ensemble-construction and
per-class coverage sections). Same pattern as the existing trims: cut connective
tissue and restated setup, keep every number/model name/citation.

Left untouched, as instructed by earlier guidance still in force: the LSTM AE
bottleneck-failure sentence and Transformer global-attention explanation in
§4.2.2 (A3 discussion), and the full "Un ansamblu practic" paragraph in §4.3.3 —
both are directly tested by Chapter 5 and shouldn't drift from their exact wording.
`6-concluzii.tex` got only a light trim; both hypotheses' verdicts, both
future-work directions, and the two limitation examples (FTP-Patator/SSH-Patator/
Bot; temporal windows hurting Sets A/C) were all kept, since deleting real ideas
was to be a last resort and wasn't needed here.

### 2. One additional annex (the one you allowed)

Added **Anexa J, "Detalii ale Mecanismelor de Model"** to `7-anexe.tex`. This is
the single biggest lever in this pass. It holds the full, previously-inline
mechanism-level detail for each model in §2.2 (Preliminarii) — the three
One-Class SVM hyperparameter bullets, the Isolation Forest path-length mechanics,
LSTM AE/LSTM Predictive internal wiring, and the full Bottleneck Transformer AE /
Causal Transformer bullet pair.

The main text (`2-preliminarii.tex`, §2.2) now gives each model a short paragraph
(paradigm, defining mechanism in one sentence, score convention) and points to
Anexa J for the rest. No graph or table was moved — this is pure prose, and the
underlying facts are all still in the document, just relocated per your rule.

### 3. Table/figure resizing (typographic only, nothing moved or dropped)

- `tab:cic-results` (§4.3.1): `\resizebox` width `0.8\textwidth` → `0.72\textwidth`
- `tab:cic-per-class` (§4.3.2): `\resizebox` width `0.85\textwidth` → `0.78\textwidth`
- Gini distribution figure (§5.1.1): `0.55\textwidth` → `0.46\textwidth`
- Temporal-SHAP feature-profile figure pair (§5.2.1): `0.42\textwidth` → `0.38\textwidth` each
- AE+Context PortScan SHAP summary figure (§5.2.2): `0.5\textwidth` → `0.44\textwidth`

All checked visually (rendered PDF pages) — still fully legible at these sizes.

### 4. One bug fix along the way

The new condensed Transformer paragraph in §2.2.6 initially produced a ~2cm
overfull hbox (`\texttt{TransformerEncoder}` wouldn't break). Fixed by adding
soft hyphenation points inside the `\texttt{}` token rather than rewording
further, since the sentence was already at the length I wanted.

## What was *not* touched

- No graphs or tables were moved out of the current 20 pages, per your
  instruction — the only relocation was prose (model-mechanism detail) into the
  one new annex.
- No real ideas/claims were deleted. The only content-level cuts in this pass
  were the same class of minor reasoning-clause trims already used in the prior
  round (e.g. dropping the explicit "same challenge as A1" callback when
  excluding Yahoo A2) — nothing a reader loses access to elsewhere in the text.
- Margins (2.1cm), `\arraystretch` (0.85), and the chapter-heading spacing
  patches were already at their previous tightened values and were left as-is.
