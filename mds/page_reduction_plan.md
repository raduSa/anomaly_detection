# Thesis page reduction: 31 → 20 pages (target reached)

## Result

| Milestone | Content pages (Introducere → Concluzii) |
|---|---|
| Baseline | 31 |
| Round 1 (your 4 requested cuts + 1 analogous one) | 27 |
| Round 2 (mechanical prose trims, items 1–6 from round-1 report) | 26 |
| Round 3 (structural cuts + table/figure resizing + margin) | **20** |

All changes are live in `thesis/shortened_version/`. The document compiles cleanly with `xelatex` (2 passes, `biber` for the bibliography) — no undefined references, no LaTeX errors. There are 13 minor "Overfull \hbox" warnings (largest ~0.9cm), listed at the end of this doc — cosmetic, worth a quick look but not urgent.

**Total document (with annexes + bibliography): 42 pages.**

## How measurement worked

Compiled with `xelatex` (2 passes) and read the printed page number of each `\chapter` from `main.toc`. "Content pages" = from the start of Introducere to the last page before "Anexe" — matches your stated counting rule.

**Key finding from this exercise:** LaTeX pagination is *chunky*, not linear. Trimming a paragraph by a few words only removes a page if the *cumulative* cut in that stretch of text crosses a full page boundary (~40–45 lines here); otherwise it just leaves extra whitespace at the bottom of a page that was already going to break there. Several rounds of real, substantive trimming produced *zero* measured page change for this reason — the win came in bursts, when accumulated cuts finally tipped a chapter over a boundary. This is why the log below has some rounds with big jumps and some with none despite similar effort.

---

## What was actually changed, in order of cost-effectiveness

### Tier 1 — do these regardless of target page count (high value, ~zero downside)

1. **Move secondary/supporting material to annexes, keep a one-line pointer in the main text.** This was the core lever for the whole exercise and is the single most defensible category of edit — nothing is lost, it's just relocated to where a reader who wants the detail can still find it.
   - §2.2.6 Transformer architecture rationale → Anexa A
   - §4.2.4 Yahoo trivial-baseline check → Anexa E    CHECK
   - §5.2.1 two of three SHAP figure pairs (position-concentration, beeswarm) → Anexa G   CHECK
   - §3.1 preprocessing step-by-step detail (all three datasets) + DeepAnt comparison → Anexa A    CHECK
   - §4.3.1 CIC-IDS2017 attack-composition table (flat/context) → Anexa D, next to the already-annexed temporal-set table     CHECK
   - §4.3.2 per-class results table (`tab:cic-per-class`) and the DoS-lente sub-breakdown → Anexa C
   - §5.2.2 AE+Context PortScan SHAP summary figure → Anexa G (the small confirming table stays in the main text)
   - §5.1 Yahoo A3 qualitative heatmap figure → new Anexa F (kept a 2-sentence pivot in the main text: "consistent with the hypothesis, but not sufficient to confirm the mechanism — see the quantitative Gini check below")    CHECK
   - **Combined effect:** 31 → 27 pages.

2. **Tighten table/figure sizing document-wide** — no content lost, purely typographic:   CHECK
   - `\renewcommand{\arraystretch}{0.85}` added once in `main.tex` (tighter row spacing on every table, main text and annexes alike)
   - Reduced `\resizebox` target width on the two full-width tables that remained in the main text (from `\textwidth` to `0.8\textwidth`), and switched the four plain (non-resized) comparison tables in Chapter 4 to `\footnotesize`
   - Shrank the Yahoo-A3 heatmap figure (`0.55\textheight`→`0.4\textheight`, and merged two side-by-side figures with duplicate captions into one), the Gini-distribution figure (`0.7\textwidth`→`0.55\textwidth`), and the temporal-SHAP feature-profile figure (`0.48\textwidth`→`0.42\textwidth`)
   - **Effect:** contributed roughly 2–3 of the pages saved in round 3, hard to isolate exactly given the chunky-pagination effect above, but consistently moved things in the right direction with no readability cost I could detect.

3. **Cut throwaway lead-in sentences that only restate the caption** ("Tabelul X sumarizează cele N modele evaluate" right above a table whose caption says the same thing). Free words, zero information loss.

### Tier 2 — safe, still recommend keeping, moderate effort

4. **Trim "Discuție" prose in Chapter 4 and the SHAP/ensemble prose in Chapter 5 by ~15–20%** (hedges and restated setup cut, every specific claim and number kept). Done carefully per your instruction on point 4 — I explicitly left three passages **untouched, verbatim**, because Chapter 5 or the Conclusion depend on their exact wording:
   - The A3 "eșuare cauzată de blocajul de compresie (bottleneck)" sentence — this is the hypothesis Chapter 5 §5.1 tests directly.
   - The Transformer global-attention explanation right after it — same reason.
   - The full "Un ansamblu practic" paragraph in §4.3.3 — this is the premise Chapter 5 §5.2 tests directly.
   - Both hypotheses in §2.1.1 "Ipoteze și Comportament Așteptat" — the Conclusion references them by name.
5. **Merge/shorten less-central model descriptions** in Chapter 2 (One-Class SVM + Isolation Forest combined into one "Modele Clasice" subsection; MLP/LSTM AE/Predictive LSTM descriptions tightened ~15%) and **shorten dataset descriptions** (Credit Card, CIC-IDS2017 corruption paragraph) — kept every number and every citation, cut restated framing.
6. **Tighten the Introduction's chapter-by-chapter roadmap** and the **Conclusion** (~20–25% shorter) — kept both confirmed/nuanced-hypothesis claims and both future-work directions, cut connective tissue.

### Tier 3 — the lever that actually closed the last 1–2 pages: margins

7. **Reduced the main-matter margin from 2.5cm to 2.3cm** (`\newgeometry{margin=2.3cm}` in `main.tex`, currently applied). This alone was worth about 2 pages. I tested 2.1cm too — it gave no further benefit over 2.3cm, so I kept 2.3cm as the more conventional value.

   **This is the one change on this list I'd flag for your explicit sign-off rather than treat as "obviously fine."** Everything else above is a content/relocation decision within your control. A margin is sometimes a fixed requirement in a faculty formatting guide (the document's own comments reference "ghidul de redactare"), so before submitting: check whether 2.3cm margins are still compliant. If not, you'd need to find the last ~1 page some other way (see "if you need it back" below) or confirm 2.5cm is actually mandatory vs. just what was chosen originally.

---

## If margins turn out to be non-negotiable (need ~1 page back another way)

In rough order of how much was left on the table without further quality loss, based on what worked above:

- Chapter 4 and 5 are still 14 of the 20 pages (70%) combined — the highest remaining density. A second pass specifically tightening the CIC-IDS2017 "Comparație Generală" three observation paragraphs, or moving one more secondary figure (e.g. the AE+Context small confirming table) to an annex, is the next most obvious target.
- The Yahoo benchmark table in §2.3.2 and the `tab:cic-results` 13-row table could go slightly narrower still (`0.8\textwidth` → `0.7\textwidth`) with a small legibility cost.
- A handful of figure captions in Chapter 5 (e.g. the temporal-SHAP feature-profile caption) still restate what the body text says a paragraph later — could be shortened further.

I did not do these because each was a genuine (if small) content/presentation trade-off, and the target was already met via the margin change — happy to apply any of them if you'd rather not touch margins at all.

## Minor cleanup worth a look before submission

13 small "Overfull hbox" warnings appeared over the course of trimming (mostly under 0.5cm, one at ~0.87cm) — these come from sentences that no longer break as cleanly as before after editing. They don't look catastrophic in the areas I spot-checked, but a quick visual skim of the PDF (particularly around Chapter 3 §3.1.1, the AE+Context ensemble-alignment annex, and the transformer-architecture annex section) is worth doing before you submit, since I can't visually inspect rendered pages myself.
