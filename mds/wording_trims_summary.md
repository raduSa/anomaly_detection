# Wording-only trims (no information lost)

Third companion to `page_reduction_plan.md` (material moved to annexes) and
`removed_content_summary.md` (the ~dozen places a real fact/example was actually cut).
This file covers everything else: the bulk of the editing — sentences and paragraphs
rewritten more densely with every fact, number, and citation preserved. Grouped by
chapter, with a few representative before/after pairs per section rather than an
exhaustive line-by-line diff.

The general pattern throughout: cut connective tissue ("din punct de vedere
arhitectural, aceasta înseamnă că…", "este important de menționat că…"), redundant
lead-ins that only restated a table/figure caption, and hedging phrases, while
keeping every number, model name, citation, and specific claim intact.

---

## `1-introducere.tex`

**"Structura Lucrării"** — five short paragraphs (one per chapter, each starting
"Capitolul~X…") merged into one dense paragraph. "Restul lucrării este organizat în
cinci capitole." folded into the flow instead of standing alone. All five
chapter-by-chapter descriptions kept.

## `2-preliminarii.tex`

**§2.1 "Proiectarea și Abordarea Cercetării"** — "am propus următoarea abordare: în
loc să…" tightened to "în loc să…"; the paradigm-axis and dataset-axis paragraphs
each lost a few connective words ("care trebuie să renunțe la orice, cu excepția…"
type elaboration) without losing the four-paradigm / three-dataset enumeration
itself.

**§2.2 One-Class SVM + Isolation Forest** — merged into one subsection ("Modele
Clasice"), itemized hyperparameter list (`kernel='rbf'`, `gamma='scale'`, `nu=0.01`)
folded into inline prose. All three hyperparameters and their justifications kept.

**§2.2 MLP AE / LSTM AE / LSTM Predictiv** — cut phrases like "Din punct de vedere
arhitectural, aceasta înseamnă că…" and "rezumând întreaga secvență"; architecture
descriptions (encoder/decoder shape, bottleneck, scoring convention) all kept.

**§2.3.1 Credit Card** — "anonimizate din motive de confidențialitate" → "anonimizate";
"analiza exploratorie a comunității" dropped from the Kaggle-listing sentence (the
citation itself stays); "dedicat special gestionării datelor dezechilibrate și
detecției anomaliilor pe acest set de date" → "dedicat gestionării datelor
dezechilibrate". Feature list (`Time`, `V1`–`V28`, `Amount`) and both citations kept.

**§2.3.2 Yahoo S5** — "Acest lucru este ceea ce permite…" → "ceea ce permite…"; minor
compression throughout, benchmark table and A1/A2–A4 split untouched.

**§2.3.3 CIC-IDS2017** — "captat pe parcursul a cinci zile" → "captat pe cinci zile";
several similar micro-trims. (The three genuine detail-losses in this section —
CICFlowMeter feature examples, the "TCP appendix" description, the mislabeled-flow
clarifier — are in `removed_content_summary.md`, not here.)

## `3-methodology.tex`

**§3.1 per-dataset preprocessing subsections** — condensed from numbered
step-by-step lists to one dense paragraph each, with full detail moved to Anexa A
(see `page_reduction_plan.md`); the summary paragraphs themselves are also more
tightly worded (e.g. "Time este eliminat (nu reprezintă informație semnificativă,
întrucât este doar timpul de la prima tranzacție din setul de date)" → "Time este
eliminat (nu reprezintă informație semnificativă)").

**§3.2.1 Metrici Raportate** — "ceea ce face ca rezultatele raportate să fie
dependente de prag prin construcție" → "ceea ce face rezultatele raportate
dependente de prag prin construcție"; similar trims throughout. The `kim2022rigorous`
citation, the direct quote, and the PR-AUC justification all kept verbatim.

**§3.2.2 Selecția Pragului** — "discutate mai sus" and similar back-references cut;
the $F_\beta$, $\beta=2$ definition and the oracle-threshold caveat kept in full.

**§3.2.3 Point Adjustment** — "este considerat ca fiind detectat" → "este considerat
detectat"; "fără nicio ajustare retroactivă" → "fără ajustare retroactivă". The
Pearson $r=-0.59$ SWaT citation kept.

## `4-results.tex`

**Table lead-ins** — "Tabelul~\ref{...} sumarizează toate cele N modele evaluate"
removed wherever the caption already said the same thing (§4.1.1, §4.2.1, §4.2.2,
§4.2.3); any non-redundant clause in those sentences (e.g. "Nu a fost folosită nicio
variantă de Transformer Cauzal pentru acest set de date") was folded into the
surrounding text instead of dropped.

**§4.1.2 Discuție (Credit Card)** — all four paragraphs ("Nu există o secvență
autentică…", "Impact asupra rezultatelor", "Observații la nivel de paradigmă",
"Sanity check") tightened ~15–20%; every AP/ROC-AUC/FP number and every named model
kept. (One real drop — OC-SVM's FP ranking — is in the sibling doc.)

**§4.2.2/§4.2.3 Discuție (A3, A4)** — trimmed surrounding connective prose while
deliberately keeping two sentences **verbatim**: the LSTM AE bottleneck-failure
sentence and the Transformer global-attention explanation, since Chapter 5 §5.1
tests that claim directly.

**§4.3.1 "Comparație Generală"** — the three observation paragraphs ("Caracteristicile
de context al gazdei domină", "Adăugarea ferestrelor temporale…", "Reconstrucția
depășește predicția") tightened; every AP delta (+0.172, +0.157, +0.255, etc.) kept.

**§4.3.2 "Detectate constant"** — DDoS and PortScan bullets reworded, numbers folded
inline (e.g. "IForest+Context este comparativ mai slab aici (0.601)" merged into the
PortScan bullet instead of its own sentence); all AP figures kept.

**§4.3.3 Discuție** — "Caracteristicile de context al gazdei…" paragraph tightened;
the **entire "Un ansamblu practic" paragraph was left untouched**, verbatim, since
Chapter 5 §5.2 tests it directly.

**§5.2.4-adjacent ensemble paragraphs** (technically in `5-explainability.tex`, see
below) and the CIC results-table paragraph both had "pe întregul set de test aliniat
pe rânduri" style methodological qualifiers condensed — the row-alignment procedure
itself is still documented in full in Anexa I (`ch:ensemble-score-alignment`).

## `5-explainability.tex`

**§5.1 intro (A3 hypothesis)** — three sentences merged into two; "Astfel, se
presupune că, pentru ferestrele A3 anormale," → "Se presupune deci că, pentru
ferestrele A3 anormale,". Both bottleneck mechanisms (LSTM AE's single hidden state
vs. Transformer's global attention) kept.

**Heatmap subsection** — the qualitative-vs-quantitative framing sentence tightened;
descriptive content that used to sit in body prose (which model's error looks more
"diffuse" vs. "concentrated") was consolidated into the merged figure's caption
instead of repeated in the text — reorganized, not lost (figure itself moved to
Anexa F, see the annex-moves doc).

**§5.1.2 Gini paragraph** — "nu doar în puținele exemple pe care le arată întâmplător
o hartă termică" → "nu doar în exemplele arătate întâmplător de o hartă termică";
similar compression on the bimodality explanation, core claim (majority cluster at
Gini≈0.6, secondary population <0.4) kept in full.

**§5.2 SHAP intro** — the two-hypothesis list reformatted from numbered prose ("1. …
2. …") to inline (1)/(2); "O validare importantă, efectuată pe parcursul
experimentării și demnă de menționat" → "O validare suplimentară". Both hypotheses
and the pointer to Anexa H kept.

**§5.2.2 AE+Context** — intro sentence tightened, the figure-description
parenthetical folded in after moving the figure itself to Anexa G. Both context
features (`ctx_min_port_cnt`, `ctx_min_flow_cnt`) and both class breakdowns (8/15,
7/15) kept.

**§5.2.3 "Construirea Scorului Combinat"** — heaviest compression in the chapter:
three paragraphs tightened by roughly a third, but both formulas
($s_{\text{soft}}$, $s_{\text{max}}$), the $w \in \{0.3,\dots,0.7\}$ range, and the
rank-normalization vs. $z$-score rationale all kept intact.

**§5.2.4 "Acoperire per Clasă vs. AP Agregat"** — both paragraphs tightened; every
number in the results table (0.882, 0.924, 0.943, 0.841, 0.205, etc.) kept.

## `6-concluzii.tex`

All three paragraphs tightened, most heavily of any chapter since this is where the
final page boundary was closed. Connective phrases like "Cât despre cele două
ipoteze formulate în…" → "Cele două ipoteze din…" throughout. Both hypotheses'
confirm/nuance verdicts and both future-work directions (systematic testing of other
assumptions; extending the ensemble study to Sets A and B) kept — only the two
concrete limitation *examples* were dropped, and that's covered as a real cut in
`removed_content_summary.md`, not here.

---

## Net effect

None of the trims above change what the thesis claims or what evidence backs each
claim — every number, citation, model name, and specific mechanism referenced stays
in the document (main text or annex). What changed is sentence density: fewer words
per fact, fewer restated setups, fewer "as discussed above" back-references. This
category did the least work per line toward the page count (see the "chunky
pagination" note in `page_reduction_plan.md`) but was low-risk to apply broadly.
