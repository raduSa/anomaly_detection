# What was actually deleted (not relocated) during the page-reduction pass

This is a companion to `page_reduction_plan.md`, which covers material *moved* to
annexes (nothing lost there). This file covers the opposite: specific facts,
examples, or reasoning clauses that were cut outright during prose-tightening and
no longer appear anywhere in the document — main text or annex.

Most of the ~15–20% prose trims were pure wording compression (shorter sentences,
same facts). The items below are the exceptions — places where an actual detail,
example, or number was dropped, not just reworded. Listed by file/section, in the
order they appear in the thesis.

---

## `2-preliminarii.tex`

**§2.3.2 Yahoo Webscope S5** — the parenthetical explaining *why* A1 is noisy was cut:
> "A1 este real și zgomotos" used to continue "(cicluri zilnice/săptămânale dezordonate)" — the specific mechanism (disordered daily/weekly cycles) is gone; the sentence now just asserts noisiness without saying what causes it.

**§2.3.3 CIC-IDS2017** — three separate cuts in the dataset-description paragraph:
- The example list of CICFlowMeter feature types — "(durată, numărul și rata pachetelor/octeților, statistici ale timpului dintre sosiri, numărul de flag-uri TCP, dimensiuni de fereastră etc.)" — was dropped entirely. The reader is told there are "~80 de caracteristici statistice extrase prin CICFlowMeter" but no longer gets any example of what those look like.
- The specific artifact type in the corruption discussion — "artefacte de tip \enquote{appendix} TCP" — was shortened to just "artefacte". The reader no longer learns that the corrupted flows are specifically TCP "appendix" flows.
- The clarifying clause "(etichetate doar formal, nu pe bază de comportament)" explaining what it means for an attack-labeled flow to lack real payload was cut.
- "adresa IP sau fereastra de captură" (the shortcut-learning risk) was trimmed to just "adresa IP" — "capture window" as a second possible shortcut feature is no longer mentioned.

---

## `4-results.tex`

**§4.1.2 "Impact asupra rezultatelor"** (Credit Card discussion) — the specific mechanism named for Transformer Bottleneck AE's mid-table performance was dropped: "apare agregarea medie globală (\emph{global average pooling}) într-un singur vector latent" was folded into a shared, generic description with LSTM AE ("ambele comprimă în continuare întreaga intrare într-un vector/stare sumară"). The global-average-pooling mechanism itself is still described in §2.2.6/Anexa A (the general transformer architecture section), so the *fact* survives elsewhere in the document — it's just no longer stated as the specific explanation in this discussion.

**§4.1.2 "Observații la nivel de paradigmă"** — dropped OC-SVM's false-positive ranking: "cu al doilea cel mai mare număr de fals-pozitive după Predictive LSTM" is gone. The sentence now just places OC-SVM "la mijlocul clasamentului" without the FP-count comparison.

**§4.2 (Yahoo intro, A2 exclusion sentence)** — the reasoning clause "ar ridica aceeași provocare de putere discriminativă discutată pentru A1" was cut; the sentence now just asserts A2 doesn't add a new anomaly type, without the explicit callback to A1's discriminative-power problem.

**§4.3.1 "Comparație Generală"** — the granularity note on the ten minor CIC-IDS2017 attack classes was cut: "toate sub 2\% individual, șapte dintre ele sub 0.2\%" lost the second clause; only "toate sub 2\% individual" remains.

**§4.3.2 "În mare parte nedetectabile"** — the explanatory framing was compressed and lost specificity: originally "Atacurile peste protocoale legitime și atacurile la nivel HTTP (injecție SQL, XSS) produc statistici de flux indistinctibile de utilizarea normală -- detectarea lor ar necesita contoare de autentificare eșuată sau inspecția payload-ului" is now just "statisticile de flux nu conțin semnalul necesar (contoare de autentificare eșuată, payload) pentru aceste atacuri". The explicit naming of SQL injection/XSS as the HTTP-level attack examples, and the "indistinguishable from normal usage" framing, are gone from this sentence (the attack names themselves still appear in Tabelul~\ref{tab:cic-per-class}, just not in this explanatory clause).

---

## `5-explainability.tex`

**§5.2.1 "Caracteristicile principale se potrivesc..."** — two specific SHAP feature names were dropped from the per-attack feature lists:
- `Packet Length Variance` removed from the DDoS/DoS GoldenEye list (now reads only `Bwd Packet Length Std/Mean/Max`).
- `Idle Mean/Max` removed from the DoS Slowhttptest/slowloris list (now reads only `Fwd/Bwd IAT Std/Max`).

These were real, specific technical details (which exact features drove the SHAP attribution) and do not appear anywhere else in the document.

---

## `6-concluzii.tex`

This chapter absorbed the largest genuine content cuts, since it was the last place trimmed to close the final page gap.

**Hypothesis 1 restatement** — "structura temporală inerentă" was shortened to "structura temporală", dropping the "inherent" qualifier that Chapter 2's original hypothesis (§2.1.1, still intact) uses to distinguish *architectural* temporal structure from merely *being given* windowed input. The distinction still exists in Chapter 2; the conclusion's restatement of it is now less precise.

**Hypothesis 2 restatement** — "demonstrabil prin SHAP" was cut from "AE+Context și Temporal IForest exploatează [demonstrabil prin SHAP] evidențe disjuncte". The conclusion no longer states explicitly that this claim was established via SHAP attribution (the SHAP evidence itself is obviously still in Chapter 5 — only this one-clause pointer to the *method of proof*, in the conclusion's summary, is gone).

**Limitations paragraph — the biggest single cut.** The two concrete examples of open questions were removed entirely:
> "(de ce ferestrele temporale dăunează performanței la Setul de Date A și C, sau de ce clase precum FTP-Patator, SSH-Patator, Bot rămân nedetectabile indiferent de setul de caracteristici)"

The paragraph now just says "testarea sistematică a celorlalte presupuneri din Capitolul 4" with no examples given. The underlying *facts* (temporal windows hurting Sets A/C; FTP-Patator/SSH-Patator/Bot being undetectable) are still stated in Chapter 4's own discussions — so nothing is lost from the thesis's factual record — but the conclusion no longer names them as the specific open questions it's pointing at. A reader skimming only the conclusion no longer gets concrete examples of what "testing the other assumptions" would mean in practice.

---

## `1-introducere.tex`

**"Structura Lucrării"** — minor: "protocolul de evaluare -- inclusiv metricile raportate și deciziile metodologice" was shortened to just "protocolul de evaluare", dropping the two-item elaboration of what that protocol section covers. Purely a roadmap-sentence trim, no findings affected.

---

## Net assessment

Almost everything cut was either (a) pure wording compression with the fact intact, or (b) a fact that still survives elsewhere in the document (moved-not-lost, covered in `page_reduction_plan.md`). The list above is the actual "signal loss": about a dozen specific examples, mechanism names, or comparative stats that no longer appear anywhere in the thesis. The most substantive of these is the Conclusion's dropped pair of limitation examples — worth restoring first if you want to claw back any of this, since it's the one place a reader loses concrete, actionable detail rather than just prose color.
