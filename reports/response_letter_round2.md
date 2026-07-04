# Response to Reviewers — Revision 2 (round-2)

31 May 2026

Prof. Duc Nguyen
Associate Editor
Journal of Chemical Information and Modeling

**Re: Manuscript ID ci-2026-008288 (Revision 2)** — "M-JEPA: Predictive Self-Supervised Learning for Molecular Graphs with Scaffold-Shift Evaluation on Tox21"

Dear Prof. Nguyen,

We thank you and Reviewer 3 for the continued careful evaluation of our manuscript. We are pleased that Reviewer 3 found the first-round revision to have addressed many of the original concerns ("The revision addresses many of my R0 comments. The definitions are more clear, and the addition of RQ1–RQ3 helps."). In this round, Reviewer 3 raised three remaining concerns and one detailed comment. We have addressed all four, and our point-by-point responses follow.

In summary, the changes in this revision are: (i) we adopted the reviewer's suggested title so that it no longer leads with a contribution we have intentionally downgraded; (ii) we removed a redundant Phase-2 preview paragraph that disrupted the Results-overview flow; (iii) we withdrew the argument that comparative Integrated-Gradients (IG) analysis would require stepping outside the leakage-resistant protocol, and instead performed the comparison directly — adding a dedicated controlled experiment (revised *Attribution Diagnostics*; new Supporting Information **Table S17**), a within-objective masking control, and a corresponding reframing of Contribution (1); and (iv) we defined "BFS" at first use.

The first-round point-by-point responses to Reviewers 1–4 are retained, unchanged, below this section for the editor's continuity.

We are grateful for the reviewer's persistence, particularly on the comparative-attribution point: the resulting experiment has clarified the source of the attribution coherence and made the manuscript's claims more precise and better supported.

Sincerely,
Karthik Iyer and Nasser Sabar
La Trobe University

---

## Reviewer 3 — Second-round comments

We thank Reviewer 3 for the positive overall assessment of the first-round revision and for the clear, constructive remaining comments. Each is addressed below in the order given. Reviewer comments are reproduced verbatim in italics.

### Remaining concern 1 — Title vs. contributions

*"Contributions now call the leakage-resistant protocol 'reproducible evaluation practice rather than a methodological claim.' The title still leads with it. I suggest: 'M-JEPA: Predictive Self-Supervised Learning for Molecular Graphs with Scaffold-Shift Evaluation on Tox21.'"*

**Response:** We agree, and we thank the reviewer for resolving the inconsistency between the title and the (intentionally narrowed) contribution statement. We have adopted the reviewer's suggested title verbatim:

> **M-JEPA: Predictive Self-Supervised Learning for Molecular Graphs with Scaffold-Shift Evaluation on Tox21.**

The running head has been updated to match. The term "leakage-resistant" is now retained only as a descriptor of the evaluation practice in the Methods and the Figure 2 caption, consistent with its downgraded status in Contribution (2).

### Remaining concern 2 — Phase-2 preview disrupts Results-overview flow (p. 17)

*"Page 17, line 17 'Phase 2 shows that augmentation choice is a dominant driver of objective-screening performance…' this paragraph previews Phase-2 conclusions in the Results overview section. It either belongs in the Phase-2 subsection itself or merged into the preceding overview paragraph; as a standalone it disrupts the Phase 1 → Phase 2 → Phase 3 flow the overview is establishing."*

**Response:** We agree. The Phase-2 conclusion is already stated in the *Phase 2 Configuration Selection* subsection (the minimal-augmentation-regime result and the subsequent "augmentation paradox" paragraph), so the standalone preview in the Results overview was redundant. We have removed it. The overview now proceeds directly from the three-phase roadmap to the downstream (Tox21) findings, restoring the uninterrupted Phase 1 → Phase 2 → Phase 3 flow the overview is intended to establish.

### Remaining concern 3 — Comparative attribution against the contrastive baseline (p. 35)

*"Page 35, line 10, 'Because no fine-tuned contrastive checkpoint exists under the locked Tox21 protocol, comparative IG analysis would require either fine-tuning the contrastive baseline outside the leakage-resistant pipeline'. I don't follow this argument. Why would this require stepping outside the leakage-resistant pipeline? The authors can take the same scaffold-split training data, fine-tune the InfoNCE-pretrained encoder under the same hybrid schedule, and test on the same scaffold-split test data. The chemical-coherence-of-masking claim currently rests on within-M-JEPA evidence only. Without a contrastive comparison, the observed attribution quality cannot be attributed to connected-subgraph masking specifically rather than to the predictive objective more broadly."*

**Response:** We agree with the reviewer, and we thank them for pressing this point. The reviewer is correct that the comparison does not require stepping outside the leakage-resistant protocol: the Bemis–Murcko scaffold split, per-assay single-task training, and validation-only calibration are all objective-agnostic, so the InfoNCE-pretrained encoder can be fine-tuned through the identical hybrid schedule on the same training fold and evaluated on the same test fold. We have **withdrawn the justification quoted by the reviewer** and removed it from the manuscript.

We have now performed exactly the comparison the reviewer describes. Because the reviewer's comment distinguishes two candidate causes of the observed attribution quality — *connected-subgraph masking specifically* versus *the predictive objective more broadly* — we designed the experiment to separate them by varying one factor at a time. The new control (revised *Attribution Diagnostics*, "Comparative attribution against the contrastive baseline", p. 35; full statistics in Supporting Information **Table S17**) comprises three pretraining arms, all fine-tuned through the identical hybrid schedule on the same scaffold-split folds with IG computed identically:

1. **JEPA vs. the compute-matched InfoNCE contrastive baseline** — isolates the *objective family* (the reviewer's "predictive objective more broadly");
2. **JEPA with connected-subgraph masking vs. JEPA with non-contiguous masking** (contiguity = 1 vs. 0, predictive objective held fixed) — isolates the *masking design* (the reviewer's "connected-subgraph masking specifically").

Because the capped test order is seed-deterministic, the same molecules appear in every arm, enabling paired per-molecule comparison with Wilcoxon signed-rank tests and bootstrap confidence intervals (matching the paired-statistics convention used elsewhere in the manuscript). Across four assays and three seeds (2,964 paired test molecules), we find:

- **At the motif level — the unit of analysis used throughout the section — JEPA produces more concentrated attributions than the contrastive baseline** (Δ top-1 motif share = +0.055; Wilcoxon p < 10⁻²⁵⁰; favouring JEPA on 82% of molecules; positive in every one of the four assays, Δ = +0.013 to +0.115).
- **At the atom level the two objectives are comparable**, and we report this transparently: the contrastive baseline is in fact marginally more concentrated on top-20% atom-attribution mass (Δ = −0.030) and salient-atom connectivity (Δ = −0.013), with atom-level Gini indistinguishable (Δ = +0.009).
- **Connected and non-contiguous masking are statistically indistinguishable** on motif-level coherence (Δ = −0.008; p = 0.13).

Mapping these results onto the reviewer's two alternatives: the motif-level attribution coherence is attributable to the **predictive (context→target) objective**, *not* to connected-subgraph masking specifically. We have therefore reframed our claims to match the evidence:

- **Contribution (1) (Introduction, p. 4)** now describes the JEPA objective as one "whose predictive formulation yields more motif-concentrated attributions than a compute-matched contrastive baseline (Table S17), with connected-subgraph masking specified at algorithm level (Algorithm 1) as a chemically-interpretable target construction rather than the source of attribution coherence."
- **The Phase-2 masking ablation (p. 22)** now extends the connected-versus-non-contiguous contrast from proxy RMSE to attribution coherence, reporting the null masking effect (Δ top-1 motif share = −0.008, p = 0.13; Table S17).
- **The Limitations section** now states that the computational comparison against the contrastive baseline has been performed; chemist-rated agreement and benchmarking against curated structural-alert libraries (DEREK, Kazius–Bursi) remain as future work.

We note one scoping point for transparency, which is stated in the manuscript: this control is run at a reduced compute budget to isolate the objective and masking factors cleanly, so its absolute coherence values are not compared against the full-protocol analysis in the main *Attribution Diagnostics* section — only the within-control paired contrasts are interpreted. We believe this directly resolves the reviewer's concern: the attribution quality is no longer attributed to masking on within-M-JEPA evidence alone, and the comparison shows it derives from the predictive objective rather than the masking design.

### Detailed comment — Define BFS (p. 8, l. 24)

*"Page 8 line 24 - BFS needs to be defined."*

**Response:** Fixed. We now spell out "breadth-first search (BFS)" at first use (Methods, masking description), so the abbreviation is defined before it appears in the *Masking algorithm* paragraph and in Algorithm 1.

---

*(First-round point-by-point responses to Reviewers 1–4 follow below, retained unchanged from Revision 1.)*
