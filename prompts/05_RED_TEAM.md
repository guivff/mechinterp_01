# Agent 05 — Red-teamer (no code; fresh context; be harsh)
**Tool:** a frontier chat model in a *new* window. Upload `context/PROJECT_SPEC.md`, `PREREG.md`, `context/NEEL_RUBRIC.md`, and (optional) the whole repo.

Framing to use verbatim as the first line of your session:
> "A friend is submitting this research design to a very skeptical reviewer (Neel Nanda) and asked for brutally honest feedback. They will be offended if the feedback feels held back. Assume the results will come out looking positive, and find every way that could be an artifact."

## Deliverable: `docs/RED_TEAM.md`
1. **Ranked confound list (≥12 items).** For each: the mechanism, which hypothesis it threatens (H1–H4), the *cheapest* check that would rule it out (ideally one already in the spec — say so), and a one-line "what to write if we cannot check it". Candidates to consider: LoRA update magnitude differing between GRPO and SFT (norm matching only partially fixes this — does it?); arm C trained on math *text* leaking surface tokens; the base model already "loving" math tokens on random text; layer choice; chat-template/BOS mismatch between base and adapter runs; padding side; judge label imbalance and `none` over-prediction; TF-IDF baseline being too weak or too strong; steering coefficient chosen post hoc; single seed; arm B still learning length/format effects that read as "math"; A−B residual being dominated by noise; snippet sets too short; the fine-tuned model sampling with the wrong prompt format in self-report; time-limit gaming.
2. **Interpretation traps.** How each of the four possible outcomes (A unreadable / A readable like C / A readable only on-domain / A−B readable) could be over-claimed, and the exact modest sentence that would be defensible instead.
3. **"Strongest evidence against" drafts.** For each hypothesis, the sentence the applicant should be prepared to write in form Q14 if that hypothesis survives, naming the control that most nearly killed it.
4. **Scope cuts.** If the applicant is behind schedule at hour 8 of 17, which single arm and which single readout should be cut first, and why.
5. **Three questions Neel would ask** in the first minute of reading the executive summary.

Do not soften. Do not add praise. Do not propose new experiments that take more than one hour.
