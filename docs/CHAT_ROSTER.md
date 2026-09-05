# CHAT_ROSTER.md — every live chat, what it owns, what it may see (Fri 2026-09-04 15:30)

The failure mode this file prevents: a blinded task being fed the key, a writer being fed unverified numbers, or two chats editing the same file. When in doubt, a chat gets **less** context, not more.

| Chat | Where | Model / tool | Owns | May see | Must NOT see | Blocked on |
|---|---|---|---|---|---|---|
| **C1 — pod runner** | Claude Code, Terminal, repo on Mac | Claude | `results/`, `tools/`, `figs/`, `docs/RESULTS_DIGEST.md`, `VERIFY.md` *scaffold only*, doc amendments | everything in the repo | — | nothing; runs Tasks B, C, D now |
| **C2 — blind tagger** | separate chat with no project context | ChatGPT or Claude | 68 tag rows + tallies | `V2_TAGGER_LLM_JUDGE.md`, `discordant_A_vs_D_math_readable.md` | `discordant_key.json`, `discordant_sample20.txt`, the unblinded 20-item md, any project doc, the hypothesis | nothing; send now |
| **G — Guiv** | offline, by hand | — | `notes/guiv_tags.csv`, `notes/reading_notes.md`, last three columns + prose of `VERIFY.md`, executive summary, form answers | review packets, digest, firewall | `discordant_key.json` and `patchscope_key.json` until the corresponding notes are written; C2's output until his 20 tags are written | C1's packets (done) |
| **W — writer** | writer chat (P13 lineage) | Claude/ChatGPT | doc body, captions, verification-appendix *structure*, facts skeletons | current digest, firewall, `acc_table_reparsed.md`, C1 reports, G's steering paragraph | anything pending (⏳) in the digest; MOCK; the key files | items 1–2 of OPEN_TASKS + G's paragraph |
| **RT — results red-team** | separate chat with no project context | — | ranked top-5 objections with answered/partial/unanswered | current digest + firewall **only** | the write-up (objections must target evidence, not prose) | nothing; send now |
| **WC — write-up critic** | fresh chat | — | voice and claim-breach review | summary + form answers + firewall | the digest (it judges prose against the firewall, not the science) | G's summary v1 |
| **R — replication** | Claude Code, `~/repl`, branch `replication` | Claude | C s1 (`c852658`), C_masked (`19524db`), `results/REPLICATION_REPORT*.md`, `logs/replication_ledger.md`, adapters on Mac | repo, R1 prompt | the write-up | done; idle |
| **T1 / T2 — theory** | Cowork chats | Claude | `docs/T1_THEORY_BLOCKS.md`, `docs/T2_THEORY_PASSES.md` (§M0 = the alternative that won), `docs/PREREG_v2.md` | digest, theory note, firewall | keys, raw items | done |
| **EV — evaluator** | Cowork subagent | Claude | `docs/EVALUATION_NEEL_STYLE{,_v2}.md` | rubric, summary, digest, firewall | the executive summary until v1 exists | done |
| **Cowork (this)** | Claude desktop | Claude | control-plane docs, prompts, routing, judgement calls | everything | — | — |

## Contamination log (goes in VERIFY.md too)
- Cowork read the unblinded 20-item `discordant_A_vs_D_math.md` and characterised it to G before G tagged → G's calibration sample was redrawn from the other 48 (ids in `discordant_sample20.txt`).
- Cowork read `steer_reading.md` and gave G a reading before G wrote his paragraph → G's paragraph discloses a prior characterisation.
- Cowork read `patchscope_for_human.md` and **withheld** what it saw → G's Patchscope guess is still blind if done before any further discussion.
- C1 generated the tooling *and* audited the re-parse (20/20) → its audit column is self-assessment; G's column in `reparse_audit.md` is unfilled by design.

## Ordering rules
1. G tags before reading C2's output. G names Patchscope letters before opening `patchscope_key.json`.
2. W receives numbers only after they are in the digest with a `results/` path. Nothing marked ⏳ reaches W.
3. RT sees evidence, WC sees prose; neither sees both.
4. One editor per file at a time: C1 edits `results/`, digest, VERIFY scaffold; G edits `notes/` and the VERIFY human columns; Cowork edits the control plane. Merge conflicts are resolved by lane ownership, never by `--theirs` on someone else's lane.

- Sat 03:15 Cowork issued a causal claim sentence that the data contradict (V(A) > V(C_masked)); sent to G, C1 and W; withdrawn 03:35 and replaced by the firewall §1 sentence. Any draft containing "which is why it is hard to detect" is stale.
