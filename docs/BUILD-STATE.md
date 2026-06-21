# Local Whisper Web GUI — BUILD-STATE
> Append-only checkpoint log. One bracketed entry per finished chunk, written in the SAME PR. An interrupted session resumes from the latest `NEXT UP`. Never rewrite history here — append. Growth is not bloat; you read the tail. When the live tail gets long, `/curate` rolls up closed entries into a dated `docs/archive/` digest (never deletes them).

**Entry format:**
```
[<CHUNK-ID> | <one-line what + why> | STATUS: done — branch <name> off <sha>. <root cause>. <the fix>.
 NAMED TEST: <file::test>. PROOF: RED→GREEN + stash-revert. GATE: <verdict, 0 blockers>.
 DOCS: <D-NNN / L-NNN / AUDIT row>. GREEN: <suite counts>; lint+build clean. NEXT UP: <next chunk>.]
```

---

[BOOTSTRAP | groundwork operating system adopted onto a midway repo | STATUS: done — docs-only, no source modified. Adopted at HEAD cd888bf (branch feat/web-gui), clean tree.
 SCAN: stack/commands/chokepoints inferred from source (manifests + CI), not guessed. BASELINE (executed, full ASR stack, Linux/Py3.12): 114 passed, 1 failed, 0 collection errors — the 1 failure is the non-portable Windows test (finding #1, see SYSTEM-SPEC "Known open issues" #1 / D-002), not a code regression; green on the Windows/Py3.9 target. INVARIANTS: INV-1..INV-10 surfaced from source, confirmed, mirrored into CLAUDE.md §5 + AUDIT.md (7 GREEN, 3 PENDING: INV-4, INV-5, INV-9). DOCS GROUNDED: SYSTEM-SPEC + CODEBASE-MAP from actual code with file:line; prior CLAUDE.md folded + archived to docs/archive/CLAUDE.pre-groundwork.md. .gitignore un-ignores the contract docs (D-003).
 NEXT UP: human reviews/merges the docs-only adoption PR; then STEP 0, then the first chunks — (1) guard the non-portable Windows test (finding #1), (2) add the INV-4/INV-5 PENDING tests, (3) optional minimal pytest CI + INV-9 version-assert.]
