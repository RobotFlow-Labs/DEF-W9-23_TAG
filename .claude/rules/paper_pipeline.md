# Rule: Paper Pipeline (MANDATORY)

## At Boot — ALWAYS Read PIPELINE_MAP.md

Before doing ANY work, check if `PIPELINE_MAP.md` exists in the project root:

```bash
cat PIPELINE_MAP.md
```

### If it EXISTS:
1. Read it completely
2. Find the first `[ ] TODO` step
3. Execute that step
4. Mark it `[x] DONE` when complete
5. Move to the next `[ ] TODO`
6. Update NEXT_STEPS.md with progress after each step

### If it DOES NOT EXIST:
**STOP immediately.** Tell the user:
```
PIPELINE_MAP.md not found. This module has no paper reproduction pipeline.
Run: /plan to create PIPELINE_MAP.md before proceeding.
Without it, I cannot guarantee paper-faithful execution.
```

Do NOT proceed with training, building, or any code changes until PIPELINE_MAP.md is created.

## Why This Rule Exists
We lost GPU time because agents built code that didn't match the paper. PIPELINE_MAP.md is the single source of truth for what to do and in what order. Every step maps to a specific section of the paper. No guessing.
