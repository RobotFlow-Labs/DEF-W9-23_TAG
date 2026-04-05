# NEXT_STEPS.md — Mandatory Session Tracking

## Rule: ALWAYS read and update NEXT_STEPS.md

**On session start:**
1. FIRST thing you do: read `NEXT_STEPS.md` in the project root
2. If it does not exist, create it
3. Summarize what was left to do before starting any work

**On session end (or when asked to save/commit):**
1. Update `NEXT_STEPS.md` with:
   - Current date (use ISO format: 2026-03-17)
   - What was accomplished this session
   - What is still TODO
   - Any blocking issues
   - Models/datasets needed (with download commands if known)
   - MVP readiness score (0-100%)

**Format:**
```markdown
# NEXT_STEPS.md
> Last updated: {date}
> MVP Readiness: {score}%

## Done
- [x] Item completed

## In Progress
- [ ] Item being worked on

## TODO
- [ ] Item not started

## Blocking
- Description of blocker

## Downloads Needed
- Model/dataset name — size — command
```

**Why:** We run 15+ modules across multiple agents. Without NEXT_STEPS.md, context is lost between sessions and work gets duplicated. This is the single source of truth for module progress.

**IMPORTANT:** This rule applies to THIS project only. Do not scan or modify other project directories. Use /anima-reboot-dev to get a full status check.
