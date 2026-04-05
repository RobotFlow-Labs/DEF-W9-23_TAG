# Boot Context — MANDATORY on every session

## Rule: Read project state BEFORE doing anything

On EVERY session start, read these files in order:
1. `NEXT_STEPS.md` — current status, blockers, what is done
2. `PRD.md` — full PRD including Build Plan table (find the first uncompleted PRD)
3. `anima_module.yaml` — module manifest (if exists)
4. `git log --oneline -10` — recent commits from this and other agents

Only AFTER reading all of these should you respond to the user or start working.

If `NEXT_STEPS.md` does not exist, create it.
If `PRD.md` has a Build Plan table, identify the first incomplete PRD.
If git log shows recent commits with PRD numbers, skip those PRDs.

## Rule: ONLY commit files in YOUR project folder

When committing, NEVER use `git add -A` or `git add .` from a parent directory.
Always use specific paths relative to this project:

```bash
git add src/ tests/ configs/ prds/ PRD.md NEXT_STEPS.md anima_module.yaml \
  docker/ Dockerfile docker-compose.yml pyproject.toml README.md .claude/
```

**NEVER stage or modify files outside this project folder ($PWD).**
Each agent owns its own folder. Keep commits clean and scoped.

**Why:** Multiple agents work in parallel across the same repo. If one agent commits files from another project folder, it creates merge conflicts and breaks other agents. Scope your commits to YOUR folder only.
