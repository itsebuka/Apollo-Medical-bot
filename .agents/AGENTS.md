# Apollo Project — Agent Rules

## Auto-Push Rule (ATOMIC COMMITS — One Change = One Commit)

After **every single file edit**, immediately commit and push that specific change to GitHub.
Each individual change must be its own commit — **do NOT batch multiple file changes into one commit**.
This ensures every edit shows as a separate contribution on the GitHub contribution graph.

Follow this exact sequence after every individual file edit:

```powershell
cd c:\Users\Ebuka Eleogu\Ebuka-s-adtc-2026
git add <the specific file(s) changed>
git commit -m "<short descriptive message of what changed>"
git push origin main
```

### Rules:
- **One change = one commit = one push.** Never wait until multiple files are done before committing.
- If a task touches 5 files, there must be 5 separate commits and 5 separate pushes.
- Commit the moment a file is saved/written — before moving on to the next file.
- Use `git add <specific file>` (not `git add -A`) to keep each commit scoped to exactly what changed.
- Commit message format: use conventional commit prefixes:
  - `feat:` — new feature or capability
  - `fix:` — bug fix
  - `refactor:` — code restructure, no behaviour change
  - `chore:` — config, tooling, non-code changes
- Be concise but descriptive (e.g. `fix: FTS5 AND query in main.py`, `feat: add /domains endpoint`)
- Never leave the working tree dirty after finishing a task.

**This rule applies to ALL file changes** — backend Python, frontend JSX/CSS, config files, etc.
