# Apollo Project — Agent Rules

## Auto-Push Rule

After **every** code change made to this workspace, immediately commit and push to GitHub.

Follow this exact sequence after any file edit:

```powershell
cd c:\Users\Ebuka Eleogu\Ebuka-s-adtc-2026
git add -A
git commit -m "<short descriptive message of what changed>"
git push origin main
```

Rules for the commit message:
- Be concise but descriptive (e.g. `fix: FTS5 AND query`, `feat: add /domains endpoint`)
- Use conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `chore:`
- Never leave the working tree dirty after finishing a task

**This rule applies to ALL file changes** — backend Python, frontend JSX/CSS, config files, etc.
