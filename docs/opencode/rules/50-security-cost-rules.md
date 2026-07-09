# Security and Cost Rules

## Secrets and credentials
- Never commit `.env` or secret-bearing files.
- Treat Supabase service key and JWT secret as sensitive server-only values.

## Cost controls
- Use dry-run whenever possible before paid runs.
- Use bounded evaluation (`--max-cases`, `--max-estimated-cost-usd`).
- Prefer smaller models for non-critical helper tasks when quality permits.

## Command safety
- Do not run destructive shell commands (`git reset --hard`, `git clean -fd`, `rm -rf`) unless explicitly requested and justified.

## Deployment constraints
- Assume free-tier infrastructure and enforce latency and efficiency discipline.
