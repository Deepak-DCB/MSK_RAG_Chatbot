# Codebase Scope

Canonical runtime files:
- `backend/main.py`
- `frontend/app.js`, `frontend/index.html`, `frontend/styles.css`
- `VectorDB/qaEngine.py`
- `scripts/run_eval_production.py`

Legacy/prototype paths are non-canonical:
- `app.py`
- `runAll.py`
- older ad hoc scripts and notebooks

Rules:
- Prefer modifying canonical runtime paths.
- Do not migrate production behavior into legacy files.
- Keep frontend and backend contracts synchronized.
