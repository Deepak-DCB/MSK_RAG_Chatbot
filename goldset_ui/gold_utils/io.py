from pathlib import Path
import json, os, shutil, time
from tempfile import NamedTemporaryFile

def atomic_write_jsonl(objs, out_path: Path, make_backup=True) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if make_backup and out_path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = out_path.with_suffix(out_path.suffix + f".bak.{ts}")
        shutil.copy2(out_path, bak)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(out_path.parent)) as tmp:
        for obj in objs:
            tmp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, str(out_path))
    return out_path

def append_edit_atomic(edit_obj: dict, edits_path: Path):
    edits_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(edit_obj, ensure_ascii=False)
    with open(edits_path, "a", encoding="utf-8") as f:
        f.write(line + "\n"); f.flush(); os.fsync(f.fileno())
    return edits_path

def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for s in f:
            s = s.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                continue
    return items

def remove_edit_for_id(edits_path: Path, qid: str):
    if not edits_path.exists():
        return
    items = load_jsonl(edits_path)
    kept = [e for e in items if e.get("id") != qid]
    atomic_write_jsonl(kept, edits_path, make_backup=True)