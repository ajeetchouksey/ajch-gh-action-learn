# Local OpenAI Integration for ajch-gh-action-learn

This document describes a local-first integration between the ajch-gh-action-learn app and OpenAI, including a minimal MVP implementation, required files, code snippets, CLI usage, GitHub Actions example, security notes, and next steps. It is intended to be committed directly to the repository main branch.

---

## Summary

This integration provides a local-only path for users to augment course JSON files (one JSON per course) with concise AI-generated metadata: a one-line `summary` and an `estimated_minutes` field for each item that lacks them. The integration is designed to keep data local by default and requires users to provide their own OpenAI API key if they choose to use the remote OpenAI models.

The document includes ready-to-drop files and full example code so maintainers can commit and run the augmentation locally.

---

## What this adds

- A short guide and plan for local-first OpenAI augmentation.
- `requirements.txt` additions.
- `openai_wrapper.py` — thin OpenAI client wrapper with retries and JSON parsing.
- `scripts/augment_courses_with_openai.py` — script to augment all course JSONs in a folder, with backup and atomic writes.
- CLI integration notes to add an `augment` subcommand to the existing `learn.py` CLI.
- Example GitHub Actions workflow (optional) and README guidance.

---

## Preconditions / Requirements

- Python 3.8+
- pip
- An OpenAI API key (if using the OpenAI augmentation path). Keep it secret.

---

## Files to add / update

1) requirements.txt

Add the following lines (or ensure they exist):

```
openai>=0.27.0
tqdm
```

2) openai_wrapper.py

Create `openai_wrapper.py` at the project root with the following content:

```python
#!/usr/bin/env python3
import os
import time
import json
from typing import Tuple
import openai

OPENAI_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = os.environ.get("AJCH_OPENAI_MODEL", "gpt-3.5-turbo")
MAX_RETRIES = 3

def get_api_key() -> str:
    key = os.environ.get(OPENAI_KEY_ENV)
    if not key:
        raise RuntimeError(f"OpenAI API key not set in env {OPENAI_KEY_ENV}")
    return key

def call_chat_completion(messages, model=DEFAULT_MODEL, max_tokens=300, temperature=0.2):
    openai.api_key = get_api_key()
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp
        except openai.error.RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2
        except openai.error.APIError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2
        except Exception:
            # treat other errors as fatal after retries
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff *= 2

def summarize_item(title: str, description: str) -> Tuple[str, int]:
    # Prompt asks for a strict JSON object so parsing is robust
    prompt = [
        {"role": "system", "content": "You are a concise assistant that writes a short summary and reasonable time estimate for learning items."},
        {"role": "user", "content":
         f"Item title: {title}\n\nDescription: {description}\n\nReturn ONLY a JSON object with two fields:\n- summary: a one-sentence summary (max 140 chars)\n- estimated_minutes: integer estimate between 5 and 120\nDo not include extra text."}
    ]
    resp = call_chat_completion(prompt)
    text = resp["choices"][0]["message"]["content"].strip()
    # Try to extract JSON
    try:
        # if assistant included text, find first '{'
        idx = text.find('{')
        if idx != -1:
            text = text[idx:]
        j = json.loads(text)
        summary = j.get("summary", "").strip()
        estimated = int(j.get("estimated_minutes", 15))
        if not summary:
            raise ValueError("empty summary")
        if estimated < 1 or estimated > 1000:
            estimated = max(5, min(120, estimated))
    except Exception:
        # fallback: naive heuristic
        summary = (description or title).splitlines()[0][:140]
        estimated = 15
    return summary, estimated
```

3) scripts/augment_courses_with_openai.py

Create `scripts/augment_courses_with_openai.py` with the following content:

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Dict, Any
import openai_wrapper as ow


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(p: Path, data):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


def augment_course_file(path: Path, local_only: bool, dry_run: bool):
    data = load_json(path)
    items = data.get("items", [])
    changed = False
    for i, item in enumerate(items):
        need_summary = not item.get("summary")
        need_est = item.get("estimated_minutes") is None
        if not (need_summary or need_est):
            continue
        title = item.get("title", "")
        desc = item.get("description", "") or item.get("notes", "") or ""
        if dry_run:
            print(f"[DRY] Would augment {path.name} item #{i+1}: {title}")
            continue
        if local_only:
            # local-only fallback values
            summary = (desc or title).splitlines()[0][:140]
            estimated = 15
        else:
            try:
                summary, estimated = ow.summarize_item(title, desc)
            except Exception as e:
                print(f"OpenAI call failed for {path.name} item #{i+1}: {e}")
                summary = (desc or title).splitlines()[0][:140]
                estimated = 15
        if need_summary:
            item["summary"] = summary
        if need_est:
            item["estimated_minutes"] = estimated
        changed = True
    if changed:
        # backup
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        save_json_atomic(path, data)
        print(f"Updated {path.name} (backup: {bak.name})")
    else:
        print(f"No changes for {path.name}")


def find_json_files(dirpath: Path):
    for p in sorted(dirpath.iterdir()):
        if p.is_file() and p.suffix.lower() == ".json":
            yield p


def main():
    parser = argparse.ArgumentParser(description="Augment course JSONs with OpenAI summaries")
    parser.add_argument("--dir", required=True, help="Folder with course JSON files")
    parser.add_argument("--local-only", action="store_true", help="Do not call OpenAI; use heuristics")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    parser.add_argument("--course", help="Only augment a single course id or filename stem")
    args = parser.parse_args()

    dirpath = Path(args.dir).expanduser().resolve()
    if not dirpath.is_dir():
        print("Directory not found:", dirpath)
        return

    files = list(find_json_files(dirpath))
    if args.course:
        files = [p for p in files if p.stem == args.course]

    if not files:
        print("No JSON course files found in", dirpath)
        return

    for p in tqdm(files, desc="Augmenting"):
        try:
            augment_course_file(p, args.local_only, args.dry_run)
        except Exception as e:
            print("Failed for", p.name, e)

if __name__ == "__main__":
    main()
```

4) learn.py CLI change (suggested)

Add a new `augment` subcommand to your existing `learn.py` CLI to forward to the script above. Example argparse addition (paste into your `learn.py` where subcommands are defined):

```python
aug_p = sub.add_parser('augment', help='Augment course JSONs with OpenAI summaries')
aug_p.add_argument('--dir', required=True, help='Folder with course JSON files')
aug_p.add_argument('--local-only', action='store_true', help='Do not call OpenAI')
aug_p.add_argument('--dry-run', action='store_true', help='Show changes without writing')
aug_p.add_argument('--course', help='Only augment one course (filename stem)')
```

And call the script (or import its main) when `cmd == 'augment'`.

5) Example GitHub Actions workflow (optional)

Create `.github/workflows/augment.yml` if you want repository-level augmentation via Actions. This is optional and requires adding a repository secret `OPENAI_API_KEY` in Settings → Secrets → Actions.

```yaml
name: Augment Courses with OpenAI
on:
  workflow_dispatch:
  push:
    paths:
      - 'courses/**.json'

jobs:
  augment:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - name: Run augment
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/augment_courses_with_openai.py --dir courses --commit-changes true
```

> Note: If you want Actions to commit changes back to the repo, add steps to configure git, set a user/email, and push the changes — and ensure Actions has write permission.

---

## Sample course JSON (before)

```json
{
  "id": "git-basics",
  "title": "Git Basics",
  "description": "Learn Git essentials",
  "items": [
    {
      "id": "1",
      "title": "Install Git",
      "type": "task",
      "completed": false,
      "notes": "Install via package manager"
    },
    {
      "id": "2",
      "title": "Create a repo",
      "type": "exercise",
      "completed": false
    }
  ]
}
```

After augmentation, missing fields `summary` and `estimated_minutes` will be added to each item where appropriate.

---

## Usage (local)

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Set OpenAI key (if using OpenAI):

```
export OPENAI_API_KEY="sk-..."
```

3. Dry-run (no network calls):

```
python scripts/augment_courses_with_openai.py --dir ~/path/to/courses --dry-run
```

4. Local-only heuristic augmentation (no network calls):

```
python scripts/augment_courses_with_openai.py --dir ~/path/to/courses --local-only
```

5. Run with OpenAI augmentation:

```
python scripts/augment_courses_with_openai.py --dir ~/path/to/courses
```

---

## Safety and privacy notes

- The app will send item title/description to OpenAI if you run without `--local-only`. Do not include sensitive or private data in course JSONs if you do not want it transmitted.
- Keep your `OPENAI_API_KEY` secret and do not commit it to the repo.
- The script creates a backup of each modified JSON file as `<file>.bak` before writing changes.
- Set `AJCH_OPENAI_MODEL` env var to change the model (defaults to `gpt-3.5-turbo`).

---

## Next steps (recommended roadmap)

1. Merge this document and add the files above to the repo.
2. Add CLI integration and automated tests.
3. Add a lightweight TUI or simple web UI for browsing and approving augmentations.
4. Implement additional AI features (recommend-next, generate-quiz, semantic search) in small iterative PRs.
5. Optionally add GitHub Actions workflow for repo-level augmentation (opt-in via secret).

---

## Contact / attribution

If you want I can open a PR that adds the files and the CLI subcommand, or I can paste the raw files here for you to copy into the repo locally. Reply with which you prefer.
