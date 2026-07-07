import os
import sqlite3
import json
import uuid
import datetime
import traceback
from typing import Optional
from threading import Thread

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

# Import pipeline modules
from pipeline.chunker import chunk_text
from pipeline.segmenter import segment_text
from pipeline.tree_builder import build_tree, annotate_stats
from pipeline.structure_extractor import extract_structure
from pipeline.taxonomy import TAG_DEFINITIONS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = "db.sqlite"


def load_local_env():
    """Load local .env values into os.environ without overriding existing env."""
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()


def resolve_openai_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """Env key takes precedence; otherwise use key supplied by the client."""
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    provided = (provided_key or "").strip()
    return provided or None


app = FastAPI(title="Narrative Story Structure Extractor")


class SourceUpdate(BaseModel):
    source: str = ""


class TitleUpdate(BaseModel):
    filename: str

# -----------------------------------------------------------------------------
# Database Setup and Helpers
# -----------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def db_init():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT,           -- 'pending', 'processing', 'completed', 'failed'
                progress_log TEXT,     -- JSON array of strings
                error_message TEXT,
                raw_text TEXT,
                segmented_blocks TEXT, -- JSON string
                tree_json TEXT,        -- JSON string
                structure_json TEXT,   -- JSON string
                source_note TEXT,
                source_url TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor = conn.execute("PRAGMA table_info(tasks)")
        existing_columns = {row["name"] for row in cursor.fetchall()}
        if "source_note" not in existing_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN source_note TEXT")
        if "source_url" not in existing_columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN source_url TEXT")
        conn.commit()

db_init()

def normalize_source(source: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    value = (source or "").strip()
    if not value:
        return None, None
    if value.startswith(("http://", "https://")):
        return None, value
    return value, None

def db_create_task(
    task_id: str,
    filename: str,
    raw_text: Optional[str] = None,
    segmented_blocks: Optional[str] = None,
    source: Optional[str] = None,
):
    log = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task created. Waiting to start..."]
    source_note, source_url = normalize_source(source)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO tasks
                (id, filename, status, progress_log, raw_text, segmented_blocks, source_note, source_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                filename,
                "pending",
                json.dumps(log, ensure_ascii=False),
                raw_text,
                segmented_blocks,
                source_note,
                source_url,
            )
        )
        conn.commit()

def db_update_source(task_id: str, source: Optional[str]) -> bool:
    source_note, source_url = normalize_source(source)
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE tasks SET source_note = ?, source_url = ? WHERE id = ?",
            (source_note, source_url, task_id),
        )
        conn.commit()
        return cursor.rowcount > 0

def db_update_filename(task_id: str, filename: str) -> bool:
    value = filename.strip()
    if not value:
        raise ValueError("Filename cannot be empty")
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE tasks SET filename = ? WHERE id = ?",
            (value, task_id),
        )
        conn.commit()
        return cursor.rowcount > 0

def db_add_log(task_id: str, message: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT progress_log FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if row:
            log = json.loads(row["progress_log"])
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            log.append(f"[{timestamp}] {message}")
            conn.execute(
                "UPDATE tasks SET progress_log = ? WHERE id = ?",
                (json.dumps(log, ensure_ascii=False), task_id)
            )
            conn.commit()

def db_update_task(task_id: str, status: str, error_message: Optional[str] = None,
                   segmented_blocks: Optional[str] = None, tree_json: Optional[str] = None,
                   structure_json: Optional[str] = None):
    with get_db() as conn:
        query = "UPDATE tasks SET status = ?"
        params = [status]
        
        if error_message is not None:
            query += ", error_message = ?"
            params.append(error_message)
        if segmented_blocks is not None:
            query += ", segmented_blocks = ?"
            params.append(segmented_blocks)
        if tree_json is not None:
            query += ", tree_json = ?"
            params.append(tree_json)
        if structure_json is not None:
            query += ", structure_json = ?"
            params.append(structure_json)
            
        query += " WHERE id = ?"
        params.append(task_id)
        
        conn.execute(query, tuple(params))
        conn.commit()

# -----------------------------------------------------------------------------
# Background processing function
# -----------------------------------------------------------------------------

def process_scenario(
    task_id: str,
    raw_text: Optional[str],
    filename: str,
    pre_segmented_blocks: Optional[list] = None,
    openai_api_key: Optional[str] = None,
):
    db_update_task(task_id, "processing")
    db_add_log(task_id, f"Started processing file: {filename}")
    
    try:
        if pre_segmented_blocks is not None:
            db_add_log(task_id, "JSON file upload detected. Skipping chunking and OpenAI segmentation.")
            all_blocks = pre_segmented_blocks
        else:
            api_key = resolve_openai_api_key(openai_api_key)
            if not api_key:
                err_msg = "OpenAI API key is not set. Add OPENAI_API_KEY to env or enter it in the interface."
                db_add_log(task_id, f"ERROR: {err_msg}")
                db_update_task(task_id, "failed", error_message=err_msg)
                return
            
            client = OpenAI(api_key=api_key)
            
            # 1. Chunker
            db_add_log(task_id, "Splitting scenario text into chunks...")
            chunks = chunk_text(raw_text, target_words=400, max_words=550)
            db_add_log(task_id, f"Text split into {len(chunks)} chunks.")
            
            # 2. Segmenter
            all_blocks = []
            open_loop_ids = set()
            introduced_heroes = []
            prev_tail = ""
            
            for i, chunk in enumerate(chunks):
                db_add_log(task_id, f"Processing chunk {i+1}/{len(chunks)} ({len(chunk.split())} words) via OpenAI GPT-4o...")
                context_note = ""
                if i > 0:
                    heroes_note = "; ".join(introduced_heroes) if introduced_heroes else "пока никого"
                    context_note = (
                        f"Предыдущий фрагмент заканчивался так: \"...{prev_tail[-200:]}\". "
                        f"Открытые ранее петли (loop_id), которые ещё могут закрываться здесь: "
                        f"{sorted(open_loop_ids) if open_loop_ids else 'нет'}. "
                        f"УЖЕ ВВЕДЁННЫЕ ГЕРОИ (через STORY_BOUNDARY) ранее в этом видео: {heroes_note}. "
                        f"Если кто-то из них упоминается в этом фрагменте снова — это НЕ STORY_BOUNDARY "
                        f"(используй ARC_OPEN/TENSION/CONTEXT/итд. по смыслу). STORY_BOUNDARY можно "
                        f"использовать в этом фрагменте только для объекта, которого нет в списке выше.\n\n"
                        f"ВАЖНЫЙ ПРИМЕР: называние героя не всегда прямое ('это машина X'). Иногда оно "
                        f"подаётся косвенно — например через вовлекающий рассказ от 2-го лица "
                        f"('ты подходишь к машине... тебе выдали не пятьдесят третий, тебе выдали "
                        f"пятьдесят второй') или просто через номер/индекс без явного 'это ГАЗ-52'. "
                        f"Такой косвенный, но фактически ПЕРВЫЙ момент опознания героя — это тоже "
                        f"STORY_BOUNDARY, даже без хрестоматийной фразы-объявления."
                    )
                
                result = segment_text(client, chunk, context_note)
                blocks = result.get("blocks", [])
                
                for b in blocks:
                    tag = b.get("tag", "")
                    loop_id = b.get("loop_id")
                    if tag == "OPEN_LOOP" and loop_id:
                        open_loop_ids.add(loop_id)
                    elif tag == "CLOSE_LOOP" and loop_id:
                        open_loop_ids.discard(loop_id)
                    elif tag == "STORY_BOUNDARY":
                        introduced_heroes.append(b.get("text", "")[:60])
                    b["chunk_index"] = i
                    
                all_blocks.extend(blocks)
                prev_tail = chunk
                
            # Post-process: Story Boundary cluster demotion
            CLUSTER_WINDOW = 20
            boundary_indices = [idx for idx, b in enumerate(all_blocks) if b.get("tag") == "STORY_BOUNDARY"]
            clusters = []
            for idx in boundary_indices:
                if clusters and idx - clusters[-1][-1] <= CLUSTER_WINDOW:
                    clusters[-1].append(idx)
                else:
                    clusters.append([idx])

            demoted = []
            for cluster in clusters:
                if len(cluster) > 1:
                    for idx in cluster[:-1]:
                        all_blocks[idx]["tag"] = "OPEN_LOOP"
                        all_blocks[idx]["_auto_demoted_from"] = "STORY_BOUNDARY"
                        demoted.append(idx)

            if demoted:
                db_add_log(task_id, f"Auto-correction: demoted {len(demoted)} duplicate STORY_BOUNDARY tags to OPEN_LOOP.")
                
            # Save segmented blocks
            db_update_task(task_id, "processing", segmented_blocks=json.dumps(all_blocks, ensure_ascii=False))

        # 3. Tree Builder
        db_add_log(task_id, "Building narrative tree...")
        tree = build_tree(all_blocks)
        annotate_stats(tree)
        db_update_task(task_id, "processing", tree_json=json.dumps(tree, ensure_ascii=False))
        db_add_log(task_id, "Narrative tree built. Extracting simplified structure...")
        
        # 4. Structure Extractor (Simplified)
        structure = extract_structure(tree)
        
        # Final save
        db_update_task(task_id, "completed", structure_json=json.dumps(structure, ensure_ascii=False))
        db_add_log(task_id, "Structure extraction completed successfully.")
        
    except Exception as e:
        err_msg = str(e)
        tb = traceback.format_exc()
        db_add_log(task_id, f"FATAL ERROR during processing: {err_msg}")
        db_add_log(task_id, f"Traceback:\n{tb}")
        db_update_task(task_id, "failed", error_message=err_msg)

# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.post("/upload")
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source: str = Form(""),
    openai_api_key: str = Form(""),
):
    if not (file.filename.endswith('.txt') or file.filename.endswith('.json')):
        raise HTTPException(status_code=400, detail="Only .txt and .json files are supported.")
        
    task_id = uuid.uuid4().hex
    
    if file.filename.endswith('.json'):
        try:
            content_str = file.file.read().decode("utf-8")
        except UnicodeDecodeError:
            file.file.seek(0)
            try:
                content_str = file.file.read().decode("cp1251")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file. Please upload a UTF-8 or CP1251 file.")
                
        try:
            data = json.loads(content_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file.")
            
        if isinstance(data, dict) and "blocks" in data:
            blocks = data["blocks"]
        elif isinstance(data, list):
            blocks = data
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid JSON structure. Must be a list of blocks or an object with a 'blocks' key."
            )
            
        if not isinstance(blocks, list) or not all(isinstance(b, dict) for b in blocks):
            raise HTTPException(status_code=400, detail="JSON data must contain a list of block objects.")
            
        db_create_task(
            task_id,
            file.filename,
            raw_text=None,
            segmented_blocks=json.dumps(blocks, ensure_ascii=False),
            source=source,
        )
        background_tasks.add_task(process_scenario, task_id, None, file.filename, blocks, openai_api_key)
        
    else:
        if not resolve_openai_api_key(openai_api_key):
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key is required for text segmentation. Set OPENAI_API_KEY or enter it in the interface.",
            )
        try:
            content = file.file.read().decode("utf-8")
        except UnicodeDecodeError:
            file.file.seek(0)
            try:
                content = file.file.read().decode("cp1251")
            except UnicodeDecodeError:
                raise HTTPException(status_code=400, detail="Unable to decode file. Please upload a UTF-8 or CP1251 text file.")
                
        db_create_task(task_id, file.filename, raw_text=content, source=source)
        background_tasks.add_task(process_scenario, task_id, content, file.filename, None, openai_api_key)
        
    return {"task_id": task_id}

@app.get("/tasks")
def list_tasks():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, status, created_at, source_note, source_url
            FROM tasks
            ORDER BY created_at DESC
            """
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

@app.patch("/tasks/{task_id}/source")
def update_task_source(task_id: str, payload: SourceUpdate):
    if not db_update_source(task_id, payload.source):
        raise HTTPException(status_code=404, detail="Task not found")
    source_note, source_url = normalize_source(payload.source)
    return {
        "status": "ok",
        "source_note": source_note,
        "source_url": source_url,
    }

@app.patch("/tasks/{task_id}/title")
def update_task_title(task_id: str, payload: TitleUpdate):
    filename = payload.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Название не может быть пустым")
    try:
        updated = db_update_filename(task_id, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": "ok",
        "filename": filename,
    }

@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
            
        task_data = dict(row)
        task_data["progress_log"] = json.loads(task_data["progress_log"]) if task_data["progress_log"] else []
        task_data["segmented_blocks"] = json.loads(task_data["segmented_blocks"]) if task_data["segmented_blocks"] else None
        task_data["tree_json"] = json.loads(task_data["tree_json"]) if task_data["tree_json"] else None
        task_data["structure_json"] = json.loads(task_data["structure_json"]) if task_data["structure_json"] else None
        if task_data["tree_json"]:
            task_data["structure_json"] = extract_structure(task_data["tree_json"])
        
        # Omit raw text in task details to keep network payload smaller unless needed
        task_data.pop("raw_text", None)
        
        return task_data

@app.get("/tasks/{task_id}/download")
def download_structure(task_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, structure_json, tree_json FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        if not row or not row["structure_json"]:
            raise HTTPException(status_code=404, detail="Structure JSON not generated yet or task not found")
            
        filename = row["filename"]
        base_name = os.path.splitext(filename)[0]
        if row["tree_json"]:
            structure_data = extract_structure(json.loads(row["tree_json"]))
        else:
            structure_data = json.loads(row["structure_json"])
        structure_data = strip_structure_blocks(structure_data)
        
        # Save temp file
        temp_filename = f"structure_{task_id}.json"
        with open(temp_filename, "w", encoding="utf-8") as f:
            json.dump(structure_data, f, ensure_ascii=False, indent=2)
            
        return FileResponse(
            temp_filename, 
            media_type="application/json", 
            filename=f"{base_name}_structure.json",
            background=BackgroundTasks() # Runs clean-up after sending
        )

@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "ok"}

# Clean up temporary download files if any remain
@app.middleware("http")
async def db_cleanup_middleware(request, call_next):
    response = await call_next(request)
    # Background cleanup of temporary structure files
    for f in os.listdir("."):
        if f.startswith("structure_") and f.endswith(".json"):
            try:
                # Remove if older than 5 minutes
                age = datetime.datetime.now().timestamp() - os.path.getmtime(f)
                if age > 300:
                    os.remove(f)
            except Exception:
                pass
    return response

# -----------------------------------------------------------------------------
# Frontend View
# -----------------------------------------------------------------------------

TAG_COLORS = {
    "VIDEO_HOOK": "#8b5cf6",
    "VIDEO_PROMISE": "#a855f7",
    "VIDEO_OUTRO": "#7c3aed",
    "STORY_BOUNDARY": "#2563eb",
    "ARC_OPEN": "#0ea5e9",
    "ARC_CLOSE": "#0284c7",
    "TRANSITION_BRIDGE": "#06b6d4",
    "QUESTION_STACK": "#f97316",
    "OPEN_LOOP": "#f59e0b",
    "CLOSE_LOOP": "#eab308",
    "CONTEXT": "#6b7280",
    "FALSE_PROMISE": "#d946ef",
    "TURN": "#ec4899",
    "TENSION": "#ef4444",
    "TENSION_PEAK": "#dc2626",
    "PROOF": "#64748b",
    "MICRO_PAYOFF": "#fb923c",
    "SCENE": "#10b981",
    "RELIEF": "#22c55e",
    "PACING_BEAT": "#84cc16",
    "CTA": "#14b8a6",
    "CUSTOM": "#4b5563",
}

TAG_LEGEND = {
    **TAG_DEFINITIONS,
    "CUSTOM": "Произвольный тег, когда ни один стандартный не подходит.",
}

def strip_structure_blocks(structure: Optional[dict]) -> Optional[dict]:
    """Убирает blocks из structure — они нужны только для UI-подсказок."""
    if not structure:
        return structure
    cleaned = json.loads(json.dumps(structure, ensure_ascii=False))
    for key in ("intro", "outro"):
        section = cleaned.get(key)
        if isinstance(section, dict):
            section.pop("blocks", None)
    for story in cleaned.get("stories") or []:
        if isinstance(story, dict):
            story.pop("blocks", None)
    return cleaned

@app.get("/", response_class=HTMLResponse)
def index():
    openai_key_set = "true" if os.environ.get("OPENAI_API_KEY") else "false"
    tag_legend_json = json.dumps(TAG_LEGEND, ensure_ascii=False)
    tag_color_css = "\n".join(
        f"        .vtag-{tag} {{ background: {color}; }}"
        for tag, color in TAG_COLORS.items()
    )

    html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Экстрактор структуры сценариев</title>
    
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        :root {{
            --bg-base: #080c14;
            --bg-surface: rgba(17, 24, 39, 0.7);
            --bg-surface-hover: rgba(31, 41, 55, 0.8);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-primary: #8b5cf6; /* Violet */
            --accent-primary-glow: rgba(139, 92, 246, 0.4);
            --accent-secondary: #06b6d4; /* Cyan */
            --accent-success: #10b981; /* Green */
            --accent-danger: #ef4444; /* Red */
            --tag-width: 11.5rem;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.1) 0%, transparent 40%);
        }}

        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            background: rgba(8, 12, 20, 0.8);
            backdrop-filter: blur(12px);
            top: 0;
            z-index: 50;
        }}

        .logo {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .api-badge {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
        }}

        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }}

        .dot.active {{
            background-color: var(--accent-success);
            box-shadow: 0 0 8px var(--accent-success);
        }}

        .dot.missing {{
            background-color: var(--accent-danger);
            box-shadow: 0 0 8px var(--accent-danger);
        }}

        .main-container {{
            display: grid;
            grid-template-columns: 320px minmax(0, 1fr);
            flex-grow: 1;
            min-height: calc(100vh - 73px);
            overflow-x: hidden;
        }}

        /* Sidebar History */
        .sidebar {{
            border-right: 1px solid var(--border-color);
            background: rgba(10, 15, 26, 0.5);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}

        .sidebar-header {{
            padding: 1.25rem;
            font-size: 0.95rem;
            font-weight: 600;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-muted);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .history-list {{
            list-style: none;
            overflow-y: auto;
            flex-grow: 1;
            padding: 0.75rem;
        }}

        .history-item {{
            padding: 0.85rem 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }}

        .history-item:hover {{
            background: var(--bg-surface-hover);
            border-color: var(--border-color);
        }}

        .history-item.active {{
            background: rgba(139, 92, 246, 0.1);
            border-color: rgba(139, 92, 246, 0.3);
        }}

        .history-title {{
            font-weight: 500;
            font-size: 0.875rem;
            color: var(--text-main);
            word-break: break-all;
            padding-right: 1.5rem;
        }}

        .history-meta {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-muted);
        }}

        .status-badge {{
            padding: 0.1rem 0.4rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .status-completed {{ background: rgba(16, 185, 129, 0.1); color: var(--accent-success); }}
        .status-processing {{ background: rgba(6, 182, 212, 0.1); color: var(--accent-secondary); }}
        .status-pending {{ background: rgba(107, 114, 128, 0.1); color: var(--text-muted); }}
        .status-failed {{ background: rgba(239, 68, 68, 0.1); color: var(--accent-danger); }}

        .delete-btn {{
            position: absolute;
            top: 0.75rem;
            right: 0.75rem;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            opacity: 0;
            transition: all 0.2s ease;
            font-size: 1rem;
        }}

        .history-item:hover .delete-btn {{
            opacity: 1;
        }}

        .delete-btn:hover {{
            color: var(--accent-danger);
        }}

        /* Content Workspace */
        .workspace {{
            display: flex;
            flex-direction: column;
            padding: 1.5rem 2rem;
            gap: 1.5rem;
            min-width: 0;
            overflow-x: hidden;
        }}

        /* Upload view when no active task */
        .upload-section {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            max-width: 760px;
            margin: auto;
            text-align: center;
            gap: 1.5rem;
        }}

        .drop-zone {{
            width: 100%;
            border: 2px dashed rgba(139, 92, 246, 0.3);
            background: var(--bg-surface);
            border-radius: 16px;
            padding: 3rem 2rem;
            cursor: pointer;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }}

        .drop-zone:hover, .drop-zone.dragover {{
            border-color: var(--accent-secondary);
            background: var(--bg-surface-hover);
            box-shadow: 0 0 30px var(--accent-primary-glow);
            transform: translateY(-2px);
        }}

        .drop-zone svg {{
            width: 48px;
            height: 48px;
            color: var(--accent-primary);
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
        }}

        .drop-zone:hover svg {{
            transform: scale(1.1);
        }}

        .drop-zone h3 {{
            font-size: 1.25rem;
            margin-bottom: 0.5rem;
        }}

        .drop-zone p {{
            font-size: 0.875rem;
            color: var(--text-muted);
        }}

        #fileInput {{
            display: none;
        }}

        .paste-section {{
            width: 100%;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.25rem;
            text-align: left;
        }}

        .paste-section h3 {{
            font-size: 1rem;
            margin-bottom: 0.45rem;
        }}

        .paste-section p {{
            color: var(--text-muted);
            font-size: 0.86rem;
            margin-bottom: 0.85rem;
        }}

        .paste-input {{
            width: 100%;
            min-height: 180px;
            resize: vertical;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            background: rgba(3, 7, 18, 0.72);
            color: var(--text-main);
            font: inherit;
            line-height: 1.45;
            padding: 0.85rem;
            outline: none;
        }}

        .paste-input:focus {{
            border-color: var(--accent-secondary);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.12);
        }}

        .paste-actions {{
            display: flex;
            justify-content: flex-end;
            margin-top: 0.85rem;
        }}

        .drag-overlay {{
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 900;
            background: rgba(8, 12, 20, 0.82);
            border: 3px dashed rgba(6, 182, 212, 0.65);
            backdrop-filter: blur(8px);
            pointer-events: none;
        }}

        .drag-overlay.active {{
            display: flex;
        }}

        .drag-overlay-content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.75rem;
            color: var(--text-main);
            font-size: 1.2rem;
            font-weight: 600;
        }}

        .drag-overlay-content span {{
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 400;
        }}

        /* Details View */
        .details-container {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-width: 0;
        }}

        .details-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--bg-surface);
            padding: 1.25rem 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}

        .details-header h2 {{
            font-size: 1.25rem;
            font-weight: 600;
        }}

        .title-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            min-width: 0;
        }}

        .title-row h2 {{
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .details-actions {{
            display: flex;
            gap: 0.75rem;
        }}

        .btn {{
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-main);
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            border-color: var(--text-muted);
        }}

        .btn-primary {{
            background: var(--accent-primary);
            border-color: transparent;
        }}

        .btn-primary:hover {{
            background: #7c3aed;
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.4);
        }}

        .details-body {{
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-width: 0;
        }}

        /* Console Log Box */
        .console-panel {{
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            height: 200px;
            background: #060910;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }}

        .panel-title {{
            padding: 0.75rem 1rem;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .console-output {{
            flex-grow: 1;
            padding: 1rem;
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            color: #38bdf8; /* light blue */
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
            background: #030712;
        }}

        /* Output & Visualization Panel */
        .output-panel {{
            display: flex;
            flex-direction: column;
            background: var(--bg-surface);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow-x: hidden;
            overflow-y: visible;
            gap: 0px;
            min-width: 0;
            max-width: 100%;
        }}

        .tabs-header {{
            display: flex;
            background: rgba(255, 255, 255, 0.02);
            border-bottom: 1px solid var(--border-color);
        }}

        .tab-btn {{
            padding: 0.75rem 1.25rem;
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            color: var(--text-main);
        }}

        .tab-btn.active {{
            color: var(--accent-secondary);
            border-bottom-color: var(--accent-secondary);
        }}

        .tab-content {{
            padding: 1.25rem;
            display: none;
            min-width: 0;
        }}

        .tab-content.active {{
            display: block;
        }}

        #visualTab {{
            min-width: 0;
            overflow: visible;
        }}

        #visualStructureContent {{
            width: 100%;
            min-width: 0;
            overflow-x: clip;
        }}

        .structure-scroll {{
            display: block;
            width: 100%;
            min-width: 0;
            max-width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
            -webkit-overflow-scrolling: touch;
        }}

        #jsonTab {{
            overflow-x: auto;
            max-height: 70vh;
            overflow-y: auto;
        }}

        pre {{
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            color: #a78bfa;
            white-space: pre-wrap;
            word-break: break-all;
        }}

        .structure-columns {{
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            gap: 1rem;
            width: max-content;
        }}

        .structure-column {{
            flex-shrink: 0;
            width: calc(var(--tag-width) + 2rem);
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
        }}

        .tag-group-title {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.35rem;
        }}

        .tag-group-title-label {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            min-width: 0;
            line-height: 1.2;
        }}

        .btn-copy-column {{
            flex-shrink: 0;
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-muted);
            cursor: pointer;
            font-size: 0.7rem;
            line-height: 1;
            transition: all 0.15s ease;
        }}

        .btn-copy-column:hover {{
            color: var(--text-main);
            border-color: var(--accent-secondary);
            background: rgba(6, 182, 212, 0.1);
        }}

        .tag-flow {{
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            align-items: stretch;
            width: 100%;
        }}

        .visual-tag {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            box-sizing: border-box;
            font-size: 0.7rem;
            font-weight: 500;
            padding: 0.25rem 0.35rem;
            border-radius: 4px;
            color: #ffffff;
            cursor: help;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: transform 0.15s ease;
        }}

        .tag-legend-item .visual-tag {{
            width: var(--tag-width);
            flex-shrink: 0;
        }}

        .visual-tag:hover {{
            transform: scale(1.03);
        }}

        .visual-tag-wrap {{
            position: relative;
            width: 100%;
        }}

        #tag-tooltip-float {{
            display: none;
            position: fixed;
            z-index: 10000;
            max-width: 440px;
            padding: 0.75rem 1rem;
            background: #0f172a;
            border: 1px solid rgba(6, 182, 212, 0.35);
            border-radius: 8px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55);
            font-size: 0.78rem;
            line-height: 1.45;
            pointer-events: none;
        }}

        .tag-tooltip-row {{
            margin-bottom: 0.4rem;
        }}

        .tag-tooltip-row:last-child {{
            margin-bottom: 0;
        }}

        .tag-tooltip-key {{
            color: var(--accent-secondary);
            font-weight: 600;
        }}

        .tag-tooltip-val {{
            color: var(--text-main);
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .tag-legend {{
            margin-bottom: 1.25rem;
            padding: 1rem 1.25rem;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            width: 100%;
            min-width: 0;
            box-sizing: border-box;
        }}

        .tag-legend-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .tag-legend-list {{
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
            margin-top: 0.85rem;
        }}

        .tag-legend-item {{
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            min-width: 0;
        }}

        .tag-legend-desc {{
            font-size: 0.78rem;
            color: var(--text-muted);
            line-height: 1.4;
            padding-top: 0.15rem;
            flex: 1;
            min-width: 0;
            overflow-wrap: anywhere;
        }}

        /* Tag color mapping — уникальный цвет на каждый тег */
{tag_color_css}

        /* Warning banner */
        .warn-banner {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            font-size: 0.875rem;
            margin-bottom: 1.5rem;
        }}

        .warn-banner svg {{
            width: 20px;
            height: 20px;
            color: var(--accent-danger);
            flex-shrink: 0;
        }}

        .api-key-panel {{
            background: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
        }}

        .api-key-panel h3 {{
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }}

        .api-key-panel p {{
            color: var(--text-muted);
            font-size: 0.85rem;
            line-height: 1.45;
            margin-bottom: 0.85rem;
        }}

        .api-key-row {{
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
        }}

        .api-key-input {{
            flex: 1;
            min-width: 240px;
            padding: 0.55rem 0.75rem;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: rgba(3, 7, 18, 0.72);
            color: var(--text-main);
            font: inherit;
            outline: none;
        }}

        .api-key-input:focus {{
            border-color: var(--accent-secondary);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.12);
        }}

        .api-key-hint {{
            margin-top: 0.65rem;
            font-size: 0.78rem;
            color: var(--text-muted);
        }}

        .spinner {{
            width: 1.25rem;
            height: 1.25rem;
            border: 2px solid rgba(255,255,255,0.2);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        .source-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.45rem;
            font-size: 0.82rem;
            color: var(--text-muted);
        }}

        .source-value {{
            min-width: 0;
            max-width: 560px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .source-value a {{
            color: var(--accent-secondary);
            text-decoration: none;
        }}

        .source-value a:hover {{
            text-decoration: underline;
        }}

        .source-edit-btn {{
            border: none;
            background: transparent;
            color: var(--accent-secondary);
            cursor: pointer;
            font-size: 0.78rem;
            padding: 0.1rem 0.2rem;
        }}

        .source-edit-btn:hover {{
            text-decoration: underline;
        }}

        .history-source {{
            font-size: 0.74rem;
            color: var(--text-muted);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            padding-right: 1.5rem;
        }}

        .modal-backdrop {{
            position: fixed;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            background: rgba(3, 7, 18, 0.72);
            backdrop-filter: blur(8px);
            padding: 1.5rem;
        }}

        .modal-backdrop.active {{
            display: flex;
        }}

        .source-modal {{
            width: min(560px, 100%);
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 14px;
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
            padding: 1.25rem;
        }}

        .source-modal h3 {{
            font-size: 1.1rem;
            margin-bottom: 0.45rem;
        }}

        .source-modal p {{
            color: var(--text-muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-bottom: 1rem;
        }}

        .source-modal textarea {{
            width: 100%;
            min-height: 110px;
            resize: vertical;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background: rgba(3, 7, 18, 0.7);
            color: var(--text-main);
            font: inherit;
            padding: 0.75rem;
            outline: none;
        }}

        .source-modal textarea:focus {{
            border-color: var(--accent-secondary);
            box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.12);
        }}

        .source-modal-actions {{
            display: flex;
            justify-content: flex-end;
            gap: 0.75rem;
            margin-top: 1rem;
        }}

        .empty-history {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 150px;
            color: var(--text-muted);
            font-size: 0.875rem;
            text-align: center;
            padding: 1rem;
        }}
    </style>
</head>
<body>

    <header>
        <div class="logo">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="color: var(--accent-primary)">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
            </svg>
            Storytelling Extractor
        </div>
        <div class="api-badge" id="apiKeyBadge">
            <span class="dot { "active" if openai_key_set == "true" else "missing" }" id="apiKeyDot"></span>
            <span id="apiKeyBadgeText">{ "OPENAI_API_KEY Активен" if openai_key_set == "true" else "OPENAI_API_KEY Отсутствует" }</span>
        </div>
    </header>

    <div class="main-container">
        <!-- Sidebar containing list of runs -->
        <aside class="sidebar">
            <div class="sidebar-header">История обработок</div>
            <ul class="history-list" id="historyList">
                <!-- Javascript populated -->
            </ul>
        </aside>

        <!-- Main Workspace -->
        <main class="workspace" id="workspace">
            <!-- Warning if key is missing -->
            { f'''
            <div class="api-key-panel" id="apiKeyPanel">
                <h3>OpenAI API Key</h3>
                <p>Переменная окружения <code>OPENAI_API_KEY</code> не обнаружена. Введите ключ ниже — он сохранится в браузере и будет использоваться для сегментации текста.</p>
                <div class="api-key-row">
                    <input type="password" class="api-key-input" id="openaiApiKeyInput" placeholder="sk-..." autocomplete="off">
                    <button type="button" class="btn btn-primary" onclick="saveOpenAiApiKey()">Сохранить ключ</button>
                    <button type="button" class="btn" onclick="clearOpenAiApiKey()">Очистить</button>
                </div>
                <div class="api-key-hint">Ключ хранится только в sessionStorage этого браузера и не сохраняется на сервере.</div>
            </div>
            ''' if openai_key_set == "false" else "" }

            <!-- Drop Zone / Upload layout (Default) -->
            <div class="upload-section" id="uploadSection">
                <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <svg fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z"></path>
                    </svg>
                    <h3>Загрузить сценарий</h3>
                    <p>Перетащите файл .txt или .json в любое место окна или нажмите для выбора</p>
                    <input type="file" id="fileInput" accept=".txt,.json" onchange="handleFileSelect(event)">
                </div>

                <div class="paste-section">
                    <h3>Или вставьте текст сценария</h3>
                    <p>Вставьте текст ниже и запустите анализ сразу, без выбора файла и дополнительных окон.</p>
                    <textarea class="paste-input" id="pasteTextInput" placeholder="Вставьте сценарий сюда..."></textarea>
                    <div class="paste-actions">
                        <button class="btn btn-primary" onclick="uploadPastedText()">Анализировать текст</button>
                    </div>
                </div>
            </div>

            <!-- Details Layout (Hidden initially) -->
            <div class="details-container" id="detailsContainer" style="display: none;">
                <div class="details-header">
                    <div>
                        <div class="title-row">
                            <h2 id="taskFilename">filename.txt</h2>
                            <button type="button" class="source-edit-btn" onclick="editCurrentTaskTitle()">Редактировать</button>
                        </div>
                        <div style="display: flex; gap: 0.5rem; align-items: center; margin-top: 0.25rem;">
                            <span class="status-badge" id="taskStatus">completed</span>
                            <span style="font-size: 0.8rem; color: var(--text-muted);" id="taskDate">2026-07-01</span>
                        </div>
                        <div class="source-row">
                            <span>Источник:</span>
                            <span class="source-value" id="taskSourceValue">не указан</span>
                            <button type="button" class="source-edit-btn" onclick="editCurrentTaskSource()">Редактировать</button>
                        </div>
                    </div>
                    <div class="details-actions">
                        <button class="btn" onclick="resetWorkspace()">
                            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15"></path></svg>
                            Новый файл
                        </button>
                        <button class="btn btn-primary" id="downloadBtn" style="display: none;" onclick="downloadStructure()">
                            <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                            Скачать JSON
                        </button>
                    </div>
                </div>

                <div class="details-body">
                    <!-- Outputs: Visual representation / JSON Code -->
                    <div class="output-panel">
                        <div class="tabs-header">
                            <button class="tab-btn active" onclick="switchTab('visualTab')">Структура сценария</button>
                            <button class="tab-btn" onclick="switchTab('jsonTab')">Исходный JSON</button>
                        </div>
                        
                        <!-- Visual Tab -->
                        <div class="tab-content active" id="visualTab">
                            <div id="visualStructureContent">
                                <div style="color: var(--text-muted); font-size: 0.9rem; text-align: center; margin-top: 3rem;">
                                    Структура будет отображена после завершения анализа.
                                </div>
                            </div>
                        </div>

                        <!-- JSON Tab -->
                        <div class="tab-content" id="jsonTab">
                            <div style="display: flex; justify-content: flex-end; margin-bottom: 0.5rem;">
                                <button class="btn" style="padding: 0.25rem 0.6rem; font-size: 0.75rem;" onclick="copyJsonToClipboard()">Копировать в буфер</button>
                            </div>
                            <pre><code id="jsonContent">// JSON появится здесь...</code></pre>
                        </div>
                    </div>

                    <!-- Logs Output -->
                    <div class="console-panel">
                        <div class="panel-title">Лог выполнения процесса</div>
                        <div class="console-output" id="consoleOutput">Ожидание логов...</div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <div id="tag-tooltip-float"></div>

    <div class="drag-overlay" id="dragOverlay">
        <div class="drag-overlay-content">
            Отпустите файл, чтобы загрузить
            <span>Поддерживаются .txt и .json</span>
        </div>
    </div>

    <div class="modal-backdrop" id="sourceModal">
        <div class="source-modal">
            <h3 id="sourceModalTitle">Источник видео</h3>
            <p id="sourceModalDescription">Добавьте комментарий или ссылку, откуда взялось видео.</p>
            <textarea id="sourceModalInput" placeholder="Например: https://youtu.be/... или заметка об источнике"></textarea>
            <div class="source-modal-actions">
                <button type="button" class="btn" id="sourceModalCancel">Отмена</button>
                <button type="button" class="btn btn-primary" id="sourceModalConfirm">Начать</button>
            </div>
        </div>
    </div>

    <script>
        let currentTaskId = null;
        let pollInterval = null;
        let activeTab = 'visualTab';
        let currentTaskData = null;
        let sourceModalResolver = null;
        let dragDepth = 0;
        const OPENAI_KEY_FROM_ENV = {"true" if openai_key_set == "true" else "false"};
        const OPENAI_KEY_STORAGE = 'storytelling_openai_api_key';
        const TAG_LEGEND = {tag_legend_json};

        // On Load
        window.addEventListener('DOMContentLoaded', () => {{
            loadHistory();
            setupDragAndDrop();
            setupSourceModal();
            initOpenAiApiKeyUi();
        }});

        function initOpenAiApiKeyUi() {{
            if (OPENAI_KEY_FROM_ENV) return;
            const input = document.getElementById('openaiApiKeyInput');
            const stored = sessionStorage.getItem(OPENAI_KEY_STORAGE);
            if (input && stored) input.value = stored;
            updateApiKeyBadge();
        }}

        function getStoredOpenAiApiKey() {{
            if (OPENAI_KEY_FROM_ENV) return '';
            return (sessionStorage.getItem(OPENAI_KEY_STORAGE) || '').trim();
        }}

        function saveOpenAiApiKey() {{
            const input = document.getElementById('openaiApiKeyInput');
            if (!input) return;
            const key = input.value.trim();
            if (!key) {{
                alert('Введите OpenAI API key.');
                input.focus();
                return;
            }}
            sessionStorage.setItem(OPENAI_KEY_STORAGE, key);
            updateApiKeyBadge();
        }}

        function clearOpenAiApiKey() {{
            sessionStorage.removeItem(OPENAI_KEY_STORAGE);
            const input = document.getElementById('openaiApiKeyInput');
            if (input) input.value = '';
            updateApiKeyBadge();
        }}

        function updateApiKeyBadge() {{
            if (OPENAI_KEY_FROM_ENV) return;
            const dot = document.getElementById('apiKeyDot');
            const text = document.getElementById('apiKeyBadgeText');
            const hasKey = !!getStoredOpenAiApiKey();
            if (!dot || !text) return;
            dot.className = `dot ${{hasKey ? 'active' : 'missing'}}`;
            text.innerText = hasKey ? 'OPENAI_API_KEY в браузере' : 'OPENAI_API_KEY Отсутствует';
        }}

        function appendOpenAiApiKeyToFormData(formData) {{
            if (OPENAI_KEY_FROM_ENV) return;
            const key = getStoredOpenAiApiKey();
            if (key) formData.append('openai_api_key', key);
        }}

        function requireOpenAiApiKeyForTextUpload() {{
            if (OPENAI_KEY_FROM_ENV) return true;
            const key = getStoredOpenAiApiKey();
            if (key) return true;
            alert('Для сегментации текста нужен OpenAI API key. Введите и сохраните ключ в блоке выше.');
            document.getElementById('apiKeyPanel')?.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            document.getElementById('openaiApiKeyInput')?.focus();
            return false;
        }}

        function setupSourceModal() {{
            document.getElementById('sourceModalCancel').addEventListener('click', () => {{
                closeSourceModal(null);
            }});
            document.getElementById('sourceModalConfirm').addEventListener('click', () => {{
                closeSourceModal(document.getElementById('sourceModalInput').value);
            }});
            document.getElementById('sourceModal').addEventListener('click', (e) => {{
                if (e.target.id === 'sourceModal') closeSourceModal(null);
            }});
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape' && document.getElementById('sourceModal').classList.contains('active')) {{
                    closeSourceModal(null);
                }}
            }});
        }}

        function openSourceModal({{ title, description, confirmText, initialValue = '' }}) {{
            document.getElementById('sourceModalTitle').innerText = title;
            document.getElementById('sourceModalDescription').innerText = description;
            document.getElementById('sourceModalConfirm').innerText = confirmText;
            const input = document.getElementById('sourceModalInput');
            input.value = initialValue || '';
            document.getElementById('sourceModal').classList.add('active');
            setTimeout(() => input.focus(), 0);
            return new Promise(resolve => {{
                sourceModalResolver = resolve;
            }});
        }}

        function closeSourceModal(value) {{
            document.getElementById('sourceModal').classList.remove('active');
            if (sourceModalResolver) {{
                sourceModalResolver(value);
                sourceModalResolver = null;
            }}
        }}

        function setupDragAndDrop() {{
            const dropZone = document.getElementById('dropZone');
            const dragOverlay = document.getElementById('dragOverlay');

            const isFileDrag = (e) => {{
                const types = Array.from(e.dataTransfer?.types || []);
                return types.includes('Files');
            }};

            const setDragActive = (active) => {{
                dropZone?.classList.toggle('dragover', active);
                dragOverlay?.classList.toggle('active', active);
            }};

            document.addEventListener('dragenter', (e) => {{
                if (!isFileDrag(e)) return;
                e.preventDefault();
                dragDepth += 1;
                setDragActive(true);
            }});

            document.addEventListener('dragover', (e) => {{
                if (!isFileDrag(e)) return;
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
                setDragActive(true);
            }});

            document.addEventListener('dragleave', (e) => {{
                if (!isFileDrag(e)) return;
                e.preventDefault();
                dragDepth = Math.max(0, dragDepth - 1);
                if (dragDepth === 0) setDragActive(false);
            }});

            document.addEventListener('drop', (e) => {{
                if (!isFileDrag(e)) return;
                e.preventDefault();
                dragDepth = 0;
                setDragActive(false);
                const files = e.dataTransfer?.files || [];
                if (files.length > 0) {{
                    uploadFile(files[0]);
                }}
            }});
        }}

        function handleFileSelect(event) {{
            const files = event.target.files;
            if (files.length > 0) {{
                uploadFile(files[0]);
            }}
        }}

        async function uploadPastedText() {{
            const input = document.getElementById('pasteTextInput');
            const text = input.value.trim();
            if (!text) {{
                alert('Вставьте текст сценария перед запуском анализа.');
                input.focus();
                return;
            }}

            const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
            const file = new File([text], `pasted_text_${{stamp}}.txt`, {{
                type: 'text/plain;charset=utf-8',
            }});
            const uploaded = await uploadFile(file, {{ requestSource: false }});
            if (uploaded) {{
                input.value = '';
            }}
        }}

        // Upload File
        async function uploadFile(file, options = {{}}) {{
            if (!(file.name.endsWith('.txt') || file.name.endsWith('.json'))) {{
                alert('Пожалуйста, выберите файл в формате .txt или .json');
                return false;
            }}

            const isTextUpload = file.name.endsWith('.txt');
            if (isTextUpload && !requireOpenAiApiKeyForTextUpload()) {{
                document.getElementById('fileInput').value = '';
                return false;
            }}

            let source = options.source || '';
            if (options.requestSource !== false) {{
                source = await openSourceModal({{
                    title: 'Источник видео',
                    description: 'Перед стартом добавьте комментарий или ссылку, откуда взялось видео.',
                    confirmText: 'Начать',
                }});
                if (source === null) {{
                    document.getElementById('fileInput').value = '';
                    return false;
                }}
            }}
            source = source || '';

            const formData = new FormData();
            formData.append('file', file);
            formData.append('source', source);
            appendOpenAiApiKeyToFormData(formData);

            // Change UI state
            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('detailsContainer').style.display = 'grid';
            document.getElementById('taskFilename').innerText = file.name;
            document.getElementById('taskStatus').className = 'status-badge status-pending';
            document.getElementById('taskStatus').innerText = 'pending';
            renderTaskSource(source.startsWith('http://') || source.startsWith('https://')
                ? {{ source_url: source, source_note: null }}
                : {{ source_url: null, source_note: source }});
            document.getElementById('consoleOutput').innerText = 'Отправка файла на сервер...\\n';
            document.getElementById('downloadBtn').style.display = 'none';
            document.getElementById('jsonContent').innerText = '// JSON появится здесь...';
            document.getElementById('visualStructureContent').innerHTML = 
                `<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 1rem; color: var(--text-muted); margin-top: 3rem;">
                    <div class="spinner"></div>
                    Выполняется лингвистический анализ структуры сценария...
                 </div>`;

            try {{
                const res = await fetch('/upload', {{
                    method: 'POST',
                    body: formData
                }});
                if (!res.ok) {{
                    const err = await res.json();
                    throw new Error(err.detail || 'Не удалось загрузить файл.');
                }}
                const data = await res.json();
                currentTaskId = data.task_id;
                currentTaskData = {{
                    id: currentTaskId,
                    filename: file.name,
                    source_note: source.startsWith('http://') || source.startsWith('https://') ? null : source,
                    source_url: source.startsWith('http://') || source.startsWith('https://') ? source : null,
                }};
                
                // Add temporary history item
                loadHistory();
                
                // Start polling progress
                startPolling(currentTaskId);
                return true;
            }} catch (error) {{
                document.getElementById('consoleOutput').innerHTML += `\\n[ОШИБКА] ${{error.message}}\\n`;
                document.getElementById('taskStatus').className = 'status-badge status-failed';
                document.getElementById('taskStatus').innerText = 'failed';
                document.getElementById('visualStructureContent').innerHTML = 
                    `<div style="color: var(--accent-danger); text-align: center; margin-top: 3rem;">
                        Ошибка загрузки файла: ${{error.message}}
                     </div>`;
                return false;
            }}
        }}

        // History
        async function loadHistory() {{
            try {{
                const res = await fetch('/tasks');
                const tasks = await res.json();
                const listEl = document.getElementById('historyList');
                listEl.innerHTML = '';
                
                if (tasks.length === 0) {{
                    listEl.innerHTML = `
                        <div class="empty-history">
                            Здесь появятся результаты ваших предыдущих запусков
                        </div>`;
                    return;
                }}

                tasks.forEach(t => {{
                    const item = document.createElement('li');
                    item.className = `history-item ${{t.id === currentTaskId ? 'active' : ''}}`;
                    item.onclick = (e) => {{
                        if (e.target.closest('.delete-btn')) return;
                        selectTask(t.id);
                    }};

                    const formattedDate = new Date(t.created_at).toLocaleString('ru-RU', {{
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                    }});

                    item.innerHTML = `
                        <button class="delete-btn" title="Удалить" onclick="deleteTask('${{t.id}}', event)">&times;</button>
                        <div class="history-title">${{escapeHtml(t.filename)}}</div>
                        ${{renderHistorySource(t)}}
                        <div class="history-meta">
                            <span>${{formattedDate}}</span>
                            <span class="status-badge status-${{t.status}}">${{t.status}}</span>
                        </div>
                    `;
                    listEl.appendChild(item);
                }});
            }} catch (err) {{
                console.error('Ошибка загрузки истории:', err);
            }}
        }}

        // Selection
        async function selectTask(taskId) {{
            currentTaskId = taskId;
            if (pollInterval) clearInterval(pollInterval);
            
            // Mark active in list
            document.querySelectorAll('.history-item').forEach(item => {{
                item.classList.remove('active');
            }});
            loadHistory(); // Reloads classes properly

            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('detailsContainer').style.display = 'grid';

            await fetchAndRenderTask(taskId);
        }}

        async function fetchAndRenderTask(taskId) {{
            try {{
                const res = await fetch(`/tasks/${{taskId}}`);
                if (!res.ok) throw new Error('Не удалось получить данные по задаче.');
                const task = await res.json();
                currentTaskData = task;

                document.getElementById('taskFilename').innerText = task.filename;
                document.getElementById('taskStatus').className = `status-badge status-${{task.status}}`;
                document.getElementById('taskStatus').innerText = task.status;
                
                const formattedDate = new Date(task.created_at).toLocaleString('ru-RU');
                document.getElementById('taskDate').innerText = formattedDate;
                renderTaskSource(task);

                // Logs
                const logBox = document.getElementById('consoleOutput');
                logBox.innerText = task.progress_log.join('\\n');
                logBox.scrollTop = logBox.scrollHeight;

                if (task.status === 'processing' || task.status === 'pending') {{
                    document.getElementById('downloadBtn').style.display = 'none';
                    document.getElementById('jsonContent').innerText = '// Идет обработка, подождите...';
                    document.getElementById('visualStructureContent').innerHTML = 
                        `<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 1rem; color: var(--text-muted); margin-top: 3rem;">
                            <div class="spinner"></div>
                            Идет разметка сценария...
                         </div>`;
                    startPolling(taskId);
                }} else if (task.status === 'completed') {{
                    document.getElementById('downloadBtn').style.display = 'flex';
                    const jsonStr = JSON.stringify(task.structure_json, null, 2);
                    document.getElementById('jsonContent').innerText = jsonStr;
                    
                    renderVisualStructure(task.structure_json, task.segmented_blocks, task.tree_json);
                }} else {{
                    document.getElementById('downloadBtn').style.display = 'none';
                    document.getElementById('jsonContent').innerText = `// Ошибка выполнения:\\n// ${{task.error_message}}`;
                    document.getElementById('visualStructureContent').innerHTML = 
                        `<div style="color: var(--accent-danger); text-align: center; margin-top: 3rem; font-weight: 500;">
                            Ошибка обработки сценария:<br>
                            <span style="font-size: 0.85rem; color: var(--text-muted); font-family: monospace; display: block; margin-top: 0.5rem;">${{task.error_message}}</span>
                         </div>`;
                }}
            }} catch (error) {{
                console.error(error);
            }}
        }}

        // Polling
        function startPolling(taskId) {{
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(async () => {{
                try {{
                    const res = await fetch(`/tasks/${{taskId}}`);
                    const task = await res.json();
                    
                    // Update logs and status
                    const logBox = document.getElementById('consoleOutput');
                    logBox.innerText = task.progress_log.join('\\n');
                    logBox.scrollTop = logBox.scrollHeight;
                    
                    document.getElementById('taskStatus').className = `status-badge status-${{task.status}}`;
                    document.getElementById('taskStatus').innerText = task.status;

                    if (task.status !== 'processing' && task.status !== 'pending') {{
                        clearInterval(pollInterval);
                        pollInterval = null;
                        loadHistory();
                        fetchAndRenderTask(taskId);
                    }}
                }} catch (err) {{
                    console.error('Ошибка опроса статуса:', err);
                }}
            }}, 1200);
        }}

        // Delete Task
        async function deleteTask(taskId, event) {{
            event.stopPropagation();
            if (!confirm('Вы уверены, что хотите удалить эту запись?')) return;
            try {{
                const res = await fetch(`/tasks/${{taskId}}`, {{ method: 'DELETE' }});
                if (res.ok) {{
                    if (currentTaskId === taskId) {{
                        resetWorkspace();
                    }}
                    loadHistory();
                }}
            }} catch (err) {{
                console.error(err);
            }}
        }}

        // Reset
        function resetWorkspace() {{
            currentTaskId = null;
            currentTaskData = null;
            if (pollInterval) clearInterval(pollInterval);
            
            document.getElementById('detailsContainer').style.display = 'none';
            document.getElementById('uploadSection').style.display = 'flex';
            document.getElementById('fileInput').value = '';
            
            document.querySelectorAll('.history-item').forEach(item => {{
                item.classList.remove('active');
            }});
        }}

        function getSourceValue(task) {{
            if (!task) return '';
            return task.source_url || task.source_note || '';
        }}

        function renderSourceHtml(task) {{
            if (!task || (!task.source_url && !task.source_note)) {{
                return '<span style="color: var(--text-muted);">не указан</span>';
            }}
            if (task.source_url) {{
                const url = escapeHtml(task.source_url);
                return `<a href="${{url}}" target="_blank" rel="noopener noreferrer">${{url}}</a>`;
            }}
            return escapeHtml(task.source_note);
        }}

        function renderHistorySource(task) {{
            const value = getSourceValue(task);
            if (!value) return '';
            return `<div class="history-source">${{escapeHtml(value)}}</div>`;
        }}

        function renderTaskSource(task) {{
            document.getElementById('taskSourceValue').innerHTML = renderSourceHtml(task);
        }}

        async function editCurrentTaskSource() {{
            if (!currentTaskData || !currentTaskData.id) return;
            const source = await openSourceModal({{
                title: 'Редактировать источник',
                description: 'Измените комментарий или ссылку, откуда взялось видео.',
                confirmText: 'Сохранить',
                initialValue: getSourceValue(currentTaskData),
            }});
            if (source === null) return;

            try {{
                const res = await fetch(`/tasks/${{currentTaskData.id}}/source`, {{
                    method: 'PATCH',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ source }}),
                }});
                if (!res.ok) {{
                    const err = await res.json();
                    throw new Error(err.detail || 'Не удалось сохранить источник.');
                }}
                const data = await res.json();
                currentTaskData.source_note = data.source_note;
                currentTaskData.source_url = data.source_url;
                renderTaskSource(currentTaskData);
                loadHistory();
            }} catch (err) {{
                alert(err.message);
            }}
        }}

        async function editCurrentTaskTitle() {{
            if (!currentTaskData || !currentTaskData.id) return;
            const filename = await openSourceModal({{
                title: 'Редактировать название',
                description: 'Введите новое название для этой обработки.',
                confirmText: 'Сохранить',
                initialValue: currentTaskData.filename || '',
            }});
            if (filename === null) return;

            const trimmed = filename.trim();
            if (!trimmed) {{
                alert('Название не может быть пустым.');
                return;
            }}

            try {{
                const res = await fetch(`/tasks/${{currentTaskData.id}}/title`, {{
                    method: 'PATCH',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ filename: trimmed }}),
                }});
                if (!res.ok) {{
                    const err = await res.json();
                    throw new Error(err.detail || 'Не удалось сохранить название.');
                }}
                const data = await res.json();
                currentTaskData.filename = data.filename;
                document.getElementById('taskFilename').innerText = data.filename;
                loadHistory();
            }} catch (err) {{
                alert(err.message);
            }}
        }}

        function renderTagLegend() {{
            const items = Object.entries(TAG_LEGEND).map(([tag, desc]) => `
                <div class="tag-legend-item">
                    <span class="visual-tag vtag-${{tag}}" title="${{tag}}">${{tag}}</span>
                    <span class="tag-legend-desc">${{desc}}</span>
                </div>
            `).join('');
            return `
                <div class="tag-legend">
                    <div class="tag-legend-title">Справка по тегам</div>
                    <div class="tag-legend-list">${{items}}</div>
                </div>
            `;
        }}

        function renderColumnTitle(label, iconSvg) {{
            return `
                <div class="tag-group-title">
                    <span class="tag-group-title-label">
                        ${{iconSvg}}
                        ${{label}}
                    </span>
                    <button type="button" class="btn-copy-column" onclick="copyColumnTags(this)" title="Копировать теги колонки">⎘</button>
                </div>
            `;
        }}

        function copyColumnTags(btn) {{
            const column = btn.closest('.structure-column');
            if (!column) return;
            const tags = [...column.querySelectorAll('.visual-tag')].map(el => el.textContent.trim());
            const text = tags.join('\\n');
            navigator.clipboard.writeText(text).then(() => {{
                const prev = btn.textContent;
                btn.textContent = '✓';
                setTimeout(() => {{ btn.textContent = prev; }}, 1200);
            }}).catch(err => {{
                alert('Не удалось скопировать: ' + err);
            }});
        }}

        // Rendering visual tags list
        const beatTooltipStore = {{}};

        function escapeHtml(str) {{
            if (str === null || str === undefined) return '—';
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;');
        }}

        function walkTreeBeats(node) {{
            const beats = [];
            for (const child of node.children || []) {{
                if (child.type === 'beat') beats.push(child);
                else beats.push(...walkTreeBeats(child));
            }}
            return beats;
        }}

        function blocksFromTreeBeats(beats, allBlocks) {{
            return beats.map(b => {{
                const block = allBlocks[b.index];
                return block ? block : {{
                    text: b.text || '',
                    tag: b.tag || '',
                    tension_score: b.tension_score ?? null,
                    loop_id: b.loop_id ?? null,
                    reasoning: b.reasoning || '',
                    chunk_index: b.chunk_index ?? null,
                }};
            }});
        }}

        function enrichStructureWithBlocks(structure, allBlocks, tree) {{
            if (!structure) return structure;
            if (!allBlocks || !allBlocks.length) return structure;

            const enriched = JSON.parse(JSON.stringify(structure));

            if (tree && tree.children) {{
                const introBeats = [];
                for (const child of tree.children) {{
                    if (child.type === 'main_story') break;
                    if (child.type === 'beat') introBeats.push(child);
                    else introBeats.push(...walkTreeBeats(child));
                }}
                if (enriched.intro) {{
                    enriched.intro.blocks = blocksFromTreeBeats(introBeats, allBlocks);
                }}

                const storyNodes = tree.children.filter(c => c.type === 'main_story');
                if (enriched.stories) {{
                    enriched.stories = enriched.stories.map((story, i) => ({{
                        ...story,
                        blocks: blocksFromTreeBeats(walkTreeBeats(storyNodes[i] || {{ children: [] }}), allBlocks),
                    }}));
                }}

                let lastStoryIdx = -1;
                tree.children.forEach((child, i) => {{
                    if (child.type === 'main_story') lastStoryIdx = i;
                }});
                const outroBeats = [];
                if (lastStoryIdx >= 0) {{
                    for (const child of tree.children.slice(lastStoryIdx + 1)) {{
                        if (child.type === 'beat') outroBeats.push(child);
                        else outroBeats.push(...walkTreeBeats(child));
                    }}
                }}
                if (enriched.outro) {{
                    enriched.outro.blocks = blocksFromTreeBeats(outroBeats, allBlocks);
                }}
                return enriched;
            }}

            // fallback без tree_json — последовательная нарезка (может расходиться между историями)
            let offset = 0;
            const take = (n) => {{
                const slice = allBlocks.slice(offset, offset + n);
                offset += n;
                return slice;
            }};
            if (enriched.intro && enriched.intro.tag_pattern) {{
                enriched.intro.blocks = take(enriched.intro.tag_pattern.length);
            }}
            if (enriched.stories) {{
                enriched.stories = enriched.stories.map(story => {{
                    const blocks = story.tag_pattern ? take(story.tag_pattern.length) : [];
                    return {{ ...story, blocks }};
                }});
            }}
            if (enriched.outro && enriched.outro.tag_pattern) {{
                enriched.outro.blocks = take(enriched.outro.tag_pattern.length);
            }}
            return enriched;
        }}

        function formatBeatTooltipHtml(block) {{
            const fields = [
                ['text', block.text],
                ['tag', block.tag],
                ['tension_score', block.tension_score],
                ['loop_id', block.loop_id],
                ['reasoning', block.reasoning],
                ['chunk_index', block.chunk_index],
            ];
            return fields.map(([key, val]) => `
                <div class="tag-tooltip-row">
                    <span class="tag-tooltip-key">${{key}}:</span>
                    <span class="tag-tooltip-val">${{escapeHtml(val)}}</span>
                </div>
            `).join('');
        }}

        function positionFloatTooltip(anchorEl) {{
            const tip = document.getElementById('tag-tooltip-float');
            if (!tip) return;
            tip.style.display = 'block';
            const margin = 8;
            const rect = anchorEl.getBoundingClientRect();
            let left = rect.left;
            let top = rect.bottom + margin;
            if (left + tip.offsetWidth > window.innerWidth - margin) {{
                left = window.innerWidth - tip.offsetWidth - margin;
            }}
            if (top + tip.offsetHeight > window.innerHeight - margin) {{
                top = rect.top - tip.offsetHeight - margin;
            }}
            if (top < margin) top = margin;
            tip.style.left = `${{left}}px`;
            tip.style.top = `${{top}}px`;
        }}

        function showBeatTooltip(beatId, anchorEl) {{
            const block = beatTooltipStore[beatId];
            const tip = document.getElementById('tag-tooltip-float');
            if (!block || !tip) return;
            tip.innerHTML = formatBeatTooltipHtml(block);
            positionFloatTooltip(anchorEl);
        }}

        function hideBeatTooltip() {{
            const tip = document.getElementById('tag-tooltip-float');
            if (tip) tip.style.display = 'none';
        }}

        function bindBeatTooltips() {{
            document.querySelectorAll('.visual-tag-wrap[data-beat-id]').forEach(wrap => {{
                wrap.addEventListener('mouseenter', () => showBeatTooltip(wrap.dataset.beatId, wrap));
                wrap.addEventListener('mouseleave', hideBeatTooltip);
            }});
        }}

        function renderVisualStructure(structure, segmentedBlocks, treeJson) {{
            const container = document.getElementById('visualStructureContent');
            if (!structure) {{
                container.innerHTML = 'Нет данных';
                return;
            }}

            Object.keys(beatTooltipStore).forEach(k => delete beatTooltipStore[k]);
            let beatIdCounter = 0;
            structure = enrichStructureWithBlocks(structure, segmentedBlocks, treeJson);

            const introIcon = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15.75 9V5.25A2.25 2.25 0 0013.5 3h-6a2.25 2.25 0 00-2.25 2.25v13.5A2.25 2.25 0 007.5 21h6a2.25 2.25 0 002.25-2.25V15M12 9l-3 3m0 0l3 3m-3-3h12.75"></path></svg>';
            const storyIcon = '<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"></path></svg>';
            const outroIcon = introIcon;

            const columns = [];

            if (structure.intro && structure.intro.tag_pattern && structure.intro.tag_pattern.length > 0) {{
                columns.push(`
                <div class="structure-column">
                    ${{renderColumnTitle('Вступление (Intro)', introIcon)}}
                    <div class="tag-flow">
                        ${{renderTagsList(structure.intro.tag_pattern, structure.intro.blocks, beatIdCounter)}}
                    </div>
                </div>`);
                beatIdCounter += structure.intro.tag_pattern.length;
            }}

            if (structure.stories && structure.stories.length > 0) {{
                structure.stories.forEach((story, idx) => {{
                    const listHtml = renderTagsList(story.tag_pattern, story.blocks, beatIdCounter);
                    beatIdCounter += (story.tag_pattern || []).length;
                    columns.push(`
                    <div class="structure-column">
                        ${{renderColumnTitle('История #' + (idx + 1), storyIcon)}}
                        <div class="tag-flow">
                            ${{listHtml}}
                        </div>
                    </div>`);
                }});
            }}

            if (structure.outro && structure.outro.tag_pattern && structure.outro.tag_pattern.length > 0) {{
                columns.push(`
                <div class="structure-column">
                    ${{renderColumnTitle('Концовка (Outro)', outroIcon)}}
                    <div class="tag-flow">
                        ${{renderTagsList(structure.outro.tag_pattern, structure.outro.blocks, beatIdCounter)}}
                    </div>
                </div>`);
            }}

            container.innerHTML = columns.length > 0
                ? renderTagLegend() + `<div class="structure-scroll"><div class="structure-columns">${{columns.join('')}}</div></div>`
                : '<span style="color: var(--text-muted); font-size: 0.9rem;">Нет данных для отображения</span>';

            bindBeatTooltips();
        }}

        function renderTagsList(tagPattern, blocks, idOffset = 0) {{
            if (!tagPattern || tagPattern.length === 0) return '<span style="color: var(--text-muted); font-size: 0.8rem;">Нет тегов</span>';
            return tagPattern.map((tag, idx) => {{
                const block = (blocks && blocks[idx]) ? blocks[idx] : {{
                    text: '',
                    tag: tag,
                    tension_score: null,
                    loop_id: null,
                    reasoning: '',
                    chunk_index: null,
                }};
                const beatId = `beat-${{idOffset + idx}}`;
                beatTooltipStore[beatId] = block;
                const cleanTag = tag.startsWith('CUSTOM:') ? 'CUSTOM' : tag;
                return `
                    <div class="visual-tag-wrap" data-beat-id="${{beatId}}">
                        <span class="visual-tag vtag-${{cleanTag}}">${{escapeHtml(tag)}}</span>
                    </div>
                `;
            }}).join('');
        }}

        // Copy JSON helper
        function copyJsonToClipboard() {{
            const codeText = document.getElementById('jsonContent').innerText;
            navigator.clipboard.writeText(codeText).then(() => {{
                alert('JSON скопирован в буфер обмена!');
            }}).catch(err => {{
                alert('Не удалось скопировать JSON: ' + err);
            }});
        }}

        // Download JSON
        function downloadStructure() {{
            if (currentTaskId) {{
                window.location.href = `/tasks/${{currentTaskId}}/download`;
            }}
        }}

        // Tabs switching
        function switchTab(tabId) {{
            activeTab = tabId;
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Find active tab trigger and content
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }}
    </script>
</body>
</html>
    """
    return html_content

# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Listen on localhost:8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
