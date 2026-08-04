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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = "db.sqlite"
LOCALE_DIR = os.path.join(BASE_DIR, "locales")
AVAILABLE_LOCALES = ("en", "ru")


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

@app.get("/locales/{locale}.json")
def get_locale(locale: str):
    if locale not in AVAILABLE_LOCALES:
        raise HTTPException(status_code=404, detail="Locale not found.")
    locale_path = os.path.join(LOCALE_DIR, f"{locale}.json")
    if not os.path.exists(locale_path):
        raise HTTPException(status_code=404, detail="Locale file not found.")
    with open(locale_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))


@app.get("/", response_class=HTMLResponse)
def index():
    openai_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    tag_color_css = "\n".join(
        f"        .vtag-{tag} {{ background: {color}; }}"
        for tag, color in TAG_COLORS.items()
    )
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html.replace("__TAG_COLOR_CSS__", tag_color_css).replace(
        "__API_KEY_DOT_CLASS__", "active" if openai_key_set else "missing"
    ).replace(
        "__OPENAI_KEY_FROM_ENV__", "true" if openai_key_set else "false"
    )


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Listen on localhost:8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
