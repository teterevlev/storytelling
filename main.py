import os
import sqlite3
import json
import uuid
import datetime
import traceback
from pathlib import Path
from typing import Optional
from threading import Thread

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

# Import pipeline modules
from pipeline.taxonomy import TAG_DEFINITIONS
from pipeline.content_lang import detect_content_lang, normalize_content_lang, resolve_content_lang
from pipeline.chunker import chunk_text
from pipeline.segmenter import segment_text, build_chunk_context_note
from pipeline.tree_builder import build_tree, annotate_stats
from pipeline.structure_extractor import extract_structure
from pipeline.subcluster_seed import seed_tag_subclusters, demote_domain_leaky_subclusters
from pipeline.subcluster_classifier import (
    classify_unassigned_blocks,
    classify_task_unassigned_blocks,
    load_active_subclusters,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = "db.sqlite"
LOCALE_DIR = os.path.join(BASE_DIR, "locales")
AVAILABLE_LOCALES = ("en", "ru")
# Pragmatic canon from one more clustering pass (see PRAGMATIC_FREEZE.md).
CLUSTERS_REVIEW_PATH = os.path.join(
    BASE_DIR, "subclusters", "clusters_canon.md"
)

# In-memory classify job status: tag -> dict
_classify_jobs: dict[str, dict] = {}


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


class ContentLangUpdate(BaseModel):
    content_lang: str


class BlockSubclustersUpdate(BaseModel):
    primary_subcluster_id: Optional[int] = None
    secondary_subcluster_ids: list[int] = []
    source: str = "manual"
    notes: Optional[str] = None


class ClassifyRequest(BaseModel):
    openai_api_key: str = ""


BLOCK_CORE_KEYS = frozenset({
    "tag", "text", "reasoning", "loop_id", "tension_score", "chunk_index",
})

# -----------------------------------------------------------------------------
# Database Setup and Helpers
# -----------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _block_row_values(task_id: str, position: int, block: dict) -> tuple:
    meta = {k: v for k, v in block.items() if k not in BLOCK_CORE_KEYS}
    return (
        task_id,
        position,
        (block.get("tag") or "").strip() or None,
        block.get("text"),
        block.get("reasoning"),
        block.get("loop_id"),
        block.get("tension_score"),
        block.get("chunk_index"),
        json.dumps(meta, ensure_ascii=False) if meta else None,
    )


def db_replace_blocks(conn: sqlite3.Connection, task_id: str, blocks: list) -> None:
    """Replace all blocks for a task. Assignments cascade-delete with blocks."""
    conn.execute("DELETE FROM blocks WHERE task_id = ?", (task_id,))
    if not blocks:
        return
    rows = [
        _block_row_values(task_id, i, b)
        for i, b in enumerate(blocks)
        if isinstance(b, dict)
    ]
    conn.executemany(
        """
        INSERT INTO blocks
            (task_id, position, tag, text, reasoning, loop_id, tension_score, chunk_index, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def db_migrate_segmented_blocks(conn: sqlite3.Connection) -> None:
    """Idempotent: unpack tasks.segmented_blocks into blocks when missing."""
    cursor = conn.execute(
        """
        SELECT t.id, t.segmented_blocks
        FROM tasks t
        WHERE t.segmented_blocks IS NOT NULL AND t.segmented_blocks != ''
          AND NOT EXISTS (SELECT 1 FROM blocks b WHERE b.task_id = t.id)
        """
    )
    for row in cursor.fetchall():
        try:
            blocks = json.loads(row["segmented_blocks"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(blocks, list):
            continue
        db_replace_blocks(conn, row["id"], blocks)


def db_init():
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                filename TEXT,
                status TEXT,           -- 'pending', 'processing', 'completed', 'failed'
                progress_log TEXT,     -- JSON array of strings
                error_message TEXT,
                raw_text TEXT,
                segmented_blocks TEXT, -- JSON string (legacy mirror)
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
        if "content_lang" not in existing_columns:
            conn.execute(
                "ALTER TABLE tasks ADD COLUMN content_lang TEXT NOT NULL DEFAULT 'ru'"
            )
            for row in conn.execute("SELECT id, raw_text FROM tasks").fetchall():
                lang = detect_content_lang(row["raw_text"], default="ru")
                conn.execute(
                    "UPDATE tasks SET content_lang = ? WHERE id = ?",
                    (lang, row["id"]),
                )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                tag TEXT,
                text TEXT,
                reasoning TEXT,
                loop_id TEXT,
                tension_score REAL,
                chunk_index INTEGER,
                meta_json TEXT,
                UNIQUE(task_id, position)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tag_subclusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_tag TEXT NOT NULL,
                slug TEXT NOT NULL,
                name TEXT NOT NULL,
                formula TEXT,
                abstract TEXT,
                notes TEXT,
                examples_json TEXT,
                sort_order INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                UNIQUE(parent_tag, slug)
            )
        """)
        sc_cols = {row["name"] for row in conn.execute("PRAGMA table_info(tag_subclusters)").fetchall()}
        if "notes" not in sc_cols:
            conn.execute("ALTER TABLE tag_subclusters ADD COLUMN notes TEXT")
        if "examples_json" not in sc_cols:
            conn.execute("ALTER TABLE tag_subclusters ADD COLUMN examples_json TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS block_subcluster_assignments (
                block_id INTEGER NOT NULL REFERENCES blocks(id) ON DELETE CASCADE,
                subcluster_id INTEGER NOT NULL REFERENCES tag_subclusters(id) ON DELETE CASCADE,
                is_primary INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'manual',
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (block_id, subcluster_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_tag ON blocks(tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_blocks_task_position ON blocks(task_id, position)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tag_subclusters_parent ON tag_subclusters(parent_tag)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assignments_subcluster "
            "ON block_subcluster_assignments(subcluster_id)"
        )
        db_migrate_segmented_blocks(conn)
        seed_tag_subclusters(conn, Path(CLUSTERS_REVIEW_PATH))
        demote_domain_leaky_subclusters(conn)
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
    content_lang: Optional[str] = None,
):
    log = [f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Task created. Waiting to start..."]
    source_note, source_url = normalize_source(source)
    sample_text = raw_text
    if not sample_text and segmented_blocks:
        try:
            blocks = json.loads(segmented_blocks)
            if isinstance(blocks, list):
                sample_text = "\n".join(
                    (b.get("text") or "") for b in blocks[:30] if isinstance(b, dict)
                )
        except (TypeError, json.JSONDecodeError):
            sample_text = None
    lang = resolve_content_lang(content_lang, sample_text)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO tasks
                (id, filename, status, progress_log, raw_text, segmented_blocks,
                 source_note, source_url, content_lang)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                lang,
            )
        )
        if segmented_blocks:
            try:
                blocks = json.loads(segmented_blocks)
            except (TypeError, json.JSONDecodeError):
                blocks = None
            if isinstance(blocks, list):
                db_replace_blocks(conn, task_id, blocks)
        conn.commit()


def db_update_content_lang(task_id: str, content_lang: str) -> bool:
    lang = normalize_content_lang(content_lang)
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE tasks SET content_lang = ? WHERE id = ?",
            (lang, task_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def db_get_content_lang(task_id: str) -> str:
    with get_db() as conn:
        row = conn.execute(
            "SELECT content_lang FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return "ru"
        return normalize_content_lang(row["content_lang"])


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
        if segmented_blocks is not None:
            try:
                blocks = json.loads(segmented_blocks)
            except (TypeError, json.JSONDecodeError):
                blocks = []
            if isinstance(blocks, list):
                db_replace_blocks(conn, task_id, blocks)
            else:
                db_replace_blocks(conn, task_id, [])
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
            content_lang = db_get_content_lang(task_id)
            db_add_log(task_id, f"Script language: {content_lang}")
            
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
                    context_note = build_chunk_context_note(
                        content_lang, prev_tail, open_loop_ids, introduced_heroes
                    )
                
                result = segment_text(
                    client, chunk, context_note, content_lang=content_lang
                )
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
    content_lang: str = Form(""),
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
            content_lang=content_lang,
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
                
        db_create_task(
            task_id,
            file.filename,
            raw_text=content,
            source=source,
            content_lang=content_lang,
        )
        background_tasks.add_task(process_scenario, task_id, content, file.filename, None, openai_api_key)
        
    return {"task_id": task_id, "content_lang": db_get_content_lang(task_id)}

@app.get("/tasks")
def list_tasks():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, status, created_at, source_note, source_url, content_lang
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


@app.patch("/tasks/{task_id}/content-lang")
def update_task_content_lang(task_id: str, payload: ContentLangUpdate):
    lang = normalize_content_lang(payload.content_lang)
    if lang not in ("en", "ru"):
        raise HTTPException(status_code=400, detail="content_lang must be 'en' or 'ru'")
    if not db_update_content_lang(task_id, lang):
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "ok", "content_lang": lang}

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


@app.get("/tags/guide")
def get_tags_guide():
    """Tag + subcluster reference from DB for the guide view."""
    groups: dict[str, list] = {}
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, parent_tag, slug, name, formula, abstract, notes,
                   examples_json, sort_order, status
            FROM tag_subclusters
            WHERE status = 'active'
            ORDER BY parent_tag ASC, sort_order ASC, id ASC
            """
        ).fetchall()
        for row in rows:
            d = dict(row)
            try:
                examples = json.loads(d.pop("examples_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                d.pop("examples_json", None)
                examples = []
            if not isinstance(examples, list):
                examples = []
            d["examples"] = examples[:2]
            groups.setdefault(d["parent_tag"], []).append(d)

    # Prefer taxonomy order, then any extra parent tags from DB
    ordered_tags = [t for t in TAG_DEFINITIONS.keys() if t in groups]
    for tag in groups:
        if tag not in ordered_tags:
            ordered_tags.append(tag)

    return {
        "tags": ordered_tags,
        "definitions": {t: TAG_DEFINITIONS.get(t, "") for t in ordered_tags},
        "groups": {t: groups[t] for t in ordered_tags},
    }


@app.get("/tags/reference")
def get_tag_reference():
    """Aggregate block texts by tag across all tasks (from normalized blocks table)."""
    examples: dict[str, list] = {tag: [] for tag in TAG_COLORS}
    subclusters_by_tag: dict[str, list] = {}
    with get_db() as conn:
        for row in conn.execute(
            """
            SELECT id, parent_tag, slug, name, formula, abstract, notes,
                   examples_json, sort_order, status
            FROM tag_subclusters
            ORDER BY parent_tag ASC, sort_order ASC, id ASC
            """
        ).fetchall():
            d = dict(row)
            try:
                d["examples"] = json.loads(d.pop("examples_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                d.pop("examples_json", None)
                d["examples"] = []
            subclusters_by_tag.setdefault(d["parent_tag"], []).append(d)

        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                b.id AS block_id,
                b.tag,
                b.text,
                b.reasoning,
                b.loop_id,
                b.tension_score,
                b.task_id,
                t.filename,
                t.source_note,
                t.source_url,
                s.sort_order AS subcluster_sort_order,
                s.name AS subcluster_name,
                s.slug AS subcluster_slug,
                s.id AS subcluster_id
            FROM blocks b
            JOIN tasks t ON t.id = b.task_id
            LEFT JOIN block_subcluster_assignments a
                ON a.block_id = b.id AND a.is_primary = 1
            LEFT JOIN tag_subclusters s ON s.id = a.subcluster_id
            WHERE b.tag IS NOT NULL AND b.tag != ''
              AND b.text IS NOT NULL AND TRIM(b.text) != ''
            ORDER BY t.created_at DESC, b.position ASC
            """
        )
        for row in cursor.fetchall():
            tag = (row["tag"] or "").strip()
            text = (row["text"] or "").strip()
            if not tag or not text:
                continue
            if tag not in examples:
                examples[tag] = []
            examples[tag].append({
                "block_id": row["block_id"],
                "text": text,
                "task_id": row["task_id"],
                "filename": row["filename"],
                "source_note": row["source_note"],
                "source_url": row["source_url"],
                "reasoning": row["reasoning"] or "",
                "loop_id": row["loop_id"],
                "tension_score": row["tension_score"],
                "subcluster_sort_order": row["subcluster_sort_order"],
                "subcluster_name": row["subcluster_name"],
                "subcluster_slug": row["subcluster_slug"],
                "subcluster_id": row["subcluster_id"],
            })
    counts = {tag: len(items) for tag, items in examples.items()}
    # Include tags that only appear as CUSTOM:* etc.
    all_tags = list(dict.fromkeys(list(TAG_COLORS.keys()) + list(examples.keys()) + list(subclusters_by_tag.keys())))
    return {
        "tags": all_tags,
        "counts": counts,
        "examples": examples,
        "subclusters": subclusters_by_tag,
    }


@app.get("/tags/{tag}/subclusters")
def list_tag_subclusters(tag: str):
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, parent_tag, slug, name, formula, abstract, notes,
                   examples_json, sort_order, status
            FROM tag_subclusters
            WHERE parent_tag = ?
            ORDER BY
                CASE WHEN status = 'residual' THEN 1 ELSE 0 END,
                sort_order ASC,
                id ASC
            """,
            (tag,),
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            try:
                d["examples"] = json.loads(d.pop("examples_json") or "[]")
            except (TypeError, json.JSONDecodeError):
                d.pop("examples_json", None)
                d["examples"] = []
            out.append(d)
        return out


def _classify_counts(conn, tag: str, task_id: Optional[str] = None) -> dict:
    params: list = [tag]
    task_clause = ""
    if task_id:
        task_clause = "AND b.task_id = ?"
        params.append(task_id)
    total = conn.execute(
        f"SELECT COUNT(*) AS c FROM blocks b WHERE b.tag = ? {task_clause}",
        params,
    ).fetchone()["c"]
    assigned = conn.execute(
        f"""
        SELECT COUNT(DISTINCT b.id) AS c
        FROM blocks b
        JOIN block_subcluster_assignments a ON a.block_id = b.id
        WHERE b.tag = ? {task_clause}
        """,
        params,
    ).fetchone()["c"]
    return {
        "total": total,
        "assigned": assigned,
        "pending": max(0, total - assigned),
    }


def _task_classify_counts(conn, task_id: str) -> dict:
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM blocks WHERE task_id = ?", (task_id,)
    ).fetchone()["c"]
    assigned = conn.execute(
        """
        SELECT COUNT(DISTINCT b.id) AS c
        FROM blocks b
        JOIN block_subcluster_assignments a ON a.block_id = b.id
        WHERE b.task_id = ?
        """,
        (task_id,),
    ).fetchone()["c"]
    return {
        "total": total,
        "assigned": assigned,
        "pending": max(0, total - assigned),
    }


def _run_classify_job(tag: str, api_key: str):
    job = _classify_jobs.setdefault(f"tag:{tag}", {})
    job.update({"running": True, "error": None, "assigned_now": 0})
    try:
        client = OpenAI(api_key=api_key)

        def on_batch(assigned, pending):
            job["assigned_now"] = assigned
            job["batch_pending"] = pending
            with get_db() as conn:
                job.update(_classify_counts(conn, tag))

        with get_db() as conn:
            job.update(_classify_counts(conn, tag))
            result = classify_unassigned_blocks(
                client, conn, tag, on_batch_done=on_batch
            )
            job["assigned_now"] = result.get("assigned") or 0
            if result.get("error"):
                job["error"] = result["error"]
            job.update(_classify_counts(conn, tag))
    except Exception as exc:
        job["error"] = str(exc)
    finally:
        job["running"] = False


def _run_task_classify_job(task_id: str, api_key: str):
    job_key = f"task:{task_id}"
    job = _classify_jobs.setdefault(job_key, {})
    job.update({"running": True, "error": None, "assigned_now": 0, "current_tag": None})
    try:
        client = OpenAI(api_key=api_key)

        def on_progress(tag, assigned, pending):
            job["current_tag"] = tag
            job["assigned_now"] = (job.get("assigned_now") or 0)
            with get_db() as conn:
                counts = _task_classify_counts(conn, task_id)
                job.update(counts)

        with get_db() as conn:
            job.update(_task_classify_counts(conn, task_id))
            result = classify_task_unassigned_blocks(
                client, conn, task_id, on_progress=on_progress
            )
            job["assigned_now"] = result.get("assigned") or 0
            if result.get("error"):
                job["error"] = result["error"]
            job.update(_task_classify_counts(conn, task_id))
            job["current_tag"] = None
    except Exception as exc:
        job["error"] = str(exc)
    finally:
        job["running"] = False


@app.post("/tags/{tag}/classify")
def start_tag_classify(tag: str, payload: Optional[ClassifyRequest] = None):
    payload = payload or ClassifyRequest()
    api_key = resolve_openai_api_key(payload.openai_api_key)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required. Set OPENAI_API_KEY or enter it in the interface.",
        )
    job_key = f"tag:{tag}"
    existing = _classify_jobs.get(job_key) or {}
    if existing.get("running"):
        with get_db() as conn:
            counts = _classify_counts(conn, tag)
        return {"status": "already_running", **counts, "running": True}

    with get_db() as conn:
        active = load_active_subclusters(conn, tag)
        if not active:
            raise HTTPException(
                status_code=400,
                detail=f"No active subclusters seeded for tag {tag}",
            )
        counts = _classify_counts(conn, tag)
        if counts["pending"] == 0:
            return {"status": "nothing_to_do", **counts, "running": False}

    _classify_jobs[job_key] = {**counts, "running": True, "error": None, "assigned_now": 0}
    Thread(target=_run_classify_job, args=(tag, api_key), daemon=True).start()
    return {"status": "started", **counts, "running": True}


@app.get("/tags/{tag}/classify/status")
def tag_classify_status(tag: str):
    with get_db() as conn:
        counts = _classify_counts(conn, tag)
    job = _classify_jobs.get(f"tag:{tag}") or {}
    return {
        **counts,
        "running": bool(job.get("running")),
        "error": job.get("error"),
        "assigned_now": job.get("assigned_now", 0),
    }


@app.get("/tasks/{task_id}/subclusters")
def get_task_subclusters(task_id: str):
    """Lightweight per-block subcluster info for one task (loaded on demand)."""
    with get_db() as conn:
        task = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        rows = conn.execute(
            """
            SELECT
                b.position,
                b.tag,
                s.sort_order AS subcluster_sort_order,
                s.name AS subcluster_name,
                s.formula AS subcluster_formula,
                s.abstract AS subcluster_abstract,
                s.slug AS subcluster_slug
            FROM blocks b
            LEFT JOIN block_subcluster_assignments a
                ON a.block_id = b.id AND a.is_primary = 1
            LEFT JOIN tag_subclusters s ON s.id = a.subcluster_id
            WHERE b.task_id = ?
            ORDER BY b.position ASC
            """,
            (task_id,),
        ).fetchall()
        return {
            "task_id": task_id,
            "blocks": [dict(row) for row in rows],
        }


@app.post("/tasks/{task_id}/classify")
def start_task_classify(task_id: str, payload: Optional[ClassifyRequest] = None):
    payload = payload or ClassifyRequest()
    api_key = resolve_openai_api_key(payload.openai_api_key)
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required. Set OPENAI_API_KEY or enter it in the interface.",
        )
    with get_db() as conn:
        task = conn.execute("SELECT id, status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        counts = _task_classify_counts(conn, task_id)

    job_key = f"task:{task_id}"
    existing = _classify_jobs.get(job_key) or {}
    if existing.get("running"):
        return {"status": "already_running", **counts, "running": True}
    if counts["pending"] == 0:
        return {"status": "nothing_to_do", **counts, "running": False}

    _classify_jobs[job_key] = {**counts, "running": True, "error": None, "assigned_now": 0}
    Thread(target=_run_task_classify_job, args=(task_id, api_key), daemon=True).start()
    return {"status": "started", **counts, "running": True}


@app.get("/tasks/{task_id}/classify/status")
def task_classify_status(task_id: str):
    with get_db() as conn:
        task = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        counts = _task_classify_counts(conn, task_id)
    job = _classify_jobs.get(f"task:{task_id}") or {}
    return {
        **counts,
        "running": bool(job.get("running")),
        "error": job.get("error"),
        "assigned_now": job.get("assigned_now", 0),
        "current_tag": job.get("current_tag"),
    }


@app.get("/tags/{tag}/blocks")
def list_tag_blocks(tag: str, unassigned: int = 0):
    query = """
        SELECT
            b.id,
            b.task_id,
            b.position,
            b.tag,
            b.text,
            b.reasoning,
            b.loop_id,
            b.tension_score,
            b.chunk_index,
            t.filename,
            t.source_note,
            t.source_url
        FROM blocks b
        JOIN tasks t ON t.id = b.task_id
        WHERE b.tag = ?
    """
    params: list = [tag]
    if unassigned:
        query += """
            AND NOT EXISTS (
                SELECT 1 FROM block_subcluster_assignments a WHERE a.block_id = b.id
            )
        """
    query += " ORDER BY t.created_at DESC, b.position ASC"
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


@app.put("/blocks/{block_id}/subclusters")
def update_block_subclusters(block_id: int, payload: BlockSubclustersUpdate):
    source = (payload.source or "manual").strip() or "manual"
    if source not in ("manual", "import", "llm"):
        raise HTTPException(status_code=400, detail="source must be manual, import, or llm")

    primary_id = payload.primary_subcluster_id
    secondary_ids = list(payload.secondary_subcluster_ids or [])
    if primary_id is not None and primary_id in secondary_ids:
        secondary_ids = [sid for sid in secondary_ids if sid != primary_id]

    with get_db() as conn:
        block = conn.execute("SELECT id, tag FROM blocks WHERE id = ?", (block_id,)).fetchone()
        if not block:
            raise HTTPException(status_code=404, detail="Block not found")

        wanted_ids = ([primary_id] if primary_id is not None else []) + secondary_ids
        if wanted_ids:
            placeholders = ",".join("?" * len(wanted_ids))
            found = conn.execute(
                f"SELECT id, parent_tag FROM tag_subclusters WHERE id IN ({placeholders})",
                wanted_ids,
            ).fetchall()
            if len(found) != len(set(wanted_ids)):
                raise HTTPException(status_code=400, detail="Unknown subcluster id")
            parent_tag = block["tag"]
            for row in found:
                if row["parent_tag"] != parent_tag:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Subcluster {row['id']} belongs to {row['parent_tag']}, not {parent_tag}",
                    )

        conn.execute(
            "DELETE FROM block_subcluster_assignments WHERE block_id = ?",
            (block_id,),
        )
        for sid in wanted_ids:
            conn.execute(
                """
                INSERT INTO block_subcluster_assignments
                    (block_id, subcluster_id, is_primary, source, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    block_id,
                    sid,
                    1 if sid == primary_id else 0,
                    source,
                    payload.notes,
                ),
            )
        conn.commit()

        rows = conn.execute(
            """
            SELECT a.subcluster_id, a.is_primary, a.source, a.notes,
                   s.slug, s.name, s.parent_tag
            FROM block_subcluster_assignments a
            JOIN tag_subclusters s ON s.id = a.subcluster_id
            WHERE a.block_id = ?
            ORDER BY a.is_primary DESC, a.subcluster_id ASC
            """,
            (block_id,),
        ).fetchall()
        return {
            "status": "ok",
            "block_id": block_id,
            "assignments": [dict(row) for row in rows],
        }


@app.get("/", response_class=HTMLResponse)
def index():
    openai_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    tag_color_css = "\n".join(
        f"        .vtag-{tag} {{ background: {color}; }}"
        for tag, color in TAG_COLORS.items()
    )
    tag_list_json = json.dumps(list(TAG_COLORS.keys()), ensure_ascii=False)
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return (
        html.replace("__TAG_COLOR_CSS__", tag_color_css)
        .replace("__TAG_LIST_JSON__", tag_list_json)
        .replace("__API_KEY_DOT_CLASS__", "active" if openai_key_set else "missing")
        .replace("__OPENAI_KEY_FROM_ENV__", "true" if openai_key_set else "false")
    )


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Listen on localhost:8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
