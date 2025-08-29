# # apps-v0/whys-v0-worker/main.py
# import os, io, json, uuid, time, math
# from typing import List, Tuple, Dict, Any

# import boto3
# import redis
# import psycopg
# import numpy as np

# # ASR + audio utils
# from faster_whisper import WhisperModel
# import librosa

# # Optional diarization (only if HUGGINGFACE_TOKEN is provided)
# try:
#     from pyannote.audio import Pipeline as PyannotePipeline
#     _HAS_PYANNOTE = True
# except Exception:
#     _HAS_PYANNOTE = False

# # Embeddings + LLM (OpenAI-compatible)
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()

# # ----------------------------
# # Env & clients
# # ----------------------------
# DB_URL = os.environ["DATABASE_URL"]                # e.g., postgres://whys:whys@localhost:5432/whys
# REDIS_URL = os.environ["REDIS_URL"]                # e.g., redis://localhost:6379
# S3_REGION = os.environ["S3_REGION"]                # e.g., us-east-1
# S3_BUCKET = os.environ["S3_BUCKET"]                # e.g., whys-v0-yourbucket
# S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
# S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]

# DEVICE = os.environ.get("DEVICE", "cpu")           # "cpu" | "cuda"
# COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
# WHISPER_DETECT_MODEL = os.environ.get("WHISPER_DETECT_MODEL", "small")
# WHISPER_EN_MODEL = os.environ.get("WHISPER_EN_MODEL", "h2oai/faster-whisper-large-v3-turbo")
# WHISPER_OTHER_MODEL = os.environ.get("WHISPER_OTHER_MODEL", "medium")
# LANG_CONF_THR = float(os.environ.get("LANG_CONF_THR", "0.85"))

# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
# SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "gpt-5-mini")

# HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")  # optional (enables diarization)
# AUDIO_KEEP_HOURS = int(os.environ.get("AUDIO_KEEP_HOURS", "72"))
# print("loading complete")
# print(HUGGINGFACE_TOKEN)
# print(OPENAI_API_KEY)
# print(SUMMARY_MODEL)
# # Redis / S3 / LLM
# r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
# print("Redis connected : ",r)
# # print("job :",r.brpop("whys:jobs", timeout=10))
# s3 = boto3.client(
#     "s3",
#     region_name=S3_REGION,
#     aws_access_key_id=S3_ACCESS_KEY,
#     aws_secret_access_key=S3_SECRET_KEY,
# )
# client = OpenAI(api_key=OPENAI_API_KEY)

# # ----------------------------
# # SQL
# # ----------------------------
# SQL_INSERT_RECORDING = (
#     "INSERT INTO recordings (id, user_id, s3_key, status) "
#     "VALUES (%s, %s, %s, %s) "
#     "ON CONFLICT (id) DO UPDATE SET status = EXCLUDED.status"
# )
# SQL_UPDATE_RECORDING = "UPDATE recordings SET status = %s WHERE id = %s"

# SQL_INSERT_TRANSCRIPT = (
#     "INSERT INTO transcripts (id, recording_id, user_id, lang, text, words_json) "
#     "VALUES (%s, %s, %s, %s, %s, %s)"
# )

# SQL_INSERT_CHUNK = (
#     "INSERT INTO transcript_chunks (transcript_id, idx, text, embedding) "
#     "VALUES (%s, %s, %s, %s)"
# )

# SQL_INSERT_MEMORY = (
#     "INSERT INTO memories (id, user_id, transcript_id, summary, action_items, tags, flags) "
#     "VALUES (%s, %s, %s, %s, %s, %s, %s)"
# )
# print("Sql complete")

# # ----------------------------
# # Diarization pipeline (optional)
# # ----------------------------
# _pyannote_pipe = None
# def _load_diarization():
#     global _pyannote_pipe
#     if not _HAS_PYANNOTE or not HUGGINGFACE_TOKEN:
#         return None
#     if _pyannote_pipe is None:
#         _pyannote_pipe = PyannotePipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1",
#             use_auth_token=HUGGINGFACE_TOKEN,
#         )
#         print("Diarization model loaded")
#     return _pyannote_pipe

# def diarize(path: str) -> List[Tuple[float, float, str]]:
#     """
#     Returns list of (start, end, speaker) diarization segments.
#     If pyannote or token is unavailable, returns empty [].
#     """
#     pipe = _load_diarization()
#     if pipe is None:
#         return []
#     diarization = pipe(path)
#     out = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         out.append((float(turn.start), float(turn.end), str(speaker)))
#     out.sort(key=lambda x: x[0])
#     print("Diarization complete:")
#     return out

# # ----------------------------
# # Language detection + ASR routing
# # ----------------------------
# def detect_language_quick(path: str, seconds: int = 12) -> Tuple[str, float]:
#     y, sr = librosa.load(path, sr=16000, mono=True)
#     y = y[: int(seconds * 16000)]
#     det_model = WhisperModel(WHISPER_DETECT_MODEL, device=DEVICE, compute_type="int8")
#     segs, info = det_model.transcribe(y, language=None, beam_size=1, temperature=0.0, vad_filter=True)
#     print("Language detection complete")
#     return info.language, float(getattr(info, "language_probability", 0.0))
    

# def transcribe_routed(path: str) -> Dict[str, Any]:
#     lang, prob = detect_language_quick(path)
#     model_id = WHISPER_EN_MODEL if (lang == "en" and prob >= LANG_CONF_THR) else WHISPER_OTHER_MODEL

#     asr = WhisperModel(model_id, device=DEVICE, compute_type=COMPUTE_TYPE)
#     segments, info = asr.transcribe(
#         path,
#         task="translate",        # English output for non-English too
#         language=None,           # auto-detect
#         vad_filter=True,
#         beam_size=5,
#         temperature=0.0,
#         word_timestamps=True     # required for word->speaker alignment
#     )
#     seg_list = list(segments)

#     # Convert segments (generator objects) into JSON-friendly structure
#     words_json = []
#     for s in seg_list:
#         wlist = []
#         if getattr(s, "words", None):
#             for w in s.words:
#                 if w.start is None or w.end is None: 
#                     continue
#                 wlist.append({
#                     "start": float(w.start),
#                     "end": float(w.end),
#                     "word": w.word,
#                     "prob": float(getattr(w, "probability", 0.0)) if hasattr(w, "probability") else None
#                 })
#         words_json.append({
#             "start": float(s.start),
#             "end": float(s.end),
#             "text": s.text,
#             "words": wlist
#         })

#     full_text = "".join(s["text"] for s in words_json)
#     print("ASR complete")
#     return {
#         "model_used": model_id,
#         "det_lang": lang,
#         "det_prob": prob,
#         "segments_json": words_json,
#         "text": full_text
#     }

# # ----------------------------
# # Word→speaker alignment
# # ----------------------------
# def _find_speaker_at(t: float, spk_segs: List[Tuple[float,float,str]], last_idx: int = 0) -> Tuple[str,int]:
#     i = last_idx
#     n = len(spk_segs)
#     while i < n and spk_segs[i][1] < t:
#         i += 1
#     for j in (i-1, i):
#         if 0 <= j < n:
#             s, e, spk = spk_segs[j]
#             if s <= t <= e:
#                 return spk, j
#     j = min(max(i, 0), n-1) if n else 0
#     print("find speaker at complete ")
#     return (spk_segs[j][2] if n else "S?"), j

# def align_words_to_speakers(spk_segs: List[Tuple[float,float,str]], segments_json: List[Dict[str,Any]], max_gap: float = 0.6):
#     """
#     Assign each word to the active speaker at its midpoint; fuse words into speaker turns.
#     Returns: List of dicts {start, end, speaker, text, overlap?}
#     """
#     if not spk_segs:
#         # No diarization: single-speaker S1
#         turns = []
#         for s in segments_json:
#             turns.append({
#                 "start": s["start"],
#                 "end": s["end"],
#                 "speaker": "S1",
#                 "text": s["text"].strip(),
#                 "overlap": False
#             })
#         return turns

#     words = []
#     last_idx = 0
#     for seg in segments_json:
#         wlist = seg.get("words") or []
#         if not wlist:
#             mid = (seg["start"] + seg["end"]) / 2.0
#             spk, last_idx = _find_speaker_at(mid, spk_segs, last_idx)
#             words.append({"start": seg["start"], "end": seg["end"], "word": seg["text"].strip(), "speaker": spk})
#             continue
#         for w in wlist:
#             if w["start"] is None or w["end"] is None:
#                 continue
#             mid = (w["start"] + w["end"]) / 2.0
#             spk, last_idx = _find_speaker_at(mid, spk_segs, last_idx)
#             words.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"], "speaker": spk})

#     # Fuse to turns
#     turns = []
#     cur = None
#     for w in words:
#         if cur is None or w["speaker"] != cur["speaker"] or (w["start"] - cur["end"]) > max_gap:
#             if cur:
#                 turns.append(cur)
#             cur = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "text": w["word"], "overlap": False}
#         else:
#             cur["end"] = w["end"]
#             cur["text"] += (" " + w["word"])
#     if cur:
#         turns.append(cur)

#     print("Alignment complete")
#     return turns

# # ----------------------------
# # Embedding + summarization
# # ----------------------------
# def embed_text(text: str) -> List[float]:
#     return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

# BASIC_PROMPT = """Summarize the following multi-speaker transcript.

# Rules:
# - Speakers are labeled (e.g., S1, S2). If attribution is uncertain, say so.
# - Be concise and factual. Do not invent details.
# - Keep it under ~180 words unless crucial details require more.

# Return:
# 1) Global summary: 5–8 bullets covering the main points.
# 2) Per-speaker notes: for each speaker (S1, S2…), 2–4 bullets on stance/requests/commitments.
# 3) Action items: owner → task → due date (if mentioned).
# 4) Open questions or decisions made.

# Transcript:
# """

# def turns_to_text(turns: List[Dict[str,Any]]) -> str:
#     lines = []
#     for t in turns:
#         start = t.get("start", 0.0)
#         end = t.get("end", 0.0)
#         spk = t.get("speaker", "S?")
#         txt = (t.get("text") or "").strip().replace("\n", " ")
#         ov = t.get("overlap", False)
#         lines.append(f"[{start:.2f}-{end:.2f}] {spk}{' (overlap)' if ov else ''}: {txt}")
#         print("Turns to text complete")
#     return "\n".join(lines)

# def summarize(turns: List[Dict[str,Any]]) -> str:
#     transcript = turns_to_text(turns)
#     msg = BASIC_PROMPT + transcript
#     resp = client.chat.completions.create(
#         model=SUMMARY_MODEL,
#         messages=[
#             {"role": "system", "content": "You are a concise, accurate meeting summarizer."},
#             {"role": "user", "content": msg},
#         ],
#         temperature=0.2,
#     )
#     print("Summarization complete")
#     return resp.choices[0].message.content

# # ----------------------------
# # S3 helpers
# # ----------------------------
# def s3_download_to_bytes(key: str) -> bytes:
#     buf = io.BytesIO()
#     s3.download_fileobj(S3_BUCKET, key, buf)
#     print("S3 download complete")
#     return buf.getvalue()

# # ----------------------------
# # Persist helpers
# # ----------------------------
# def chunk_text(text: str, max_words: int = 600) -> List[str]:
#     out, cur = [], []
#     for s in text.split(". "):
#         if sum(len(x.split()) for x in cur) + len(s.split()) > max_words:
#             out.append(". ".join(cur).strip()); cur = []
#         cur.append(s)
#     if cur: out.append(". ".join(cur).strip())
#     print("Text chunking complete")
#     return [c for c in out if c]

# # ----------------------------
# # Worker loop
# # ----------------------------
# def process_job(job: Dict[str, Any]):
#     job_id = job.get("job_id", str(uuid.uuid4()))
#     user_id = uuid.UUID(job["user_id"])
#     key = job["s3_key"]
#     print("process_job complete ")

#     with psycopg.connect(DB_URL, autocommit=True) as conn:
#         conn.execute(SQL_INSERT_RECORDING, (uuid.UUID(job_id), user_id, key, "processing"))

#     # 1) Fetch audio (bytes on memory)
#     audio_bytes = s3_download_to_bytes(key)
#     tmp_path = f"/tmp/{uuid.uuid4()}.m4a"
#     with open(tmp_path, "wb") as f:
#         f.write(audio_bytes)

#     # 2) Diarize (optional) + transcribe (routed by language)
#     spk_segs = diarize(tmp_path)
#     asr = transcribe_routed(tmp_path)
#     turns = align_words_to_speakers(spk_segs, asr["segments_json"])

#     # 3) Persist transcript + chunks + memory
#     transcript_id = uuid.uuid4()
#     with psycopg.connect(DB_URL, autocommit=True) as conn:
#         # transcripts
#         conn.execute(
#             SQL_INSERT_TRANSCRIPT,
#             (
#                 transcript_id,
#                 uuid.UUID(job_id),
#                 user_id,
#                 asr["det_lang"],
#                 asr["text"],
#                 json.dumps(asr["segments_json"]),
#             ),
#         )
#         # chunks + embeddings
#         for idx, ch in enumerate(chunk_text(asr["text"])):
#             vec = embed_text(ch)
#             conn.execute(SQL_INSERT_CHUNK, (transcript_id, idx, ch, vec))

#         # summary/coaching
#         summary_text = summarize(turns)
#         memory_id = uuid.uuid4()
#         conn.execute(
#             SQL_INSERT_MEMORY,
#             (
#                 memory_id,
#                 user_id,
#                 transcript_id,
#                 summary_text,
#                 json.dumps([]),                  # action_items (v0: parse later)
#                 json.dumps([]),                  # tags (v0: add tone later)
#                 json.dumps([]),                  # flags (v0: contradictions/sensitive later)
#             ),
#         )
#         conn.execute(SQL_UPDATE_RECORDING, ("done", uuid.UUID(job_id)))

#     # 4) (Optional) S3 lifecycle handles deletion of raw audio after N hours (bucket rule)
#     try:
#         os.remove(tmp_path)
#     except Exception:
#         pass

#     print(f"[OK] job={job_id} transcript={transcript_id}")

# def main():
#     print("Worker online. Queue: whys:jobs")
#     while True:
#         print("Waiting for job...")
#         try:
#             popped = r.brpop("whys:jobs", timeout=10)
#             print("brpop complete ")
#             if not popped:
#                 continue
#             _, payload = popped
#             job = json.loads(payload)
#             # print("Job received:", job)
#             process_job(job)
#         except Exception as e:
#             print("ERROR:", e)
#             time.sleep(1)

# if __name__ == "__main__":
#     main()


# apps-v0/whys-v0-worker/main.py
import os, io, json, uuid, time
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")
import boto3
import redis
import psycopg
import numpy as np

# ASR + audio utils
from faster_whisper import WhisperModel
import librosa

# Optional diarization (only if HUGGINGFACE_TOKEN is provided)
try:
    from pyannote.audio import Pipeline as PyannotePipeline
    _HAS_PYANNOTE = True
except Exception:
    _HAS_PYANNOTE = False

# Embeddings + LLM (OpenAI-compatible) – used for summary only here
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Env & clients
# ----------------------------
DB_URL = os.environ["DATABASE_URL"]                # e.g., postgres://user:pass@pooler.neon.tech/db?sslmode=require
REDIS_URL = os.environ["REDIS_URL"]                # e.g., redis://localhost:6379
S3_REGION = os.environ["S3_REGION"]                # e.g., us-east-1
S3_BUCKET = os.environ["S3_BUCKET"]                # e.g., whys-v0-yourbucket
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]

DEVICE = os.environ.get("DEVICE", "cpu")           # "cpu" | "cuda"
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
WHISPER_DETECT_MODEL = os.environ.get("WHISPER_DETECT_MODEL", "small")
WHISPER_EN_MODEL = os.environ.get("WHISPER_EN_MODEL", "h2oai/faster-whisper-large-v3-turbo")
WHISPER_OTHER_MODEL = os.environ.get("WHISPER_OTHER_MODEL", "medium")
LANG_CONF_THR = float(os.environ.get("LANG_CONF_THR", "0.85"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "gpt-5-mini")

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")  # optional (enables diarization)
# print("loading complete")
# print(HUGGINGFACE_TOKEN)
# print(OPENAI_API_KEY)
# print(SUMMARY_MODEL)

# Redis / S3 / LLM
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
# print("Redis connected:", r)

s3 = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
)
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# SQL (Neon schema)
# ----------------------------
SQL_UPSERT_SESSION_FROM_JOB = """
INSERT INTO sessions (id, user_id, title, started_at, audio_file_url, status, kind)
VALUES (%s, %s, %s, %s, %s, 'processing', 'batch')
ON CONFLICT (id) DO UPDATE
SET audio_file_url = EXCLUDED.audio_file_url,
    status = 'processing'
RETURNING id
"""

SQL_UPDATE_SESSION_READY = """
UPDATE sessions
SET status = 'ready', ended_at = now()
WHERE id = %s
"""

SQL_FIND_FILE_BY_S3 = "SELECT id FROM files WHERE s3_key = %s LIMIT 1"
SQL_UPDATE_FILE_STATUS = "UPDATE files SET status = 'processed', updated_at = now() WHERE id = %s"

SQL_INSERT_ASR_RUN = """
INSERT INTO asr_runs (id, session_id, kind, status, params, started_at)
VALUES (%s, %s, 'batch_whisper', 'running', %s, now())
"""
SQL_COMPLETE_ASR_RUN = """
UPDATE asr_runs SET status = 'completed', finished_at = now() WHERE id = %s
"""
SQL_FAIL_ASR_RUN = """
UPDATE asr_runs SET status = 'failed', finished_at = now() WHERE id = %s
"""

SQL_INSERT_TRANSCRIPT = """
INSERT INTO transcripts (id, session_id, text, language, word_count)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (session_id) DO UPDATE
SET text = EXCLUDED.text,
    language = EXCLUDED.language,
    word_count = EXCLUDED.word_count,
    created_at = transcripts.created_at
RETURNING id
"""

SQL_INSERT_SEGMENT = """
INSERT INTO segments (id, session_id, start_time, end_time, speaker_label, text)
VALUES (%s, %s, %s, %s, %s, %s)
"""

SQL_INSERT_INSIGHT = """
INSERT INTO insights (id, session_id, type, content)
VALUES (%s, %s, %s, %s)
"""

SQL_INSERT_JOB_IF_PRESENT = """
INSERT INTO jobs (id, session_id, type, status, started_at)
VALUES (%s, %s, 'transcription', 'running', now())
ON CONFLICT (id) DO UPDATE SET status = 'running'
"""
SQL_COMPLETE_JOB = "UPDATE jobs SET status = 'completed', finished_at = now() WHERE id = %s"
SQL_FAIL_JOB = "UPDATE jobs SET status = 'failed', finished_at = now() WHERE id = %s"

# print("Sql complete")

# ----------------------------
# Diarization pipeline (optional)
# ----------------------------
_pyannote_pipe = None
# def _load_diarization():
#     global _pyannote_pipe
#     if not _HAS_PYANNOTE or not HUGGINGFACE_TOKEN:
#         return None
#     if _pyannote_pipe is None:
#         _pyannote_pipe = PyannotePipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1",
#             use_auth_token=HUGGINGFACE_TOKEN,
#         )
#         # print("Diarization model loaded")
#     return _pyannote_pipe
def _load_diarization():
    global _pyannote_pipe
    if not _HAS_PYANNOTE:
        print("PyAnnote not available - install with: pip install pyannote.audio")
        return None
    if not HUGGINGFACE_TOKEN:
        print("HUGGINGFACE_TOKEN not set - diarization disabled")
        return None
    if _pyannote_pipe is None:
        try:
            print("Loading diarization model...")
            _pyannote_pipe = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HUGGINGFACE_TOKEN,
            )
            print("Diarization model loaded successfully")
        except Exception as e:
            print(f"Failed to load diarization model: {e}")
            return None
    return _pyannote_pipe

# def diarize(path: str) -> List[Tuple[float, float, str]]:
#     pipe = _load_diarization()
#     if pipe is None:
#         return []
#     diarization = pipe(path)
#     out = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         out.append((float(turn.start), float(turn.end), str(speaker)))
#     out.sort(key=lambda x: x[0])
#     # print("Diarization complete")
#     return out
def diarize(path: str) -> List[Tuple[float, float, str]]:
    pipe = _load_diarization()
    if pipe is None:
        print("WARNING: Diarization not available - missing token or pyannote")
        return []
    
    try:
        print(f"Starting diarization for: {path}")
        diarization = pipe(path)
        out = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            out.append((float(turn.start), float(turn.end), str(speaker)))
        out.sort(key=lambda x: x[0])
        print(f"Diarization complete - found {len(out)} speaker segments")
        for i, (start, end, spk) in enumerate(out):
            print(f"  Segment {i}: {start:.2f}-{end:.2f}s = {spk}")
        return out
    except Exception as e:
        print(f"Diarization failed with error: {e}")
        return []

# ----------------------------
# Language detection + ASR routing (your logic)
# ----------------------------
def detect_language_quick(path: str, seconds: int = 12) -> Tuple[str, float]:
    y, sr = librosa.load(path, sr=16000, mono=True)
    y = y[: int(seconds * 16000)]
    det_model = WhisperModel(WHISPER_DETECT_MODEL, device=DEVICE, compute_type="int8")
    segs, info = det_model.transcribe(y, language=None, beam_size=1, temperature=0.0, vad_filter=True)
    # print("Language detection complete")
    return info.language, float(getattr(info, "language_probability", 0.0))

def transcribe_routed(path: str) -> Dict[str, Any]:
    lang, prob = detect_language_quick(path)
    model_id = WHISPER_EN_MODEL if (lang == "en" and prob >= LANG_CONF_THR) else WHISPER_OTHER_MODEL

    asr = WhisperModel(model_id, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, info = asr.transcribe(
        path,
        task="translate",
        language=None,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
        word_timestamps=True
    )
    seg_list = list(segments)

    words_json = []
    for s in seg_list:
        wlist = []
        if getattr(s, "words", None):
            for w in s.words:
                if w.start is None or w.end is None:
                    continue
                wlist.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word,
                    "prob": float(getattr(w, "probability", 0.0)) if hasattr(w, "probability") else None
                })
        words_json.append({
            "start": float(s.start),
            "end": float(s.end),
            "text": s.text,
            "words": wlist
        })

    full_text = "".join(s["text"] for s in words_json)
    # print("ASR complete")
    return {
        "model_used": model_id,
        "det_lang": lang,
        "det_prob": prob,
        "segments_json": words_json,
        "text": full_text
    }

# ----------------------------
# Word→speaker alignment
# ----------------------------
def _find_speaker_at(t: float, spk_segs: List[Tuple[float,float,str]], last_idx: int = 0) -> Tuple[str,int]:
    i = last_idx
    n = len(spk_segs)
    while i < n and spk_segs[i][1] < t:
        i += 1
    for j in (i-1, i):
        if 0 <= j < n:
            s, e, spk = spk_segs[j]
            if s <= t <= e:
                return spk, j
    j = min(max(i, 0), n-1) if n else 0
    return (spk_segs[j][2] if n else "S?"), j

def align_words_to_speakers(spk_segs: List[Tuple[float,float,str]], segments_json: List[Dict[str,Any]], max_gap: float = 0.6):
    if not spk_segs:
        return [{
            "start": s["start"],
            "end": s["end"],
            "speaker": "S1",
            "text": s["text"].strip(),
            "overlap": False
        } for s in segments_json]

    words = []
    last_idx = 0
    for seg in segments_json:
        wlist = seg.get("words") or []
        if not wlist:
            mid = (seg["start"] + seg["end"]) / 2.0
            spk, last_idx = _find_speaker_at(mid, spk_segs, last_idx)
            words.append({"start": seg["start"], "end": seg["end"], "word": seg["text"].strip(), "speaker": spk})
            continue
        for w in wlist:
            if w["start"] is None or w["end"] is None:
                continue
            mid = (w["start"] + w["end"]) / 2.0
            spk, last_idx = _find_speaker_at(mid, spk_segs, last_idx)
            words.append({"start": float(w["start"]), "end": float(w["end"]), "word": w["word"], "speaker": spk})

    turns = []
    cur = None
    for w in words:
        if cur is None or w["speaker"] != cur["speaker"] or (w["start"] - cur["end"]) > max_gap:
            if cur: turns.append(cur)
            cur = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "text": w["word"], "overlap": False}
        else:
            cur["end"] = w["end"]
            cur["text"] += (" " + w["word"])
    if cur: turns.append(cur)
    # print("Alignment complete")
    return turns

# ----------------------------
# Summarization
# ----------------------------
BASIC_PROMPT = """Summarize the following multi-speaker transcript.

Rules:
- Speakers are labeled (e.g., S1, S2). If attribution is uncertain, say so.
- Be concise and factual. Do not invent details.
- Keep it under ~180 words unless crucial details require more.

Return:
1) Global summary: 5–8 bullets covering the main points.
2) Per-speaker notes: for each speaker (S1, S2…), 2–4 bullets on stance/requests/commitments.
3) Action items: owner → task → due date (if mentioned).
4) Open questions or decisions made.

Transcript:
"""

def turns_to_text(turns: List[Dict[str,Any]]) -> str:
    lines = []
    for t in turns:
        start = t.get("start", 0.0)
        end = t.get("end", 0.0)
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        ov = t.get("overlap", False)
        lines.append(f"[{start:.2f}-{end:.2f}] {spk}{' (overlap)' if ov else ''}: {txt}")
    return "\n".join(lines)

def summarize(turns: List[Dict[str,Any]]) -> str:
    transcript = turns_to_text(turns)
    msg = BASIC_PROMPT + transcript
    resp = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, accurate meeting summarizer."},
            {"role": "user", "content": msg},
        ],
        # temperature=0.2,
    )
    # print("Summarization complete")
    return resp.choices[0].message.content

# ----------------------------
# S3 helpers
# ----------------------------
def s3_download_to_bytes(key: str) -> bytes:
    buf = io.BytesIO()
    s3.download_fileobj(S3_BUCKET, key, buf)
    # print("S3 download complete")
    return buf.getvalue()

# ----------------------------
# Worker core
# ----------------------------
def ensure_session(conn, session_id: uuid.UUID, user_id: uuid.UUID, s3_key: str):
    # Minimal title + started_at if unknown
    title = "Session"
    started_at = datetime.now(timezone.utc)
    conn.execute(SQL_UPSERT_SESSION_FROM_JOB, (session_id, user_id, title, started_at, s3_key))
    return session_id

def process_job(job: Dict[str, Any]):
    # Expected payload (best effort): { job_id?, user_id, session_id, s3_key, title?, started_at? }
    job_id = job.get("job_id")
    user_id = uuid.UUID(job["user_id"])
    session_id = uuid.UUID(job.get("session_id", str(uuid.uuid4())))
    s3_key = job["s3_key"]
    title = job.get("title", "Session")
    started_at_str = job.get("started_at")

    # print(f"Processing session={session_id} user={user_id} key={s3_key}")

    asr_run_id = uuid.uuid4()

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        # If the enqueue system also creates a job row, update it; otherwise create a running job record if id exists.
        if job_id:
            try:
                conn.execute(SQL_INSERT_JOB_IF_PRESENT, (uuid.UUID(job_id), session_id))
            except Exception:
                pass

        # Ensure session exists / is processing
        ensure_session(conn, session_id, user_id, s3_key)

        # Create ASR run row
        params = {
            "detect_model": WHISPER_DETECT_MODEL,
            "en_model": WHISPER_EN_MODEL,
            "other_model": WHISPER_OTHER_MODEL,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "lang_conf_thr": LANG_CONF_THR,
            "diarization": bool(HUGGINGFACE_TOKEN and _HAS_PYANNOTE)
        }
        conn.execute(SQL_INSERT_ASR_RUN, (asr_run_id, session_id, json.dumps(params)))

    # 1) Download audio
    # audio_bytes = s3_download_to_bytes(s3_key)
    # tmp_path = f"/tmp/{uuid.uuid4()}.wav"
    # with open(tmp_path, "wb") as f:
    #     f.write(audio_bytes)

    # 1) Download audio
    audio_bytes = s3_download_to_bytes(s3_key)

    # Validate downloaded data
    if not audio_bytes:
        raise ValueError("Downloaded audio file is empty")

    tmp_path = f"/tmp/{uuid.uuid4()}.wav"

    # Ensure /tmp exists and write file with validation
    os.makedirs("/tmp", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
        f.flush()

    # Verify file was created
    if not os.path.exists(tmp_path):
        raise FileNotFoundError(f"Failed to create temp file: {tmp_path}")

    try:
        # 2) Diarize + Transcribe
        spk_segs = diarize(tmp_path)
        asr = transcribe_routed(tmp_path)
        # print("ASR :", asr)
        # print("ASR text :", asr["text"])
        turns = align_words_to_speakers(spk_segs, asr["segments_json"])

        # 3) Persist transcript, segments, insights, session status
        word_count = len(asr["text"].split())

        with psycopg.connect(DB_URL, autocommit=True) as conn:
            # transcripts (one per session)
            tid = uuid.uuid4()
            conn.execute(
                SQL_INSERT_TRANSCRIPT,
                (tid, session_id, asr["text"], asr["det_lang"], word_count)
            )

            # segments
            for t in turns:
                seg_id = uuid.uuid4()
                print(f"Segment: {t['start']:.2f}-{t['end']:.2f} {t['speaker']}: {t['text'][:40]}...")
                conn.execute(
                    SQL_INSERT_SEGMENT,
                    (seg_id, session_id, float(t["start"]), float(t["end"]), t["speaker"], t["text"])
                )

            # insights: summary
            summary_text = summarize(turns)
            print("Summary:", summary_text)
            conn.execute(
                SQL_INSERT_INSIGHT,
                (uuid.uuid4(), session_id, 'summary', json.dumps({
                    "model": SUMMARY_MODEL,
                    "language": asr["det_lang"],
                    "confidence": asr["det_prob"],
                    "summary": summary_text
                }))
            )

            # mark session ready
            conn.execute(SQL_UPDATE_SESSION_READY, (session_id,))

            # mark file processed if present
            try:
                row = conn.execute(SQL_FIND_FILE_BY_S3, (s3_key,)).fetchone()
                if row and row[0]:
                    conn.execute(SQL_UPDATE_FILE_STATUS, (row[0],))
            except Exception:
                pass

            # complete job + asr_run
            if job_id:
                try:
                    conn.execute(SQL_COMPLETE_JOB, (uuid.UUID(job_id),))
                except Exception:
                    pass
            conn.execute(SQL_COMPLETE_ASR_RUN, (asr_run_id,))

        print(f"[OK] session={session_id} transcript saved")

    except Exception as e:
        print("ERROR during processing:", e)
        # mark failures
        with psycopg.connect(DB_URL, autocommit=True) as conn:
            try:
                conn.execute(SQL_FAIL_ASR_RUN, (asr_run_id,))
            except Exception:
                pass
            if job_id:
                try:
                    conn.execute(SQL_FAIL_JOB, (uuid.UUID(job_id),))
                except Exception:
                    pass
        raise
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def main():
    print("Worker online. Queue: whys:jobs")
    # while True:
        # print("Waiting for job...")
    try:
        popped = r.brpop("whys:jobs", timeout=10)
        # if not popped:
        #     continue
        _, payload = popped
        job = json.loads(payload)
        process_job(job)
    except Exception as e:
        print("ERROR:", e)
        time.sleep(1)

if __name__ == "__main__":
    main()
