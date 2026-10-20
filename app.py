from flask import Flask, jsonify, send_from_directory, send_file, request
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pathlib import Path
import os
import re
import json
import asyncio
import uuid
from typing import Any
import edge_tts

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
TTS_VOICE = "en-US-JennyNeural"
AUDIO_DIR = BASE_DIR / "tts_audio"
AUDIO_DIR.mkdir(exist_ok=True)


def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a .env file beside app.py and add GEMINI_API_KEY=your_key."
        )
    return genai.Client(api_key=key)


def classify_request(message: str):
    text = message.lower().strip()
    count_match = re.search(
        r"\b(\d{1,3})\s+(?:points?|principles?|factors?|functions?|steps?|stages?|causes?|effects?|advantages?|disadvantages?|characteristics?|features?|reasons?|types?|items?)\b",
        text,
        re.I,
    )
    requested_count = int(count_match.group(1)) if count_match else None
    point_form = bool(
        re.search(r"\b(point form|point-form|points|bullet points|bullets|list form)\b", text)
    )
    numbered = bool(re.search(r"\b(numbered|numbering|numbered list|steps)\b", text))
    detail = bool(
        re.search(r"\b(explain in detail|explain fully|detailed explanation|in depth|elaborate|discuss in detail)\b", text)
    )
    compare = bool(re.search(r"\b(compare|comparison|difference between|differences between|similarities and differences)\b", text))
    if compare:
        mode = "comparison"
    elif point_form or requested_count is not None:
        mode = "points"
    elif numbered:
        mode = "numbered"
    elif detail:
        mode = "detailed"
    else:
        mode = "normal"
    return mode, requested_count


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": ["normal", "detailed", "points", "numbered", "comparison"]},
        "title": {"type": "string"},
        "content": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["mode", "title", "content"],
}

SYSTEM_PROMPT = """
You are Study AI, a warm, thoughtful conversational AI and university study companion. You are not a human, and must never claim to be one or invent personal experiences. Sound natural and attentive, as in a real conversation. Respond to the student's actual words and emotional tone when relevant. Use contractions where they fit, vary sentence rhythm, avoid canned openings and repeating the question, and remember the conversation history provided. Ask a concise follow-up only when it genuinely helps; otherwise answer directly.
Your job is to answer the student's actual request directly. Never ask the student to specify a topic when the topic is already present in the request.
GENERAL RULES:
- Be accurate, clear, natural and useful for university study.
- Give the direct answer instead of asking unnecessary follow-up questions.
- Do not give one-word answers when the student asks for an explanation.
- Do not use Markdown headings, ##, #, asterisks, emojis, arrows or decorative separators.
- Use ordinary punctuation.
- Do not invent facts, examples, sources or statistics.
DETAILED EXPLANATION:
When the student says "explain in detail", "explain fully", "in depth", "elaborate", or asks to "discuss in detail", produce a genuinely developed explanation.
Include the main idea, meaning or definition, important features or components, how/why it works or matters, and relevant examples where useful.
Do not answer with one sentence or one word.
For a detailed response, content should normally contain several substantial paragraphs, each covering a distinct part of the explanation.
POINT FORM:
When the student asks for point form, bullet points, a list, principles, factors, advantages, disadvantages, characteristics, functions, causes, effects, features, types, or a specific number of items, return ALL relevant items separately.
Each item must be one distinct point.
Never merge several requested points into one item.
If the student requests an exact number and that number matches an established list, return exactly that many items.
Do not invent points just to reach a number.
Do not stop after the first item.
NUMBERED:
For numbered lists or ordered steps, put every step/item in its own content item and preserve the sequence.
COMPARISON:
For comparisons, give multiple distinct comparison points rather than reducing the comparison to one sentence.
NORMAL:
For ordinary questions, answer naturally in paragraphs. Give enough explanation to satisfy the question rather than being artificially brief.
"""


def generate_response(message: str, mode: str, requested_count, history=None):
    client = get_gemini_client()
    extra = f"\n\nREQUEST MODE: {mode}."
    if requested_count is not None:
        extra += f" The student explicitly requested {requested_count} items. Return exactly {requested_count} separate items when the topic has an established list of that size."
    if mode == "detailed":
        extra += " The answer must be genuinely detailed and developed, with multiple substantial content items."
    elif mode == "points":
        extra += " Return every relevant point as a separate content item."
    elif mode == "numbered":
        extra += " Return every numbered step/item as a separate content item."
    elif mode == "comparison":
        extra += " Return several distinct comparison points as separate content items."
    prompt = message + extra
    history = history if isinstance(history, list) else []
    # Keep a small, bounded window of recent turns so replies can follow the conversation.
    transcript = []
    for turn in history[-12:]:
        if not isinstance(turn, dict):
            continue
        role = "Student" if turn.get("role") == "user" else "Study AI"
        content = str(turn.get("content", "")).strip()[:5000]
        if content:
            transcript.append(f"{role}: {content}")
    conversation_context = "\n\nRECENT CONVERSATION (oldest first):\n" + "\n".join(transcript) if transcript else ""

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text=SYSTEM_PROMPT + conversation_context + "\n\nCURRENT STUDENT REQUEST:\n" + prompt)],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.55,
            max_output_tokens=3200 if mode == "detailed" else 1800,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )

    raw = response.text or ""
    data = json.loads(raw)
    content = [str(item).strip() for item in data.get("content", []) if str(item).strip()]
    return {
        "mode": data.get("mode", mode),
        "title": str(data.get("title", "")).strip(),
        "content": content,
        "model_used": MODEL,
    }


def format_reply(result, requested_mode, requested_count):
    content = result["content"]
    mode = requested_mode if requested_mode in {"normal", "detailed", "points", "numbered", "comparison"} else result["mode"]
    if requested_count is not None and mode in {"points", "numbered"} and len(content) != requested_count:
        raise ValueError(f"Gemini returned {len(content)} items, but the request requires {requested_count}.")
    if not content:
        return "I could not generate a response. Please try the question again."
    if mode == "points":
        return "\n".join(f"• {item}" for item in content)
    if mode == "numbered":
        return "\n".join(f"{i}. {item}" for i, item in enumerate(content, 1))
    return "\n\n".join(content)


def clean_tts_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[*_`#]+", "", text)
    text = re.sub(r"^\s*[•●▪◦]\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.M)
    text = re.sub(r"\s+", " ", text).strip()
    return text


async def generate_tts(text, output_path):
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    await communicate.save(str(output_path))


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(BASE_DIR, "next.html")


@app.route("/next.html", methods=["GET"])
@app.route("/next", methods=["GET"])
def next_page():
    return send_from_directory(BASE_DIR, "next.html")


@app.route("/library.html", methods=["GET"])
@app.route("/library", methods=["GET"])
def library_page():
    return send_from_directory(BASE_DIR, "library.html")


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "provider": "gemini",
        "model": MODEL,
        "gemini_key_loaded": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "tts_provider": "edge-tts",
        "tts_voice": TTS_VOICE,
        "tts_ready": True,
    })


@app.post("/api/ai")
def ai():
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    history = payload.get("history", [])
    if not message:
        return jsonify({"error": "Enter a message first."}), 400
    try:
        mode, requested_count = classify_request(message)
        result = generate_response(message, mode, requested_count, history)
        try:
            reply = format_reply(result, mode, requested_count)
        except ValueError:
            retry_message = (
                message
                + f"\n\nIMPORTANT RETRY: Return exactly {requested_count} distinct items. "
                + "Each item must be separate. Do not combine multiple points into one item."
            )
            result = generate_response(retry_message, mode, requested_count, history)
            reply = format_reply(result, mode, requested_count)
        return jsonify({"reply": reply, "mode": mode, "item_count": len(result["content"])})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/tutor-file")
def tutor_file():
    """Let Study Tutor explain a student-uploaded PDF, text file, or homework image."""
    uploaded = request.files.get("file")
    question = str(request.form.get("question", "")).strip() or "Explain this material step by step and help me learn it."
    topic = str(request.form.get("topic", "General study")).strip()[:200]
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose a file to upload."}), 400
    raw = uploaded.read(10 * 1024 * 1024 + 1)
    if not raw:
        return jsonify({"error": "The uploaded file is empty."}), 400
    if len(raw) > 10 * 1024 * 1024:
        return jsonify({"error": "File is larger than the 10 MB limit."}), 413

    import mimetypes
    filename = uploaded.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    mime = (uploaded.mimetype or mimetypes.guess_type(filename)[0] or "").lower()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    extension_mimes = {"pdf": "application/pdf", "txt": "text/plain", "md": "text/markdown", "csv": "text/csv", "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    allowed = {
        "application/pdf", "text/plain", "text/markdown", "text/csv",
        "image/jpeg", "image/png", "image/webp"
    }
    if mime not in allowed and ext in extension_mimes:
        mime = extension_mimes[ext]
    if mime not in allowed:
        return jsonify({"error": "Supported files: PDF, TXT, MD, CSV, JPG, PNG, and WEBP."}), 415

    try:
        client = get_gemini_client()
        instructions = (
            "You are Study Match's patient AI Tutor. Read the student's attached study material carefully. "
            "Answer their question using the uploaded material where possible. Explain step by step in clear language, "
            "define key terms, work through examples when useful, and finish with one short check-for-understanding question. "
            "If the image or document is unreadable, say so rather than guessing. If information is not in the material, "
            "distinguish general knowledge from what the file says. Subject/topic: " + topic + "\nStudent request: " + question
        )
        if mime.startswith("text/"):
            try:
                extracted = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                extracted = raw.decode("latin-1", errors="replace")
            if len(extracted) > 60000:
                extracted = extracted[:60000] + "\n[File truncated at 60,000 characters.]"
            contents = instructions + "\n\nUPLOADED FILE: " + filename + "\n\n" + extracted
        else:
            contents = [
                types.Part(text=instructions + "\n\nUploaded file: " + filename),
                types.Part.from_bytes(data=raw, mime_type=mime),
            ]
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.45, max_output_tokens=2400),
        )
        reply = (response.text or "I could not read enough information from that file. Try a clearer file or ask a more specific question.").strip()
        return jsonify({"reply": reply, "filename": filename, "mime_type": mime})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/tts")
def text_to_speech():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"success": False, "error": "No text provided."}), 400

    try:
        text = clean_tts_text(text)
        if not text:
            return jsonify({"success": False, "error": "No speakable text provided."}), 400
        if len(text) > 5000:
            text = text[:5000]

        filename = f"tts_{uuid.uuid4().hex}.mp3"
        output_path = AUDIO_DIR / filename
        asyncio.run(generate_tts(text, output_path))

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Edge TTS did not create an audio file.")

        return send_file(
            output_path,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name=filename,
            max_age=0,
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"Edge TTS error: {exc}"}), 500


@app.route("/tts_audio/<path:filename>", methods=["GET"])
def tts_audio(filename):
    return send_from_directory(AUDIO_DIR, filename, mimetype="audio/mpeg")


if __name__ == "__main__":
    print("=" * 60)
    print("Study Match - GEMINI + JENNY TTS BACKEND")
    print("Project folder:", BASE_DIR)
    print("Home:    http://127.0.0.1:5000/")
    print("Next:    http://127.0.0.1:5000/next.html")
    print("Library: http://127.0.0.1:5000/library.html")
    print("Health:  http://127.0.0.1:5000/health")
    print("Gemini model:", MODEL)
    print("Gemini key loaded:", bool(os.getenv("GEMINI_API_KEY", "").strip()))
    print("TTS provider: edge-tts")
    print("TTS voice:", TTS_VOICE)
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
