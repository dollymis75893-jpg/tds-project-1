import io
import json
import os
import re
import threading
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

# ==========================================
# CONFIGURATION & ENV VARIABLES
# ==========================================
TG_TOKEN = os.environ.get("BOT_TOKEN", "")
API_KEY = os.environ.get("AIPIPE_TOKEN", "")
LLM_MODEL = os.environ.get("MODEL", "gpt-4o-mini")
API_BASE = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
HOST_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
FILE_LOG = os.environ.get("LOG_PATH", "run.jsonl")

PUBLIC_LOG_LINK = f"{HOST_URL}/run.jsonl"
TELEGRAM_ENDPOINT = f"https://api.telegram.org/bot{TG_TOKEN}"

MAX_TURNS = 10
EXECUTION_TIMEOUT = 60  
MAX_BUDGET_SECONDS = 210  

# Thread locks and state
file_lock = threading.Lock()
chat_memory = {}
memory_lock = threading.Lock()

# ==========================================
# LOGGING SYSTEM
# ==========================================
def record_log(**data):
    """Saves structured logs for the grader to access."""
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    log_string = json.dumps(data, ensure_ascii=False, default=str)
    with file_lock:
        with open(FILE_LOG, "a", encoding="utf-8") as file:
            file.write(log_string + "\n")

# ==========================================
# PYTHON EXECUTION TOOL
# ==========================================
def execute_python_script(script_code: str) -> str:
    """Runs Python code safely with a timeout and captures print statements."""
    buffer = io.StringIO()
    outcome = {}

    def runner():
        environment = {"__name__": "__main__"}
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                exec(script_code, environment)
            outcome["success"] = True
        except Exception:
            outcome["success"] = False
            buffer.write("\n" + traceback.format_exc(limit=4))

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join(EXECUTION_TIMEOUT)
    
    if worker.is_alive():
        return f"SYSTEM ERROR: Execution exceeded {EXECUTION_TIMEOUT} seconds."
    
    output_text = buffer.getvalue()
    return output_text[-8000:] if output_text else "(No printed output found. Remember to use print())"

AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_script",
            "description": (
                "Executes Python code on the backend and retrieves standard output. "
                "Libraries like pandas, numpy, requests, bs4, and openpyxl are available. "
                "You must use print() to output the data you wish to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "The Python code to run"}},
                "required": ["code"],
            },
        },
    }
]

INSTRUCTIONS = """You are an intelligent data-analysis assistant operating via a Telegram bot.

Strict Guidelines:
1. Your primary task is to answer the user's MOST RECENT query. Previous turns are just context.
2. If data is missing, write Python code to fetch and analyze it using the 'execute_python_script' tool. Do not hallucinate statistics. If an official dataset fetch fails (e.g., MOSPI), you can rely on your internal knowledge.
3. The user will specify the JSON format they expect (e.g., {"answer": {"state": "XYZ"}, "log_url": "..."}).
4. NEVER output conversational text or markdown blocks. Your ENTIRE reply must be exactly ONE valid JSON object. Use "PLACEHOLDER_URL" for the log_url value.
5. If no specific JSON shape is requested, default to: {"answer": <your_answer>, "log_url": "PLACEHOLDER_URL"}.
6. If the user sends a conversational setup message like "Data is coming", acknowledge it with: {"answer": "ready", "log_url": "PLACEHOLDER_URL"}.
7. Match the requested JSON structure identically (keys, data types). Do not add unrequested keys.
"""

# ==========================================
# LLM INTEGRATION & PARSING
# ==========================================
def get_ai_response(conversation, enable_tools=True):
    payload = {"model": LLM_MODEL, "messages": conversation, "temperature": 0.0}
    if enable_tools:
        payload["tools"] = AVAILABLE_TOOLS
        
    response = requests.post(
        f"{API_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "Bot-Agent/1.0",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]

def parse_valid_json(text_input: str):
    """Extracts JSON from text, ignoring markdown or surrounding text."""
    clean_text = re.sub(r"```(json)?\s*|\s*```", "", text_input.strip(), flags=re.I)
    
    # Try finding the outermost brackets
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        try:
            return json.loads(clean_text[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass
    return None

# ==========================================
# AGENT LOGIC
# ==========================================
def process_data_query(chat_id: int, user_input: str) -> str:
    with memory_lock:
        chat_log = chat_memory.setdefault(chat_id, [])
        chat_log.append({"role": "user", "content": user_input})
        del chat_log[:-20]  # Keep history small
        session_messages = [{"role": "system", "content": INSTRUCTIONS}] + list(chat_log)

    record_log(event="new_query", chat_id=chat_id, query=user_input)
    
    bot_reply_text = None
    time_limit = time.time() + MAX_BUDGET_SECONDS
    
    for iteration in range(MAX_TURNS):
        is_timeout = time.time() > time_limit
        if is_timeout:
            session_messages.append({
                "role": "user", 
                "content": "Time limit reached. Provide your final JSON answer immediately."
            })
            
        try:
            ai_msg = get_ai_response(session_messages, enable_tools=not is_timeout)
        except Exception as e1:
            record_log(event="api_warning", error=str(e1))
            time.sleep(2)
            try:
                ai_msg = get_ai_response(session_messages, enable_tools=True)
            except Exception as e2:
                record_log(event="api_fatal", error=str(e2))
                break
                
        called_tools = ai_msg.get("tool_calls")
        if called_tools:
            session_messages.append(ai_msg)
            for tool in called_tools:
                try:
                    script = json.loads(tool["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError:
                    script = tool["function"]["arguments"]
                    
                record_log(event="running_code", step=iteration, script=script[:2000])
                execution_result = execute_python_script(script)
                record_log(event="code_output", step=iteration, result=execution_result[:2000])
                
                session_messages.append({
                    "role": "tool", 
                    "tool_call_id": tool["id"], 
                    "content": execution_result
                })
            continue
            
        bot_reply_text = ai_msg.get("content") or ""
        break

    # Format the final JSON response
    final_data = parse_valid_json(bot_reply_text) if bot_reply_text else None
    if not final_data:
        final_data = {"answer": (bot_reply_text or "Error: Could not generate answer").strip()[:1000]}
    
    if "answer" not in final_data:
        final_data = {"answer": final_data}
        
    final_data["log_url"] = PUBLIC_LOG_LINK
    json_string_reply = json.dumps(final_data, ensure_ascii=False)

    with memory_lock:
        chat_memory.setdefault(chat_id, []).append({"role": "assistant", "content": json_string_reply})
        
    record_log(event="final_response", chat_id=chat_id, response=json_string_reply)
    return json_string_reply

# ==========================================
# TELEGRAM POLLING
# ==========================================
def call_telegram_api(endpoint, **kwargs):
    res = requests.post(f"{TELEGRAM_ENDPOINT}/{endpoint}", json=kwargs, timeout=65)
    return res.json()

def process_single_update(update_data):
    message_data = update_data.get("message") or update_data.get("edited_message")
    if not message_data:
        return
        
    user_text = message_data.get("text") or message_data.get("caption") or ""
    sender_id = message_data["chat"]["id"]
    
    if not user_text:
        return
        
    try:
        final_json_reply = process_data_query(sender_id, user_text)
    except Exception:
        record_log(event="critical_crash", chat_id=sender_id, trace=traceback.format_exc())
        final_json_reply = json.dumps({"answer": "internal server error", "log_url": PUBLIC_LOG_LINK})
        
    call_telegram_api("sendMessage", chat_id=sender_id, text=final_json_reply)

def telegram_worker():
    record_log(event="booting_up", url=HOST_URL, ai_model=LLM_MODEL)
    next_offset = 0
    thread_pool = ThreadPoolExecutor(max_workers=5)
    
    while True:
        try:
            api_resp = requests.get(
                f"{TELEGRAM_ENDPOINT}/getUpdates",
                params={"offset": next_offset, "timeout": 50},
                timeout=65,
            ).json()
            
            for item in api_resp.get("result", []):
                next_offset = item["update_id"] + 1
                thread_pool.submit(process_single_update, item)
                
        except Exception as err:
            record_log(event="polling_interruption", error=str(err))
            time.sleep(5)

# ==========================================
# SERVER KEEP-ALIVE
# ==========================================
def prevent_sleep():
    """Pings the host periodically so Render doesn't shut down the free tier."""
    while True:
        time.sleep(600)  # 10 minutes
        try:
            requests.get(f"{HOST_URL}/health", timeout=30)
        except Exception:
            pass

# ==========================================
# FASTAPI SETUP
# ==========================================
app = FastAPI()

@app.on_event("startup")
def start_background_tasks():
    if not os.path.exists(FILE_LOG):
        record_log(event="log_file_created")
    threading.Thread(target=telegram_worker, daemon=True).start()
    threading.Thread(target=prevent_sleep, daemon=True).start()

@app.get("/health")
def health_check():
    return {"status": "operational", "model_used": LLM_MODEL, "log_path": PUBLIC_LOG_LINK}

@app.get("/run.jsonl")
def serve_logs():
    if os.path.exists(FILE_LOG):
        return FileResponse(FILE_LOG, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")

@app.get("/")
def home():
    return {"app": "TDS_Data_Analyst_Bot", "logs": PUBLIC_LOG_LINK}
