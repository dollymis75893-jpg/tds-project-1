import os
import json
import time
import threading
import requests
import io
import sys
import re
from contextlib import redirect_stdout
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn

# ---------------------------------------------------------
# 1. ENVIRONMENT VARIABLES & SETUP
# ---------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# When Render hosts your app, it will be something like https://my-bot.onrender.com
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
LOG_FILE = "run.jsonl"
log_lock = threading.Lock()
chat_sessions = {}

# ---------------------------------------------------------
# 2. LOGGING SYSTEM
# ---------------------------------------------------------
def log_event(data):
    """Safely writes a JSON object to the log file on a new line."""
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

# Ensure the log file exists
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        pass

# ---------------------------------------------------------
# 3. FASTAPI WEB SERVER (Grader accesses these)
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/run.jsonl")
def get_logs():
    return FileResponse(LOG_FILE)

# ---------------------------------------------------------
# 4. AI TOOL: PYTHON EXECUTION
# ---------------------------------------------------------
def run_python(code: str) -> str:
    """Executes Python code to analyze data and returns the printed output."""
    log_event({"action": "tool_call", "code": code})
    
    output = io.StringIO()
    try:
        # Redirect print() statements to our variable
        with redirect_stdout(output):
            exec(code, globals())
        result = output.getvalue()
        if not result.strip():
            result = "Code executed, but nothing was printed. Use print() to output results."
    except Exception as e:
        result = f"Error executing code: {str(e)}"
        
    # Cap output at 8000 chars to avoid crashing the LLM context
    result = result[:8000]
    log_event({"action": "tool_result", "result": result})
    return result

# ---------------------------------------------------------
# 5. AI AGENT LOGIC
# ---------------------------------------------------------
system_instruction = """
You are a data analyst Telegram bot. 
1. The user will ask data questions. You have a 'run_python' tool to write and execute Python code (pandas, numpy, requests, openpyxl are installed).
2. NEVER guess statistics. Always write code to download, parse, and print the answer.
3. CRITICAL: You must reply to EVERY message with EXACTLY ONE JSON OBJECT. 
4. DO NOT include markdown formatting, backticks (```json), or any conversational text. 
5. The JSON must have exactly two keys: "answer" (shaped exactly as the user requested) and "log_url" (set to "placeholder").
6. If the user sends a setup message like "I will send data next", reply with: {"answer": "acknowledged", "log_url": "placeholder"}.
"""

# We use gemini-1.5-flash as it is fast, smart, and supports tools well
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    tools=[run_python],
    system_instruction=system_instruction
)

def process_message(chat_id: int, text: str) -> str:
    log_event({"timestamp": time.time(), "chat_id": chat_id, "user_text": text})
    
    # Keep history for multi-turn conversations
    if chat_id not in chat_sessions:
        chat_sessions[chat_id] = model.start_chat(enable_automatic_function_calling=True)
    
    chat = chat_sessions[chat_id]
    
    try:
        # Send message to Gemini. With enable_automatic_function_calling=True, 
        # it will automatically call run_python and feed the result back to itself!
        response = chat.send_message(text)
        raw_text = response.text
        log_event({"action": "model_raw_response", "text": raw_text})
        
        # Defensive JSON parsing: Strip markdown backticks if the model ignores instructions
        clean_text = re.sub(r"```json\s*", "", raw_text)
        clean_text = re.sub(r"```\s*", "", clean_text).strip()
        
        # Extract the first valid JSON object using regex
        match = re.search(r'\{.*\}', clean_text.replace('\n', ' '), re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = {"answer": clean_text}
            
        # Ensure 'answer' key exists
        if "answer" not in parsed:
            parsed = {"answer": parsed}
            
        # Overwrite log_url with the real public URL
        parsed["log_url"] = f"{BASE_URL}/run.jsonl"
        final_reply = json.dumps(parsed)
        
        log_event({"action": "final_reply", "reply": parsed})
        return final_reply
        
    except Exception as e:
        error_msg = str(e)
        log_event({"action": "error", "error": error_msg})
        fallback = {"answer": f"Internal error: {error_msg}", "log_url": f"{BASE_URL}/run.jsonl"}
        return json.dumps(fallback)

# ---------------------------------------------------------
# 6. TELEGRAM LONG POLLING
# ---------------------------------------------------------
def poll_telegram():
    offset = None
    while True:
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {"timeout": 30, "offset": offset}
            resp = requests.get(url, params=params).json()
            
            if resp.get("ok"):
                for item in resp["result"]:
                    offset = item["update_id"] + 1
                    if "message" in item and "text" in item["message"]:
                        chat_id = item["message"]["chat"]["id"]
                        text = item["message"]["text"]
                        
                        # Process and get JSON reply
                        reply_json = process_message(chat_id, text)
                        
                        # Send back to Telegram
                        send_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                        requests.post(send_url, json={"chat_id": chat_id, "text": reply_json})
        except Exception as e:
            print(f"Telegram polling error: {e}")
            time.sleep(2)

# ---------------------------------------------------------
# 8. STARTUP
# ---------------------------------------------------------
@app.on_event("startup")
def startup_event():
    # Start Telegram polling in the background
    threading.Thread(target=poll_telegram, daemon=True).start()
    # Start Keep-Alive pinger in the background
    threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
