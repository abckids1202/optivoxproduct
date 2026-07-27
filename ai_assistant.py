import os
import sys
import json
import time
import threading
import datetime
from typing import Optional, Any, Dict, List

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[AI ASSISTANT] openai package not found.")
    print("  Install: pip install openai")
    print("  Assistant will run in text-only demo mode.")

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("[AI ASSISTANT] pyttsx3 not found. Voice output disabled.")
    print("  Install: pip install pyttsx3")

try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("[AI ASSISTANT] speech_recognition not found. Voice input disabled.")
    print("  Install: pip install SpeechRecognition pyaudio")

DEFAULT_CONFIG = {
    "openai_api_key": "",                 
    "openai_model": "gpt-4o",             
    "openai_whisper_model": "whisper-1",   
    "openai_tts_model": "tts-1",         
    "openai_tts_voice": "nova",           
    "tts_engine": "pyttsx3",              
    "tts_rate": 170,                    
    "tts_volume": 0.9,                    

    "stt_engine": "google",              
    "stt_language": "en-US",              
    "stt_timeout": 8,                     
    "stt_phrase_time_limit": 30,          

    "system_prompt": None,                
    "max_tool_iterations": 5,            
    "conversation_history_limit": 10,     

    "greeting_message": "Security assistant activated. How can I help you?",
    "farewell_message": "Security assistant deactivated.",
    "error_message": "Sorry, I encountered an error processing your request.",
    "listen_indicator": "[AI] Listening... (speak now, or press Ctrl+C to cancel)",
    "thinking_indicator": "[AI] Thinking...",
    "speaking_indicator": "[AI] Speaking...",
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": (
                "Get the current overall status of the security system including "
                "number of enrolled faces, recent event count, system uptime, "
                "and active alert channels."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_enrolled_faces",
            "description": (
                "Get a list of all people enrolled in the face recognition system. "
                "Returns their names and when they were registered."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_events",
            "description": (
                "Get the most recent security events with details such as event type, "
                "person involved, confidence score, timestamp, and severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default: 20, max: 100).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_event_summary",
            "description": (
                "Get a statistical summary of events grouped by type over a time period. "
                "Useful for understanding overall activity patterns and identifying trends."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_by_type",
            "description": (
                "Search for events of a specific type. Useful for finding all instances "
                "of suspicious behavior, crowd alerts, dangerous objects, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "description": (
                            "The event type to search for. Common types: RECOGNITION, "
                            "SPOOF_DETECTED, LOITERING, RUNNING, HESITATION, PACING, "
                            "SCANNING, SPATIAL_ANOMALY, CROWD_FORMING, DANGEROUS_OBJECT, "
                            "EVACUATION_ALERT, OBJECT_INTERACTION, IN, OUT."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default: 20).",
                    },
                },
                "required": ["event_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_person_events",
            "description": (
                "Get all events associated with a specific person. Shows their "
                "detection history, behavior flags, and activity timeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "The name of the enrolled person to look up.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return (default: 50).",
                    },
                },
                "required": ["person_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_person_behavior_profile",
            "description": (
                "Get the behavioral analysis profile for a specific person. This includes "
                "typical visit patterns, stress levels, suspicious activity flags, and "
                "movement characteristics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {
                        "type": "string",
                        "description": "The name of the enrolled person.",
                    },
                },
                "required": ["person_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alert_history",
            "description": (
                "Get the recent alert/notification history showing what alerts were "
                "sent, through which channels (email, Telegram, Discord, SMS), "
                "and their severity levels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of alerts to return (default: 20).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_audit_log",
            "description": (
                "Get the system audit log showing configuration changes, enrollments, "
                "deletions, and system start/stop events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Filter by action type (optional). Common actions: "
                            "SYSTEM_START, SYSTEM_STOP, ENROLL_PERSON, CONFIG_CHANGE."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries (default: 20).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": (
                "Full-text search across recent events. Searches event types, person names, "
                "and detail descriptions for the given query string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text to find matching events.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to search back (default: 7).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 30).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

def _build_system_prompt() -> str:
    return (
        "You are an AI assistant integrated into an intelligent security monitoring system. "
        "You have real-time access to the security database and can answer questions about:\n\n"
        "- **Face Recognition**: Who is enrolled, when they were detected, recognition confidence\n"
        "- **Security Events**: Alerts, suspicious behavior, crowd detection, dangerous objects\n"
        "- **Behavior Analysis**: Person profiles, stress levels, movement patterns, loitering\n"
        "- **System Status**: Active cameras, enrolled faces count, alert channel status\n"
        "- **Audit Trail**: System events, configuration changes, enrollment history\n\n"
        "Guidelines:\n"
        "1. Be concise but informative. Users are monitoring security and need quick answers.\n"
        "2. Always specify timestamps relative to now (e.g., '3 hours ago', 'yesterday at 2pm').\n"
        "3. Highlight security concerns prominently (high severity events, unknown faces, etc.).\n"
        "4. If asked about something not in the database, say so clearly.\n"
        "5. When reporting events, include event type, person (if applicable), time, and severity.\n"
        "6. Use the provided tools to query the database. Do NOT make up data.\n"
        "7. If a query returns no results, report that honestly.\n"
        "8. For time-sensitive questions, convert UTC timestamps to the local context.\n"
        "9. Suggest follow-up actions when relevant (e.g., 'Would you like to see more details about this event?').\n"
        "Current time: {current_time}\n"
    ).format(current_time=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

class AIAssistant:
    def __init__(self, db, alert_manager=None, vision_system=None, config=None):
        self.db = db
        self.alert_manager = alert_manager
        self.vision_system = vision_system
        self.config = {**DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        self._start_time = time.time()
        self._is_listening = False
        self._stop_event = threading.Event()
        self._thread = None
        self._conversation_history = []
        self._lock = threading.Lock()

        self._tts_engine = None
        self._init_tts()

        self._stt_recognizer = None
        self._init_stt()

        self._openai_client = None
        self._init_openai()

    def _init_openai(self):
        if not OPENAI_AVAILABLE:
            return

        api_key = self.config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[AI ASSISTANT] No OpenAI API key provided.")
            print("  Set OPENAI_API_KEY environment variable or pass openai_api_key in config.")
            print("  Assistant will run in text-only demo mode.")
            return

        try:
            self._openai_client = openai.OpenAI(api_key=api_key)
            model = self.config.get("openai_model", "gpt-4o")
            print(f"[AI ASSISTANT] OpenAI client initialized (model: {model}).")
        except Exception as e:
            print(f"[AI ASSISTANT] OpenAI init failed: {e}")
            self._openai_client = None

    def _init_tts(self):
        if not TTS_AVAILABLE:
            return

        try:
            self._tts_engine = pyttsx3.init()
            rate = self.config.get("tts_rate", 170)
            volume = self.config.get("tts_volume", 0.9)
            self._tts_engine.setProperty("rate", rate)
            self._tts_engine.setProperty("volume", volume)

            voices = self._tts_engine.getProperty("voices")
            if voices:
                for v in voices:
                    if "english" in v.name.lower() or "en" in v.id.lower():
                        self._tts_engine.setProperty("voice", v.id)
                        break

            print(f"[AI ASSISTANT] TTS engine ready (pyttsx3, {len(voices)} voices available).")
        except Exception as e:
            print(f"[AI ASSISTANT] TTS init failed: {e}")
            self._tts_engine = None

    def _init_stt(self):
        if not STT_AVAILABLE:
            return

        try:
            self._stt_recognizer = sr.Recognizer()
            self._stt_recognizer.energy_threshold = 300
            self._stt_recognizer.dynamic_energy_threshold = True
            self._stt_recognizer.pause_threshold = 0.8
            self._stt_recognizer.phrase_threshold = 0.3
            self._stt_recognizer.non_speaking_duration = 0.5
            print("[AI ASSISTANT] Speech recognizer ready.")
        except Exception as e:
            print(f"[AI ASSISTANT] STT init failed: {e}")
            self._stt_recognizer = None

    def is_ready(self) -> bool:
        has_api = self._openai_client is not None
        has_stt = self.config.get("stt_engine") == "text" or STT_AVAILABLE
        has_tts = TTS_AVAILABLE
        return has_api and has_stt and has_tts

    def is_listening(self) -> bool:
        return self._is_listening

    def get_capabilities(self) -> dict:
        return {
            "openai": OPENAI_AVAILABLE and self._openai_client is not None,
            "tts": TTS_AVAILABLE and self._tts_engine is not None,
            "tts_engine": self.config.get("tts_engine", "pyttsx3"),
            "stt": STT_AVAILABLE and self._stt_recognizer is not None,
            "stt_engine": self.config.get("stt_engine", "google"),
            "voice_input": STT_AVAILABLE or self.config.get("stt_engine") == "text",
            "voice_output": TTS_AVAILABLE,
        }

    def start_listening(self):
        if self._is_listening:
            print("[AI ASSISTANT] Already listening. Press 'A' again to cancel.")
            return

        if self._openai_client is None:
            print("[AI ASSISTANT] Cannot start: OpenAI API not configured.")
            print("  Set OPENAI_API_KEY environment variable.")
            return

        self._is_listening = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            name="AIAssistant",
            daemon=True,
        )
        self._thread.start()
        print("[AI ASSISTANT] Background thread started.")

    def stop_listening(self):
        if not self._is_listening:
            return
        self._stop_event.set()
        self._is_listening = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        print("[AI ASSISTANT] Stopped.")

    def ask(self, question: str) -> str:
        with self._lock:
            response = self._process_question(question)
        return response

    def _listen_loop(self):
        try:
            greeting = self.config.get("greeting_message",
                                       "Security assistant activated. How can I help you?")
            self._speak(greeting)
            print(f"\n[AI] {greeting}")
            print("[AI] Say 'exit', 'stop', or 'quit' to end the session.")
            print("[AI] Or press Ctrl+C to cancel.\n")

            while not self._stop_event.is_set():
                user_text = self._listen_for_speech()
                if user_text is None:
                    continue

                if any(cmd in user_text.lower() for cmd in
                       ["exit", "stop", "quit", "done", "that's all", "that is all",
                        "thank you goodbye", "goodbye", "bye"]):
                    farewell = self.config.get("farewell_message",
                                               "Security assistant deactivated.")
                    print(f"\n[AI] {farewell}\n")
                    self._speak(farewell)
                    break

                print(f"\n[USER] {user_text}")
                self._speak(self.config.get("thinking_indicator", "[AI] Thinking..."))
                print("[AI] Thinking...")

                try:
                    answer = self.ask(user_text)
                    print(f"[AI] {answer}")
                    self._speak(answer)
                except Exception as e:
                    error_msg = self.config.get("error_message",
                                                "Sorry, I encountered an error.")
                    print(f"[AI ERROR] {e}")
                    print(f"[AI] {error_msg}")
                    self._speak(error_msg)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[AI ASSISTANT ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_listening = False
            print("[AI ASSISTANT] Session ended. Returning to camera view.\n")

    def _listen_for_speech(self) -> Optional[str]:
        stt_engine = self.config.get("stt_engine", "google")

        if stt_engine == "text" or self._stt_recognizer is None:
            print(self.config.get("listen_indicator", "[AI] Type your question:"))
            try:
                user_text = input("[AI] You: ").strip()
                if user_text:
                    return user_text
                return None
            except (EOFError, KeyboardInterrupt):
                return None

        timeout = self.config.get("stt_timeout", 8)
        phrase_limit = self.config.get("stt_phrase_time_limit", 30)
        language = self.config.get("stt_language", "en-US")

        print(self.config.get("listen_indicator", "[AI] Listening..."))

        with sr.Microphone() as source:
            try:
                self._stt_recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._stt_recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )
            except sr.WaitTimeoutError:
                print("[AI] No speech detected. Listening again...")
                return None
            except Exception as e:
                print(f"[AI] Microphone error: {e}")
                return None

        try:
            if stt_engine == "whisper" and self._openai_client:
                audio_data = audio.get_wav_data()
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    f.write(audio_data)
                    f.flush()
                    with open(f.name, "rb") as audio_file:
                        transcript = self._openai_client.audio.transcriptions.create(
                            model=self.config.get("openai_whisper_model", "whisper-1"),
                            file=audio_file,
                            language=language.split("-")[0],
                        )
                return transcript.text.strip()

            else:
                text = self._stt_recognizer.recognize_google(audio, language=language)
                return text.strip()

        except sr.UnknownValueError:
            print("[AI] Could not understand audio. Please try again.")
            return None
        except sr.RequestError as e:
            print(f"[AI] Speech recognition service error: {e}")
            return None
        except Exception as e:
            print(f"[AI] Transcription error: {e}")
            return None

    def _speak(self, text: str):
        if not text or not text.strip():
            return

        tts_engine = self.config.get("tts_engine", "pyttsx3")

        if tts_engine == "openai" and self._openai_client:
            self._speak_openai(text)
            return

        if self._tts_engine is not None:
            try:
                self._tts_engine.stop()
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as e:
                print(f"[AI] TTS error: {e}")
        else:
            pass

    def _speak_openai(self, text: str):
        try:
            import tempfile
            import subprocess

            response = self._openai_client.audio.speech.create(
                model=self.config.get("openai_tts_model", "tts-1"),
                voice=self.config.get("openai_tts_voice", "nova"),
                input=text[:4096],  
                response_format="mp3",
            )

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
                temp_path = f.name

            if sys.platform == "win32":
                os.startfile(temp_path)
            elif sys.platform == "darwin":
                subprocess.run(["afplay", temp_path], check=True)
            else:
                for player in ["mpv", "aplay", "paplay", "ffplay"]:
                    try:
                        subprocess.run(
                            [player, temp_path],
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        break
                    except FileNotFoundError:
                        continue

            try:
                os.unlink(temp_path)
            except OSError:
                pass

        except Exception as e:
            print(f"[AI] OpenAI TTS error: {e}")

    def _process_question(self, question: str) -> str:
        if self._openai_client is None:
            return _demo_response(question, self.db)

        system_prompt = self.config.get("system_prompt") or _build_system_prompt()

        with self._lock:
            self._conversation_history.append({
                "role": "user",
                "content": question,
            })

            limit = self.config.get("conversation_history_limit", 10)
            if len(self._conversation_history) > limit * 2:
                self._conversation_history = self._conversation_history[-(limit * 2):]

            messages = [{"role": "system", "content": system_prompt}] + self._conversation_history

        max_iterations = self.config.get("max_tool_iterations", 5)
        model = self.config.get("openai_model", "gpt-4o")

        for _iteration in range(max_iterations):
            try:
                response = self._openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=1024,
                )
            except openai.RateLimitError:
                return "I'm experiencing high demand right now. Please try again in a moment."
            except openai.APIConnectionError:
                return "I can't reach the AI service. Please check your internet connection."
            except openai.APIStatusError as e:
                return f"API error: {e.status_code} - {e.message}"
            except Exception as e:
                return f"An unexpected error occurred: {e}"

            choice = response.choices[0]
            assistant_message = choice.message

            messages.append(assistant_message)

            if assistant_message.tool_calls and len(assistant_message.tool_calls) > 0:
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    result = self._execute_tool(func_name, func_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                continue

            final_answer = assistant_message.content or "I don't have a response for that."

            with self._lock:
                self._conversation_history.append({
                    "role": "assistant",
                    "content": final_answer,
                })

            return final_answer

        return "I needed to gather too much information. Please try a more specific question."

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        try:
            if tool_name == "get_system_status":
                return self._tool_get_system_status()

            elif tool_name == "get_enrolled_faces":
                return self._tool_get_enrolled_faces()

            elif tool_name == "get_recent_events":
                limit = min(args.get("limit", 20), 100)
                return self._tool_get_recent_events(limit)

            elif tool_name == "get_event_summary":
                days = min(args.get("days", 7), 365)
                return self._tool_get_event_summary(days)

            elif tool_name == "get_events_by_type":
                event_type = args.get("event_type", "")
                limit = min(args.get("limit", 20), 100)
                return self._tool_get_events_by_type(event_type, limit)

            elif tool_name == "get_person_events":
                person_name = args.get("person_name", "")
                limit = min(args.get("limit", 50), 200)
                return self._tool_get_person_events(person_name, limit)

            elif tool_name == "get_person_behavior_profile":
                person_name = args.get("person_name", "")
                return self._tool_get_person_behavior_profile(person_name)

            elif tool_name == "get_alert_history":
                limit = min(args.get("limit", 20), 100)
                return self._tool_get_alert_history(limit)

            elif tool_name == "get_audit_log":
                action = args.get("action")
                limit = min(args.get("limit", 20), 100)
                return self._tool_get_audit_log(action, limit)

            elif tool_name == "search_events":
                query = args.get("query", "")
                days = min(args.get("days", 7), 365)
                limit = min(args.get("limit", 30), 100)
                return self._tool_search_events(query, days, limit)

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    def _tool_get_system_status(self) -> str:
        face_count = self.db.get_face_count()
        recent_events = self.db.get_recent_events(limit=1)
        last_event_time = None
        if recent_events:
            last_event_time = recent_events[0]["timestamp"]

        uptime_sec = time.time() - self._start_time
        uptime_str = _format_duration(uptime_sec)

        alert_channels = {}
        if self.alert_manager:
            alert_channels = {
                "email": self.alert_manager.email_enabled,
                "telegram": self.alert_manager.telegram_enabled,
                "discord": self.alert_manager.discord_enabled,
                "sms": self.alert_manager.sms_enabled,
                "webhook": self.alert_manager.webhook_enabled,
            }

        today = datetime.datetime.utcnow().date().isoformat()
        try:
            today_events = self.db._fetchall(
                "SELECT COUNT(*) as cnt FROM events WHERE date(timestamp) = ?",
                (today,),
            )
            today_count = today_events[0]["cnt"] if today_events else 0
        except Exception:
            today_count = "unknown"

        status = {
            "status": "online",
            "uptime": uptime_str,
            "enrolled_faces": face_count,
            "events_today": today_count,
            "last_event_time": last_event_time,
            "alert_channels": alert_channels,
            "current_time": datetime.datetime.utcnow().isoformat() + "Z",
            "assistant_capabilities": self.get_capabilities(),
        }
        return json.dumps(status, default=str)

    def _tool_get_enrolled_faces(self) -> str:
        faces = self.db.get_known_face_names(limit=100)
        result = []
        for face in faces:
            result.append({
                "id": face["id"],
                "name": face["name"],
                "registered": face["created_at"],
                "has_thumbnail": face["thumbnail_path"] is not None if "thumbnail_path" in face.keys() else False,
            })
        return json.dumps({
            "total": len(result),
            "faces": result,
        }, default=str)

    def _tool_get_recent_events(self, limit: int) -> str:
        events = self.db.get_recent_events(limit=limit)
        result = []
        for e in events:
            result.append({
                "id": e["id"],
                "timestamp": e["timestamp"],
                "event_type": e["event_type"],
                "person": e.get("person_name") or "Unknown/System",
                "confidence": round(e["confidence"], 3) if e["confidence"] else None,
                "severity": e["severity"],
                "location": e.get("location"),
                "camera": e.get("camera_id"),
                "details": _safe_json_parse(e.get("details_json")),
            })
        return json.dumps({
            "count": len(result),
            "events": result,
        }, default=str)

    def _tool_get_event_summary(self, days: int) -> str:
        summary = self.db.get_event_summary(days=days)
        result = []
        for row in summary:
            result.append({
                "event_type": row["event_type"],
                "total_count": int(row["total"]),
            })
        return json.dumps({
            "period_days": days,
            "summary": result,
        })

    def _tool_get_events_by_type(self, event_type: str, limit: int) -> str:
        events = self.db.get_events_by_type(event_type=event_type, limit=limit)
        result = []
        for e in events:
            result.append({
                "id": e["id"],
                "timestamp": e["timestamp"],
                "person": e.get("person_name") or "Unknown/System",
                "confidence": round(e["confidence"], 3) if e["confidence"] else None,
                "severity": e["severity"],
                "details": _safe_json_parse(e.get("details_json")),
            })
        return json.dumps({
            "event_type": event_type,
            "count": len(result),
            "events": result,
        }, default=str)

    def _tool_get_person_events(self, person_name: str, limit: int) -> str:
        person_id = self.db.get_person_id(person_name)
        if person_id is None:
            return json.dumps({
                "error": f"Person '{person_name}' not found in the database.",
                "suggestion": "Use get_enrolled_faces to see all registered people.",
            })

        events = self.db.get_person_timeline(person_id=person_id, limit=limit)
        result = []
        for e in events:
            result.append({
                "id": e["id"],
                "timestamp": e["timestamp"],
                "event_type": e["event_type"],
                "confidence": round(e["confidence"], 3) if e["confidence"] else None,
                "severity": e["severity"],
                "details": _safe_json_parse(e.get("details_json")),
                "location": e.get("location"),
            })

        profile = self.db.get_behavior_profile(person_id)

        return json.dumps({
            "person": person_name,
            "person_id": person_id,
            "total_events": len(result),
            "events": result,
            "behavior_profile": profile,
        }, default=str)

    def _tool_get_person_behavior_profile(self, person_name: str) -> str:
        person_id = self.db.get_person_id(person_name)
        if person_id is None:
            return json.dumps({
                "error": f"Person '{person_name}' not found in the database.",
            })

        profile = self.db.get_behavior_profile(person_id)

        try:
            type_counts = self.db._fetchall("""
                SELECT event_type, COUNT(*) as cnt
                FROM events WHERE person_id = ?
                GROUP BY event_type ORDER BY cnt DESC
            """, (person_id,))
        except Exception:
            type_counts = []

        event_breakdown = [
            {"event_type": r["event_type"], "count": r["cnt"]}
            for r in type_counts
        ]

        return json.dumps({
            "person": person_name,
            "behavior_profile": profile,
            "event_breakdown": event_breakdown,
        }, default=str)

    def _tool_get_alert_history(self, limit: int) -> str:
        if self.alert_manager is None:
            return json.dumps({"error": "Alert manager not configured."})

        history = self.alert_manager.get_alert_history(limit=limit)
        return json.dumps({
            "count": len(history),
            "alerts": history,
        }, default=str)

    def _tool_get_audit_log(self, action: Optional[str], limit: int) -> str:
        entries = self.db.get_audit_log(action=action, limit=limit)
        result = []
        for entry in entries:
            result.append({
                "id": entry["id"],
                "timestamp": entry["timestamp"],
                "action": entry["action"],
                "target": entry.get("target"),
                "details": _safe_json_parse(entry.get("details_json")),
            })
        return json.dumps({
            "count": len(result),
            "entries": result,
        }, default=str)

    def _tool_search_events(self, query: str, days: int, limit: int) -> str:
        search_term = f"%{query}%"

        try:
            since = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).isoformat()
            rows = self.db._fetchall("""
                SELECT e.*, p.name AS person_name
                FROM events e
                LEFT JOIN people p ON e.person_id = p.id
                WHERE e.timestamp >= ?
                  AND (
                    e.event_type LIKE ?
                    OR e.details_json LIKE ?
                    OR p.name LIKE ?
                    OR e.location LIKE ?
                  )
                ORDER BY e.timestamp DESC
                LIMIT ?
            """, (since, search_term, search_term, search_term, search_term, limit))

            result = []
            for e in rows:
                result.append({
                    "id": e["id"],
                    "timestamp": e["timestamp"],
                    "event_type": e["event_type"],
                    "person": e.get("person_name") or "Unknown/System",
                    "confidence": round(e["confidence"], 3) if e["confidence"] else None,
                    "severity": e["severity"],
                    "details": _safe_json_parse(e.get("details_json")),
                })

            return json.dumps({
                "query": query,
                "period_days": days,
                "count": len(result),
                "events": result,
            }, default=str)

        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})

    def cleanup(self):
        self.stop_listening()
        if self._tts_engine:
            try:
                self._tts_engine.stop()
            except Exception:
                pass

def _safe_json_parse(value) -> Any:
    if value is None:
        return {}
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {"raw": value}


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours < 24:
        return f"{hours}h {minutes}m"
    days = int(hours // 24)
    hours = hours % 24
    return f"{days}d {hours}h {minutes}m"


def _demo_response(question: str, db) -> str:
    q = question.lower()

    if "status" in q or "system" in q:
        face_count = db.get_face_count()
        return (f"System is online. {face_count} faces enrolled. "
                f"OpenAI API is not configured - running in demo mode.")

    if "face" in q or "enrolled" in q or "who" in q:
        faces = db.get_known_face_names(limit=50)
        if not faces:
            return "No faces are currently enrolled in the system."
        names = [f["name"] for f in faces]
        return f"Enrolled faces ({len(names)}): {', '.join(names[:20])}"

    if "event" in q or "recent" in q or "alert" in q or "happened" in q:
        events = db.get_recent_events(limit=10)
        if not events:
            return "No recent events found."
        lines = []
        for e in events[:10]:
            person = e.get("person_name") or "System"
            lines.append(
                f"  [{e['timestamp']}] {e['event_type']} - {person} "
                f"(severity: {e['severity']})"
            )
        return "Recent events:\n" + "\n".join(lines)

    if "help" in q:
        return (
            "I can answer questions about your security system. Try asking:\n"
            "  - 'What is the system status?'\n"
            "  - 'Who is enrolled in face recognition?'\n"
            "  - 'What events happened today?'\n"
            "  - 'Show me recent alerts'\n"
            "  - 'Tell me about [person name]'s activity'\n"
            "  - 'Were there any suspicious events this week?'\n\n"
            "Note: Running in demo mode. Connect OpenAI API for full NLP capabilities."
        )

    return (
        "I understand your question, but I'm running in demo mode without OpenAI API. "
        "Set the OPENAI_API_KEY environment variable for full conversational AI capabilities. "
        "In demo mode, I can show system status, enrolled faces, and recent events."
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  AI SECURITY ASSISTANT - STANDALONE TEST")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("\n[WARN] OPENAI_API_KEY not set. Running in demo mode.")
        print("  export OPENAI_API_KEY='<your-openai-api-key>'\n")

    import glob
    db_paths = glob.glob("**/*.db", recursive=True) + glob.glob("**/security.db", recursive=True)

    if db_paths:
        db_path = db_paths[0]
        print(f"[INFO] Found database: {db_path}")
    else:
        db_path = ":memory:"
        print("[INFO] No database found. Using in-memory database (demo only).")

    from database import EventDatabase

    db = EventDatabase(db_path=db_path)
    db.setup_database()

    assistant = AIAssistant(
        db=db,
        config={
            "stt_engine": "text",  
        },
    )

    print(f"\nCapabilities: {json.dumps(assistant.get_capabilities(), indent=2)}")
    print("\nType your questions below. Type 'quit' to exit.\n")

    try:
        while True:
            question = input("You: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break

            answer = assistant.ask(question)
            print(f"\nAI: {answer}\n")
    except KeyboardInterrupt:
        pass

    assistant.cleanup()
    print("\nGoodbye!")
