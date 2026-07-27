import os
import time
import threading
from test_database import EventDatabase
from test_assistant import AIAssistant

# --- NUCLEAR TIMEOUT WRAPPER ---
# This forces the code to stop even if OpenAI hangs forever
def _run_with_timeout(func, timeout=10):
    result = [None]
    exc = [None]

    def worker():
        try:
            result[0] = func()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    t.join(timeout)
    
    if t.is_alive():
        raise TimeoutError(f"System timed out after {timeout} seconds. (Network/Key Error?)")
    
    if exc[0]:
        raise exc[0]
    
    return result[0]
# ---------------------------

def main():
    print("=" * 60)
    print("  AI ASSISTANT TEST (CONSOLE - FORCED TIMEOUT)")
    print("=" * 60)

    # Setup Database
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "security.db")
    db = EventDatabase(db_path=db_path)
    db.setup_database()
    print("[INFO] Database loaded.")

    # Setup Assistant
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"[DEBUG] Using API Key: {api_key[:10]}...") # Verify key is loaded
    
    assistant = AIAssistant(
        db=db, 
        alert_manager=None, 
        vision_system=None, 
        config={
            "openai_api_key": api_key,
            "stt_engine": "text", 
            "tts_engine": "pyttsx3"
        }
    )

    print("\n[SYSTEM] AI Assistant Ready (10 second timeout enforced).")
    print("[SYSTEM] Type 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ")
            if not user_input.strip(): continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Run with timeout wrapper
            print("[AI] Thinking... (Max wait: 10s)", end="", flush=True)
            start = time.time()
            
            try:
                response = _run_with_timeout(lambda: assistant.ask(user_input), timeout=10)
                elapsed = time.time() - start
                print(f" Done ({elapsed:.2f}s)")
                print(f"\nAI: {response}")
            except TimeoutError as te:
                print(f" FAILED.")
                print(f"\n[ERROR] {te}")
                print("[HINT] If you get this repeatedly:")
                print("  1. Your API Key 'OPENAI_API_KEY_PLACEHOLDER' might be fake/invalid.")
                print("  2. Your internet might be blocking OpenAI.")
                print("  3. Try setting key via: setx OPENAI_API_KEY '<your-openai-api-key>'")
            except Exception as e:
                print(f"\n[ERROR] {e}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()