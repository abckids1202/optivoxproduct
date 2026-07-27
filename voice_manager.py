import pyttsx3
import threading
import time

class VoiceManager:
    """
    Handles Text-to-Speech (TTS) alerts for the security system.
    Uses pyttsx3 for offline voice generation.
    """

    def __init__(self):
        # 1. Initialize a temporary engine just to get available voices
        temp_engine = pyttsx3.init()
        voices = []
        
        try:
            voices = temp_engine.getProperty('voices')
        except Exception:
            print("[VoiceManager] Warning: Could not get voice list.")
            voices = []

        selected_voice = None
        for voice in voices:
            if voice:
                name_lower = voice.name.lower()
                if "zira" in name_lower or "huihui" in name_lower:
                    selected_voice = voice
                    break
                elif "female" in name_lower:
                    selected_voice = voice
                    
        self.voice_id = selected_voice.id if selected_voice else 0
        self.rate = 150  
        self.volume = 1.0 

    def _run_speech(self, text, engine):
        try:
            engine.setProperty('voice', self.voice_id)
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            
            engine.say(text)
            engine.runAndWait()
        except RuntimeError:
            pass
        except Exception as e:
            print(f"[VoiceManager] Error speaking: {e}")

    def speak(self, text):

        if not text:
            return
        
        t = threading.Thread(
            target=self._run_speech, 
            args=(text, pyttsx3.init()),
            daemon=True 
        )
        t.start()

    def alert_danger(self):
        """Critical Alert: Weapons, Evacuation, Fire."""
        msg = "Alert! Dangerous object detected."
        print(f"[Voice] {msg}")
        self.speak(msg)

    def alert_intruder(self):
        """Warning Alert: Unknown person, unauthorized access."""
        msg = "Warning! Unknown individual detected."
        print(f"[Voice] {msg}")
        self.speak(msg)

    def alert_evacuation(self):
        """Critical Alert: Emergency Evacuation detected."""
        msg = "Emergency! Evacuation pattern detected."
        print(f"[Voice] {msg}")
        self.speak(msg)

    def alert_system_start(self):
        """Info: System has started."""
        msg = "Intelligent security monitoring system online."
        self.speak(msg)

    def speak_text(self, text):
        """Custom text-to-speech for any string."""
        print(f"[Voice] Speaking: {text}")
        self.speak(text)

