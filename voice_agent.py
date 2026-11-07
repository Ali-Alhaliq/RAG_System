import os
import re
import time
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from advanced_rag import run_adaptive_rag
import arabic_reshaper
from bidi.algorithm import get_display
load_dotenv()

# ==========================================
# 1. SETUP AZURE SPEECH CONFIGURATION
# ==========================================
speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")

if not speech_key or not speech_region:
    raise ValueError("Please set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in .env")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "ar-EG", "ar-SA"])
audio_config_stt = speechsdk.audio.AudioConfig(use_default_microphone=True)

speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config, 
    auto_detect_source_language_config=auto_detect_source_language_config, 
    audio_config=audio_config_stt
)

active_synthesizer = None
conversation_history = []
is_speaking = False  

def synthesis_started(evt):
    global is_speaking
    is_speaking = True

def synthesis_ended(evt):
    global is_speaking
    is_speaking = False

# ==========================================
# 2. MAIN VOICE LOOP & BARGE-IN LOGIC
# ==========================================
def start_voice_agent():
    global active_synthesizer, conversation_history, is_speaking
    print("\n" + "="*50)
    print("🎙️ Voice RAG Agent with Auto-Language & Barge-In Started!")
    print("Talk in English or Arabic. Start talking while the agent is speaking to interrupt it.")
    print("Say 'Stop', 'Exit', or 'خروج' to end.")
    print("="*50 + "\n")
    
    while True:
        print("\n🟢 Listening... (Speak into the microphone)")
        
        result = speech_recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            user_text = result.text.strip()
            
            if len(user_text) <= 2:
                continue

            stt_lang = result.properties.get(speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult, "en-US")
            
            if is_speaking and active_synthesizer:
                active_synthesizer.stop_speaking_async()
                is_speaking = False
                print("\n[🎙️ BARGE-IN DETECTED! Agent stopped speaking to listen to you.]")
            reshaped_user_text = arabic_reshaper.reshape(user_text)
            bidi_user_text = get_display(reshaped_user_text)
            print(f"👤 You said (STT guess: {stt_lang}): {bidi_user_text}")
            
            if "stop" in user_text.lower() or "exit" in user_text.lower() or "خروج" in user_text:
                print("Ending voice session. Goodbye!")
                break
                
            print("🧠 Processing via RAG...")
            
            rag_result = run_adaptive_rag(user_text)
            final_answer = rag_result["final_answer"]
            
            if "could not find sufficient information" in final_answer and stt_lang.startswith("ar"):
                final_answer = "عذراً، بناءً على المستندات المتاحة، لم أتمكن من العثور على معلومات كافية للإجابة على سؤالك."
            
            conversation_history.append(f"User: {user_text}")
            conversation_history.append(f"Agent: {final_answer}")
            
            # ==========================================
            # 3. TEXT-TO-SPEECH (SMART VOICE SELECTION)
            # ==========================================
            has_arabic_letters = bool(re.search("[\u0600-\u06FF]", final_answer))
            if has_arabic_letters:
                speech_config.speech_synthesis_voice_name = "ar-EG-SalmaNeural"
            else:
                speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
                
            active_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            
            active_synthesizer.synthesis_started.connect(synthesis_started)
            active_synthesizer.synthesis_completed.connect(synthesis_ended)
            active_synthesizer.synthesis_canceled.connect(synthesis_ended)
            
            reshaped_text = arabic_reshaper.reshape(final_answer)
            bidi_output = get_display(reshaped_text)
            
            print(f"\n🤖 Agent is speaking: {bidi_output}")
            active_synthesizer.speak_text_async(final_answer)
            
            time.sleep(0.5)
            
        elif result.reason == speechsdk.ResultReason.NoMatch:
            pass
        elif result.reason == speechsdk.ResultReason.Canceled:
            print("Speech Recognition canceled. Check your keys.")
            break

if __name__ == "__main__":
    start_voice_agent()