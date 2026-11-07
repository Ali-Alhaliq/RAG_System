# import os
# import json
# from flask import Flask, render_template, request, jsonify
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# from agent_tools import rag_search, calculate, create_support_ticket, get_current_datetime

# load_dotenv()

# app = Flask(__name__)

# openai_client = AzureOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION")
# )

# tools_definition = [
#     {
#         "type": "function",
#         "function": {
#             "name": "rag_search",
#             "description": "Search the internal knowledge base documents for relevant information based on the query.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "query": {"type": "string", "description": "The search query to look up in documents."}
#                 },
#                 "required": ["query"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "calculate",
#             "description": "Perform mathematical calculations. Pass a valid math expression as a string (e.g., '25 - 8').",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "expression": {"type": "string", "description": "The mathematical expression to evaluate."}
#                 },
#                 "required": ["expression"]
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_datetime",
#             "description": "Get the current exact date and time. Use this whenever the user asks about today's date or time.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {},
#                 "required": []
#             }
#         }
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "create_support_ticket",
#             "description": "Create a mock support ticket when human assistance is required.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "issue_description": {"type": "string", "description": "Description of the issue for the ticket."},
#                     "user_email": {"type": "string", "description": "User email address, defaults to user@company.com"}
#                 },
#                 "required": ["issue_description"]
#             }
#         }
#     }
# ]

# available_tools = {
#     "rag_search": rag_search,
#     "calculate": calculate,
#     "create_support_ticket": create_support_ticket,
#     "get_current_datetime": get_current_datetime
# }

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get('message')
#     chat_history = data.get('history', [])
    
#     if not user_message:
#         return jsonify({"error": "Please provide a message."}), 400

#     try:
#         messages = [
#             {"role": "system", "content": "You are a smart agentic assistant with access to tools. Use tools when necessary to answer user requests accurately."}
#         ]
        
#         for msg in chat_history:
#             messages.append(msg)
            
#         messages.append({"role": "user", "content": user_message})

#         for _ in range(3):
#             response = openai_client.chat.completions.create(
#                 model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
#                 messages=messages,
#                 tools=tools_definition,
#                 tool_choice="auto"
#             )
            
#             response_message = response.choices[0].message
            
#             if not response_message.tool_calls:
#                 return jsonify({"reply": response_message.content})
            
#             messages.append(response_message)
            
#             for tool_call in response_message.tool_calls:
#                 function_name = tool_call.function.name
#                 function_to_call = available_tools.get(function_name)
                
#                 if function_to_call:
#                     function_args = json.loads(tool_call.function.arguments)
#                     print(f"Executing tool: {function_name} with args: {function_args}")
#                     tool_output = function_to_call(**function_args)
                    
#                     messages.append({
#                         "tool_call_id": tool_call.id,
#                         "role": "tool",
#                         "name": function_name,
#                         "content": str(tool_output),
#                     })

#         return jsonify({"reply": "Maximum execution steps reached."})

#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# import os
# from flask import Flask, render_template, request, jsonify
# from advanced_rag import run_adaptive_rag

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     user_message = data.get('message')
    
#     if not user_message:
#         return jsonify({"error": "Please provide a message."}), 400

#     try:
#         # Advanced Adaptive RAG
#         result = run_adaptive_rag(user_query=user_message)
        
#         final_answer = result["final_answer"]
#         logs = result["logs"]
        
#         print("\n" + "="*50)
#         print(f"🗣️ USER QUERY: {user_message}")
#         for log in logs:
#             print(f"🧠 {log}")
#         print("\n📚 EXACT RAW CONTEXT FROM AZURE SEARCH:")
#         for doc in result["retrieved_docs"]:
#             print(f"   [Source ID: {doc['id']}] {doc['content']}")
#         print("="*50 + "\n")
        
#         return jsonify({"reply": final_answer})

#     except Exception as e:
#         print("Error:", e)
#         return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
import os
import re
from flask import Flask, render_template, request, jsonify
from advanced_rag import run_adaptive_rag
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import base64
import openai
load_dotenv()

app = Flask(__name__)

# Azure Speech Configuration
speech_key = os.getenv("AZURE_SPEECH_KEY")
speech_region = os.getenv("AZURE_SPEECH_REGION")
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = ""
    image_base64 = None
    is_voice_request = False
    
    # Check if an image was uploaded
    if 'image' in request.files:
        img_file = request.files['image']
        if img_file.filename != '':
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Handle incoming JSON or Form data
    if request.is_json:
        data = request.json or {}
        user_message = data.get('message', '')
        is_voice_request = data.get('is_voice', False)
    else:
        user_message = request.form.get('message', '')

    if not user_message and not image_base64:
        return jsonify({"error": "Please provide a message or an image."}), 400

    try:
        if image_base64:
            # Process Image Understanding & OCR task
            final_answer = process_image_agent(image_base64, user_message)
        else:
            # Run existing Advanced Adaptive RAG pipeline
            result = run_adaptive_rag(user_query=user_message)
            final_answer = result["final_answer"]
            
        print(f"\n🗣️ USER QUERY: {user_message} | Image Attached: {bool(image_base64)}")
        print(f"🤖 ANSWER: {final_answer}\n")
        
        response_data = {"reply": final_answer}
        
        # Handle TTS audio response if requested via voice
        if is_voice_request:
            has_arabic_letters = bool(re.search("[\u0600-\u06FF]", final_answer))
            if has_arabic_letters:
                speech_config.speech_synthesis_voice_name = "ar-EG-SalmaNeural"
            else:
                speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
                
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            tts_result = synthesizer.speak_text_async(final_answer).get()
            
            output_audio_path = "static/response_audio.wav"
            os.makedirs("static", exist_ok=True)
            with open(output_audio_path, "wb") as f:
                f.write(tts_result.audio_data)
                
            response_data["audio_url"] = f"/{output_audio_path}"

        return jsonify(response_data)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



def process_image_agent(image_base64, query):
    """
    Sends the image and query to Azure OpenAI GPT-4o Vision to perform 
    Image Understanding, OCR (Arabic/English), and answer questions.
    """
    if not query:
        query = "Analyze this image, describe what you see, and perform OCR if any text exists."

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": "You are an advanced AI Vision and OCR Assistant. Your job is to analyze uploaded images, perform accurate OCR for both Arabic and English text when present, understand visual objects/scenes, and answer the user's question directly in the same language as their query."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": query
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=1000
        )
        
        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        print(f"Azure Vision API Error: {e}")
        return f"Error processing image with Azure Vision: {str(e)}"
if __name__ == '__main__':
    app.run(debug=True)