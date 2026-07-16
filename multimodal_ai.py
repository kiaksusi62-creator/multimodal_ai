import os
import json
import tempfile
import gradio as gr
import requests
import replicate
from openai import OpenAI
from PyPDF2 import PdfReader
from huggingface_hub import InferenceClient
from anthropic import Anthropic
import google.generativeai as genai
import cohere

# ==================== 1. API-Clients initialisieren ====================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
co_client = cohere.Client(os.getenv("COHERE_API_KEY"))
anthro = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# HuggingFace Clients
hf_mistral = InferenceClient("mistralai/Mistral-7B-Instruct-v0.3", token=os.getenv("HF_TOKEN"))
hf_llama3 = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=os.getenv("HF_TOKEN"))
hf_sd = InferenceClient("stabilityai/stable-diffusion-xl-base-1.0", token=os.getenv("HF_TOKEN"))
hf_whisper = InferenceClient("openai/whisper-large-v3", token=os.getenv("HF_TOKEN"))
hf_bark = InferenceClient("suno/bark", token=os.getenv("HF_TOKEN"))

# OpenAI-kompatible Clients
openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
together_client = OpenAI(base_url="https://api.together.xyz/v1", api_key=os.getenv("TOGETHER_API_KEY"))
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))
deepseek_client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY"))
deepseek_coder_client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=os.getenv("DEEPSEEK_API_KEY"))

# Replicate (für Video)
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN", "")

# Globaler Kontext für Dokumente
document_context = ""

# ==================== 2. Modell-Funktionen ====================

# --- Textmodelle ---
def gemini_flash(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

def cohere_command(prompt):
    resp = co_client.generate(model='command-light', prompt=prompt, max_tokens=300)
    return resp.generations[0].text

def claude_haiku(prompt):
    msg = anthro.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text

def hf_mistral_7b(prompt):
    return hf_mistral.text_generation(prompt, max_new_tokens=300)

def hf_llama3_8b(prompt):
    return hf_llama3.text_generation(prompt, max_new_tokens=300)

def openrouter_llama3_1_free(prompt):
    return openrouter_client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def together_llama3(prompt):
    return together_client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def cloudflare_llama3(prompt):
    model = "@cf/meta/llama-3-8b-instruct"
    url = f"https://api.cloudflare.com/client/v4/accounts/{os.getenv('CF_ACCOUNT_ID')}/ai/run/{model}"
    headers = {"Authorization": f"Bearer {os.getenv('CF_API_TOKEN')}"}
    resp = requests.post(url, json={"messages":[{"role":"user","content":prompt}]})
    return resp.json()["result"]["response"]

def groq_llama3_instant(prompt):
    return groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def groq_mixtral(prompt):
    return groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# --- Code-Modelle ---
def deepseek_coder(prompt):
    return deepseek_coder_client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def openrouter_codellama_free(prompt):
    return openrouter_client.chat.completions.create(
        model="codellama/llama-3.1-8b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# --- Bildgenerierung ---
def generate_image_sd(prompt):
    image = hf_sd.text_to_image(prompt)
    path = tempfile.mktemp(suffix=".png")
    image.save(path)
    return path

# --- Sprachsynthese ---
def text_to_speech(text):
    audio = hf_bark.text_to_speech(text)  # gibt URL zurück
    return f'<audio controls src="{audio}"></audio>'

# --- Spracherkennung (separat) ---
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        data = f.read()
    result = hf_whisper.automatic_speech_recognition(data)
    return result["text"]

# --- Video-Analyse (Replicate) ---
def analyze_video(video_path, question):
    if not os.getenv("REPLICATE_API_TOKEN"):
        return "❌ Replicate-Token nicht gesetzt."
    output = replicate.run(
        "nvidia/vila",
        input={
            "video": open(video_path, "rb"),
            "question": question
        }
    )
    return output

# --- Dokumenten-Analyse ---
def extract_text_from_pdf(pdf_file):
    global document_context
    reader = PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    document_context = text
    return f"✅ PDF geladen – {len(text)} Zeichen Kontext gespeichert."

# ==================== 3. Intelligenter Router ====================
def classify_intent(prompt, has_document=False, has_video=False):
    if has_document and not prompt.startswith("/"):
        return "document_qa"
    if prompt.startswith("/speak"):
        return "audio_speech"
    if prompt.startswith("/image"):
        return "image"
    if has_video and not prompt.startswith("/"):
        return "video_question"
    system = """
    Du bist ein Router. Analysiere die folgende Benutzereingabe und antworte NUR mit einem JSON:
    {"intent": "text"|"code"|"image"|"audio_speech"|"document_qa"}
    Regeln:
    - "image": wenn der Nutzer ein Bild generieren lassen will (z.B. "erstell ein Bild von...", "male mir...")
    - "code": wenn es um Programmierung, Code, Bugfixing, Algorithmen geht
    - "audio_speech": wenn der Nutzer eine Sprachausgabe wünscht (z.B. "sprich: ...")
    - sonst "text"
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"{system}\n\nEingabe: {prompt}")
        return json.loads(response.text.strip())["intent"]
    except:
        return "text"

# ==================== 4. Fallback-Listen ====================
TEXT_MODELS = [
    ("Gemini Flash", gemini_flash),
    ("Groq Llama3", groq_llama3_instant),
    ("HF Mistral", hf_mistral_7b),
    ("HF Llama3", hf_llama3_8b),
    ("Together Llama3", together_llama3),
    ("Cloudflare Llama3", cloudflare_llama3),
    ("Cohere Command", cohere_command),
    ("Claude Haiku", claude_haiku),
    ("OpenRouter Llama3.1 Free", openrouter_llama3_1_free),
]

CODE_MODELS = [
    ("DeepSeek Coder", deepseek_coder),
    ("OpenRouter CodeLlama Free", openrouter_codellama_free),
    ("Groq Mixtral", groq_mixtral),
]

def try_models(model_list, prompt):
    for name, func in model_list:
        try:
            return func(prompt)
        except Exception:
            continue
    return "❌ Kein Modell der Kategorie verfügbar."

# ==================== 5. Haupt-Chat-Funktion ====================
def chat_with_router(message, history, pdf_state, video_state):
    global document_context
    has_document = pdf_state and document_context
    has_video = video_state

    if message.startswith("/image"):
        prompt = message[7:].strip()
        try:
            path = generate_image_sd(prompt)
            return f"![Bild]({path})"
        except Exception as e:
            return f"❌ Bildgenerierung fehlgeschlagen: {e}"

    if message.startswith("/speak"):
        text = message[7:].strip()
        try:
            return text_to_speech(text)
        except Exception as e:
            return f"❌ Sprachsynthese fehlgeschlagen: {e}"

    intent = classify_intent(message, has_document, has_video)

    if intent == "document_qa" and has_document:
        full_prompt = f"Kontext:\n{document_context[:3000]}\n\nFrage: {message}\nAntworte basierend auf dem Dokument."
        return try_models(TEXT_MODELS, full_prompt)

    if intent == "video_question" and has_video:
        return "Bitte nutze den Video-Tab, um deine Frage zum Video zu stellen."

    if intent == "code":
        return try_models(CODE_MODELS, message)

    return try_models(TEXT_MODELS, message)

# ==================== 6. Gradio UI ====================
with gr.Blocks(title="Multimodale KI – alle Modelle per API") as demo:
    gr.Markdown("# 🌐 Multimodale KI – Text, Code, Bild, Sprache, Video, PDF\n*Alle Modelle kostenlos per API – automatisch gewählt*")

    pdf_loaded = gr.State(False)
    video_loaded = gr.State(False)

    with gr.Tab("💬 Chat"):
        gr.Markdown("Schreib einfach drauf los, z. B. *„Schreib ein Python‑Script“*, *„Erstell ein Bild von…“* oder *„/speak Guten Morgen“*.")
        chatbot = gr.ChatInterface(
            fn=chat_with_router,
            additional_inputs=[pdf_loaded, video_loaded]
        )

    with gr.Tab("📄 PDF‑Analyse"):
        gr.Markdown("Lade eine PDF hoch und stelle dann im **Chat‑Tab** Fragen dazu.")
        pdf_upload = gr.File(label="PDF hochladen", file_types=[".pdf"])
        pdf_status = gr.Textbox(label="Status")
        upload_btn = gr.Button("PDF verarbeiten")
        upload_btn.click(
            fn=extract_text_from_pdf,
            inputs=pdf_upload,
            outputs=pdf_status
        ).then(fn=lambda: True, outputs=pdf_loaded)

    with gr.Tab("🎤 Audio‑Transkription"):
        gr.Markdown("Sprich etwas ein oder lade eine Audiodatei hoch.")
        audio_in = gr.Audio(type="filepath", label="Audio")
        transcribe_btn = gr.Button("Transkribieren")
        transcribe_out = gr.Textbox(label="Text")
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=audio_in,
            outputs=transcribe_out
        )

    with gr.Tab("🎥 Video‑Frage"):
        gr.Markdown("Lade ein kurzes Video hoch und stelle eine Frage dazu (benötigt Replicate‑Token).")
        video_file = gr.Video(label="Video")
        video_question = gr.Textbox(label="Frage zum Video")
        video_answer_btn = gr.Button("Antwort holen")
        video_answer = gr.Textbox(label="Antwort")
        video_answer_btn.click(
            fn=analyze_video,
            inputs=[video_file, video_question],
            outputs=video_answer
        ).then(fn=lambda: True, outputs=video_loaded)

demo.launch()
