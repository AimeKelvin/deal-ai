import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json
import edge_tts
import asyncio

# Voice feature config (set to True to enable voice, False to disable)
ENABLE_VOICE = True  # Change to True if you want Alphonse to speak
if ENABLE_VOICE:
    try:
        # No additional downloads needed; Edge TTS uses online voices
        pass
    except ImportError:
        print("Alphonse: I’d speak if I could, but edge-tts isn’t installed. Try 'pip install edge-tts'.")
        ENABLE_VOICE = False

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Cache file
CACHE_FILE = "alphonse_cache.json"

# Fetch web data (or use cache)
def fetch_web_data(query):
    cache = load_cache()
    query_key = query.lower().strip()
    if query_key in cache:
        print("Alphonse: Let me recall that for you...")
        return cache[query_key]
    try:
        url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.select(".result__snippet")
        text = " ".join([snippet.get_text() for snippet in snippets[:3]])
        if not text:
            text = "I couldn’t find much on that, I’m afraid."
        cache[query_key] = text
        save_cache(cache)
        return text
    except Exception as e:
        return f"Alphonse: Seems the web’s playing hard to get. Error: {str(e)}"

# Load cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save cache
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Generate response with complete sentences
def generate_response(question, web_text):
    personality = "You’re Alphonse, a calm and thoughtful AI who answers based ONLY on the web data provided, in a moderate and gentle tone."
    prompt = f"{personality}\nWeb data: {web_text}\nQuestion: {question}\nAnswer: "
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=200)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    generated = input_ids
    for _ in range(50):  # Max 50 tokens
        outputs = model(generated, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat((generated, next_token), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1))), dim=1)
        decoded_token = tokenizer.decode(next_token[0])
        if decoded_token in ['.', '!', '?']:
            break
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = response.split("Answer: ")[-1].strip()
    return answer

# Async function to handle Edge TTS audio generation
async def speak_with_edge_tts(text):
    voice = "en-US-GuyNeural"  # Natural male voice
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("output.mp3")
    os.system("start output.mp3" if os.name == "nt" else "aplay output.mp3" if os.name == "posix" else "afplay output.mp3")

# Chat loop
def chat_with_alphonse():
    print("Alphonse: Hello there, I’m here to help you out. What’s on your mind?")
    while True:
        question = input("You: ")
        if question.lower() in ["quit", "exit"]:
            print("Alphonse: Take care, friend. Until next time.")
            break
        web_text = fetch_web_data(question)
        print(f"Debug: Web data = '{web_text[:100]}...'")
        answer = generate_response(question, web_text)
        print(f"Alphonse: {answer}")
        if ENABLE_VOICE:
            # Run the async Edge TTS function
            asyncio.run(speak_with_edge_tts(answer))

if __name__ == "__main__":
    chat_with_alphonse()