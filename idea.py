import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json
import asyncio
import edge_tts
from gtts import gTTS

# Voice feature config
ENABLE_VOICE = True  # Set to False to disable voice
USE_ONLINE_VOICE = False  # Set to True for online (gTTS), False for offline (Edge TTS)

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Cache file for web results
CACHE_FILE = "idea_analyzer_cache.json"

def fetch_web_data(query):
    cache = load_cache()
    query_key = query.lower().strip()
    if query_key in cache:
        print("Recalling previously fetched data...")
        return cache[query_key]
    try:
        url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.select(".result__snippet")
        text = " ".join([snippet.get_text() for snippet in snippets[:3]])
        if not text:
            text = "I couldn’t find much on that topic."
        cache[query_key] = text
        save_cache(cache)
        return text
    except Exception as e:
        return f"Error accessing the web: {str(e)}"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def generate_response(idea, web_text):
    personality = ("You’re an Idea Analyzer AI. Evaluate the idea on several dimensions: "
                   "creativity, feasibility, environmental impact, opportunity, need, and geolocation. "
                   "Provide a thoughtful, detailed analysis and suggestions based only on the provided web data.")
    prompt = f"{personality}\n\nWeb data: {web_text}\n\nIdea: {idea}\n\nAnalysis: "
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=200)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    generated = input_ids
    for _ in range(50):
        outputs = model(generated, attention_mask=attention_mask)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        generated = torch.cat((generated, next_token), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1))), dim=1)
        decoded_token = tokenizer.decode(next_token[0])
        if decoded_token in ['.', '!', '?']:
            break
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = response.split("Analysis: ")[-1].strip()
    return answer

async def speak_edge_tts(text):
    tts = edge_tts.Communicate(text, "en-US-JennyNeural")
    await tts.save("response.mp3")
    os.system("start response.mp3")

def speak_gtts(text):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save("response.mp3")
    os.system("start response.mp3")

def play_audio(text):
    if USE_ONLINE_VOICE:
        speak_gtts(text)
    else:
        asyncio.run(speak_edge_tts(text))

def analyze_ideas():
    print("Welcome to the Idea Analyzer! Input an idea for analysis or type 'quit' to exit.")
    while True:
        idea = input("Enter your idea: ")
        if idea.lower() in ["quit", "exit"]:
            print("Goodbye! Thanks for using the Idea Analyzer.")
            break
        
        web_text = fetch_web_data(idea)
        print(f"Debug: Retrieved web data: '{web_text[:100]}...'")
        analysis = generate_response(idea, web_text)
        print(f"\nAnalysis:\n{analysis}\n")
        
        if ENABLE_VOICE:
            play_audio(analysis)

if __name__ == "__main__":
    analyze_ideas()
