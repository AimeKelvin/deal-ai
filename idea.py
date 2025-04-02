import requests
from bs4 import BeautifulSoup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json

# Voice feature config (set to True to enable voice, False to disable)
ENABLE_VOICE = True  # Change to True if you want the analyzer to speak
if ENABLE_VOICE:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('voice', 'english_rp+m3')  # Set to a male voice
    except ImportError:
        print("I’d speak if I could, but pyttsx3 isn’t installed. Try 'pip install pyttsx3'.")
        ENABLE_VOICE = False

# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Cache file for web results
CACHE_FILE = "idea_analyzer_cache.json"

# Fetch web data (or use cache) based on the idea description
def fetch_web_data(query):
    cache = load_cache()
    query_key = query.lower().strip()
    if query_key in cache:
        return cache[query_key]
    try:
        url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")
        snippets = soup.select(".result__snippet")
        text = " ".join([snippet.get_text() for snippet in snippets[:3]])
        if not text:
            text = "No relevant data found."
        cache[query_key] = text
        save_cache(cache)
        return text
    except Exception as e:
        return f"Error accessing the web: {str(e)}"

# Load cache from file
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

# Save cache to file
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

# Generate analysis response
def generate_response(idea, web_text):
    prompt = f"{web_text}\n\nIdea: {idea}\n\nAnalysis: "
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=200)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    generated = input_ids
    for _ in range(50):  # Generate up to 50 tokens
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

# Idea analysis loop
def analyze_ideas():
    print("Welcome to the Idea Analyzer! Input an idea for analysis or type 'quit' to exit.")
    while True:
        idea = input("Enter your idea: ")
        if idea.lower() in ["quit", "exit"]:
            print("Goodbye! Thanks for using the Idea Analyzer.")
            break
        web_text = fetch_web_data(idea)
        analysis = generate_response(idea, web_text)
        print(f"\n{analysis}\n")
        if ENABLE_VOICE:
            engine.say(analysis)
            engine.runAndWait()

if __name__ == "__main__":
    analyze_ideas()
