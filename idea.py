# Block 1: Basic Setup and Input
print("Welcome to the Idea Analyzer AI (with Research)!")
print("Enter your idea (e.g., 'A solar-powered phone charger in rural areas'): ")
user_idea = input("> ")
print(f"\nYou entered: {user_idea}")

# Block 2: Load Models and Define Criteria
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
model_name = "cerebras/Cerebras-GPT-111M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
research_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
area_analyzer = SentenceTransformer("paraphrase-MiniLM-L6-v2")

creativity_keywords = ["new", "unique", "creative", "innovative", "original", "fun", "smart"]
feasibility_keywords = ["easy", "simple", "cheap", "fast", "practical", "available"]
feasibility_negatives = ["hard", "expensive", "complex", "slow", "impossible"]
continuity_keywords = ["sustainable", "long-term", "durable", "ongoing", "renewable"]
continuity_negatives = ["short-term", "temporary", "one-time"]
default_area = "Rwanda"

def find_area(idea):
    idea = idea.lower()
    if " in " in idea:
        parts = idea.split(" in ")
        if len(parts) > 1:
            return parts[1].strip()
    return default_area

# Block 3: Research and Analyze the Idea
def analyze_idea(idea):
    idea = idea.lower()
    area = find_area(idea)
    
    sentiment = sentiment_analyzer(idea)[0]
    creativity_base = 5 + int(sentiment["score"] * 5) if sentiment["label"] == "POSITIVE" else 5 - int(sentiment["score"] * 5)
    creativity_boost = sum(1 for word in creativity_keywords if word in idea)
    creativity_score = min(10, creativity_base + creativity_boost)
    
    feasibility_pos = sum(1 for word in feasibility_keywords if word in idea)
    feasibility_neg = sum(1 for word in feasibility_negatives if word in idea)
    feasibility_score = min(10, max(0, 5 + feasibility_pos - feasibility_neg))
    
    continuity_pos = sum(1 for word in continuity_keywords if word in idea)
    continuity_neg = sum(1 for word in continuity_negatives if word in idea)
    continuity_score = min(10, max(0, 5 + continuity_pos - continuity_neg))
    
    research_prompt = f"Briefly research: Is '{idea}' a good idea in {area}?"
    research_output = research_generator(research_prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
    
    idea_embedding = area_analyzer.encode(idea)
    area_embedding = area_analyzer.encode(f"Conditions in {area}")
    area_fit = torch.cosine_similarity(
        torch.tensor(idea_embedding).unsqueeze(0),
        torch.tensor(area_embedding).unsqueeze(0)
    ).item()
    area_score = min(10, int(area_fit * 10))
    
    overall_score = (creativity_score + feasibility_score + continuity_score + area_score) // 4
    
    return {
        "creativity": creativity_score,
        "feasibility": feasibility_score,
        "continuity": continuity_score,
        "area": area,
        "area_fit": area_score,
        "research": research_output,
        "overall": overall_score
    }

results = analyze_idea(user_idea)
print("\nAnalysis Results:")
print(f"Creativity: {results['creativity']}/10")
print(f"Feasibility: {results['feasibility']}/10")
print(f"Continuity: {results['continuity']}/10")
print(f"Area Fit: {results['area_fit']}/10")
print(f"Area: {results['area']}")
print(f"Research Insight: {results['research']}")
print(f"Overall Score: {results['overall']}/10")

# Block 4: Provide Feedback
def give_feedback(results):
    feedback = f"\nIdea Analysis for '{user_idea}':\n"
    feedback += f"Overall Score: {results['overall']}/10\n"
    if results['creativity'] >= 7:
        feedback += "- Creativity: Super original!\n"
    elif results['creativity'] >= 4:
        feedback += "- Creativity: Fairly unique.\n"
    else:
        feedback += "- Creativity: Needs more spark.\n"
    if results['feasibility'] >= 7:
        feedback += "- Feasibility: Easy to do!\n"
    elif results['feasibility'] >= 4:
        feedback += "- Feasibility: Doable with effort.\n"
    else:
        feedback += "- Feasibility: Tough—simplify it.\n"
    if results['continuity'] >= 7:
        feedback += "- Continuity: Long-lasting!\n"
    elif results['continuity'] >= 4:
        feedback += "- Continuity: Moderately sustainable.\n"
    else:
        feedback += "- Continuity: Short-lived—think ahead.\n"
    feedback += f"- Area: {results['area']} (Fit: {results['area_fit']}/10)\n"
    if results['area_fit'] >= 7:
        feedback += "  Fits the area well!\n"
    elif results['area_fit'] >= 4:
        feedback += "  Decent fit for the area.\n"
    else:
        feedback += "  Might not suit the area—reconsider.\n"
    if results['area'] == "Rwanda" and " in " not in user_idea.lower():
        feedback += "  (No area given, defaulted to Rwanda.)\n"
    feedback += f"- Research Insight: {results['research']}\n"
    return feedback

feedback = give_feedback(results)
print(feedback)