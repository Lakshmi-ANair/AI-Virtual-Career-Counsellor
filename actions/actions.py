from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import os

# Import the Google Generative AI library
import google.generativeai as genai

# For loading environment variables
from dotenv import load_dotenv
load_dotenv()

# NLTK for preprocessing keywords
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- Initialize NLTK resources ---
NLTK_RESOURCES = ['wordnet', 'omw-1.4', 'stopwords', 'punkt']
resources_to_download = []
nltk_data_path_found = False

try:
    if nltk.data.path:
        nltk_data_path_found = True
except AttributeError:
    print("Warning: nltk.data.path not found, NLTK might not be configured correctly.")
    pass

if nltk_data_path_found:
    for resource in NLTK_RESOURCES:
        try:
            if resource == 'punkt':
                nltk.data.find(f'tokenizers/{resource}.zip')
            elif resource == 'omw-1.4':
                nltk.data.find(f'corpora/{resource}/omw-1.4.words')
            else:
                nltk.data.find(f'corpora/{resource}.zip')
            print(f"NLTK resource '{resource}' found.")
        except LookupError:
            print(f"NLTK resource '{resource}' not found. Adding to download list.")
            resources_to_download.append(resource)
        except Exception as e_find:
            print(f"Error finding NLTK resource '{resource}': {e_find}. Adding to download list.")
            if resource not in resources_to_download:
                resources_to_download.append(resource)
else:
    print("No NLTK data path configured or accessible. Assuming all resources need to be downloaded.")
    resources_to_download = NLTK_RESOURCES


if resources_to_download:
    print(f"Attempting to download NLTK resources: {', '.join(resources_to_download)}")
    all_downloads_successful = True
    for res_to_download in resources_to_download:
        try:
            print(f"Downloading '{res_to_download}'...")
            downloaded = nltk.download(res_to_download, quiet=False)
            if not downloaded:
                print(f"Warning: NLTK reported that download of '{res_to_download}' might not have been successful (returned False).")
                all_downloads_successful = False
            else:
                print(f"NLTK resource '{res_to_download}' downloaded (or was already up-to-date).")
        except Exception as download_exc:
            print(f"CRITICAL: Failed to download NLTK resource '{res_to_download}': {download_exc}")
            all_downloads_successful = False

    if all_downloads_successful:
        print("All pending NLTK resources seem to be downloaded successfully.")
    else:
        print("One or more NLTK resource downloads failed or were problematic.")
        print("Please ensure you have an internet connection and sufficient permissions.")
        print("You can also try downloading them manually in a Python interpreter:")
        for res in resources_to_download:
            print(f">>> import nltk; nltk.download('{res}')")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- LLM Configuration (for Gemini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("DEBUG: Gemini API key configured.")
else:
    print("CRITICAL: GEMINI_API_KEY not found in environment variables. Gemini API calls will fail.")

# Choose a Gemini model. Common ones are 'gemini-pro' or 'gemini-1.5-flash'
GEMINI_MODEL = "gemini-pro" # Or "gemini-1.5-flash" for a faster, cheaper option

# --- Career Data (remains the same) ---
CAREER_DATA = {
    "software engineer":{
        "keywords": ["coding", "tech", "programming", "computers", "software", "development", "problem solving", "cyber security"],
        "description": "Designs, develops, and maintains software applications, including security aspects.",
        "prompt_enhancer": "Explain what a software engineer does daily, including aspects of system security if relevant, and what makes it exciting for someone who likes problem-solving and technology."},
    "cyber security analyst": {
        "keywords": ["cyber security", "security", "network", "data protection", "threats", "hacking", "firewall", "problem solving", "tech"],
        "description": "Protects computer systems and networks from threats, analyzing security breaches and implementing defensive measures.",
        "prompt_enhancer": "Describe the critical role of a Cyber Security Analyst. What kind of challenges do they face, and why is it a vital career for someone interested in technology and protection?"
    },
    "graphic designer": {
        "keywords": ["art", "design", "visuals", "creative", "drawing", "illustration", "painting"],
        "description": "Creates visual concepts to communicate ideas that inspire, inform, or captivate consumers.",
        "prompt_enhancer": "Describe the world of a graphic designer. What kind of projects do they work on, and how does a passion for art like painting translate into this career?"},
    "ai engineer": {
        "keywords": ["ai", "artificial intelligence", "machine learning", "neural networks", "automation", "algorithms"],
        "description": "Builds AI systems that simulate human intelligence, helping automate and enhance decision-making processes.",
        "prompt_enhancer": "What does an AI Engineer do daily, and how can someone fascinated by intelligent systems and automation thrive in this role?"
    },
    "registered nurse": {
        "keywords": ["health", "care", "medicine", "patients", "hospital", "nursing", "empathy", "treatment"],
        "description": "Provides direct patient care, educates individuals about health conditions, and supports patient recovery.",
        "prompt_enhancer": "Can you explain the daily responsibilities of a Registered Nurse and how compassion and a desire to help others make it a meaningful career?"
    },
    "digital marketing specialist": {
        "keywords": ["marketing", "social media", "branding", "seo", "ads", "strategy", "creativity"],
        "description": "Uses online platforms to promote products, build brand awareness, and reach target audiences effectively.",
        "prompt_enhancer": "Describe what it's like to be a Digital Marketing Specialist. How do creativity and strategic thinking come into play?"
    },
    "ux ui designer": {
        "keywords": ["design", "user experience", "interface", "creativity", "usability", "prototyping", "research"],
        "description": "Designs intuitive and engaging user interfaces by understanding user behavior and creating visually appealing layouts.",
        "prompt_enhancer": "What is the role of a UX/UI Designer, and how does creativity meet functionality in their work?"
    },
    "educational technology specialist": {
        "keywords": ["education", "technology", "teaching", "e-learning", "tools", "training", "innovation"],
        "description": "Explain how an Educational Technology Specialist blends teaching and technology to improve how people learn in the digital age.",
        "prompt_enhancer": "Explain how an Educational Technology Specialist blends teaching and technology to improve how people learn in the digital age."
    },
    "financial analyst": {
        "keywords": ["finance", "numbers", "investing", "economics", "analytics", "business"],
        "description": "Analyzes financial data to guide investment decisions and business strategies.",
        "prompt_enhancer": "Explain the role of a Financial Analyst and how strong analytical skills and an interest in numbers can lead to success in this field."
    },
    "accountant": {
        "keywords": ["accounting", "numbers", "finance", "auditing", "bookkeeping", "business"],
        "description": "Manages and interprets financial records for businesses and individuals.",
        "prompt_enhancer": "What does an accountant do, and why is attention to detail and an understanding of financial regulations key to this career?"
    },
}

def preprocess_text(text: str) -> List[str]:
    if not text:
        return []
    try:
        tokens = nltk.word_tokenize(text.lower())
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
        filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]
        return filtered_tokens
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return []


class ActionRecommendCareerEnhanced(Action):
    def name(self) -> Text:
        return "action_recommend_career_enhanced"

    async def run(self, dispatcher: CollectingDispatcher,
                tracker: Tracker,
                domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("DEBUG: ActionRecommendCareerEnhanced started.")
        user_interest_keywords_slot = tracker.get_slot("interest_keywords")
        print(f"DEBUG: Slot 'interest_keywords' value: {user_interest_keywords_slot}")

        if not user_interest_keywords_slot:
            dispatcher.utter_message(response="utter_ask_interest")
            print("DEBUG: No interest keywords found in slot, uttered utter_ask_interest.")
            return []

        if isinstance(user_interest_keywords_slot, str):
            user_interest_keywords = [user_interest_keywords_slot]
        elif isinstance(user_interest_keywords_slot, list):
            user_interest_keywords = user_interest_keywords_slot
        else:
            user_interest_keywords = []
            print(f"DEBUG: interest_keywords slot was not a string or list: {user_interest_keywords_slot}")

        processed_user_interests = set()
        if user_interest_keywords:
            for interest_phrase in user_interest_keywords:
                if isinstance(interest_phrase, str):
                    processed_user_interests.update(preprocess_text(interest_phrase))
                else:
                    print(f"DEBUG: Skipping non-string interest phrase: {interest_phrase}")

        print(f"DEBUG: Processed user interests: {processed_user_interests}")

        if not processed_user_interests and user_interest_keywords:
            dispatcher.utter_message(text="I couldn't process the keywords from your interests. Could you try phrasing them differently, e.g., 'I enjoy coding' or 'I'm interested in art'?")
            print("DEBUG: Keywords provided but not processed, uttered custom message.")
            return []
        elif not processed_user_interests:
            dispatcher.utter_message(response="utter_ask_interest")
            print("DEBUG: No keywords processed, uttered utter_ask_interest.")
            return []

        recommendations = []
        for career, data in CAREER_DATA.items():
            career_keywords_processed = set()
            for kw in data.get("keywords", []):
                career_keywords_processed.update(preprocess_text(kw))

            match_score = len(processed_user_interests.intersection(career_keywords_processed))
            if match_score > 0:
                recommendations.append({
                    "career": career.title(),
                    "base_description": data.get("description", "No description available."),
                    "prompt_enhancer": data.get("prompt_enhancer", f"Tell me more about being a {career.title()}."),
                    "score": match_score
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        top_n_recommendations = recommendations[:1]

        if top_n_recommendations:
            rec = top_n_recommendations[0]
            user_interests_str = ", ".join(user_interest_keywords) if user_interest_keywords else "your interests"
            print(f"DEBUG: Top recommendation: {rec['career']}")

            if not GEMINI_API_KEY: # Check for Gemini API key
                print("DEBUG: Gemini API key is NOT configured. Falling back to basic description.")
                dispatcher.utter_message(text="LLM API key not configured. Falling back to basic description.")
                message = f"Based on your interests in: {user_interests_str},\n"
                message += f"You might like **{rec['career']}**: {rec['base_description']}\n"
                dispatcher.utter_message(text=message)
                dispatcher.utter_message(response="utter_ask_for_more_details")
                return []
            else:
                print(f"DEBUG: Gemini API key IS configured. Attempting LLM call for {rec['career']}.")

            try:
                prompt_to_llm = (
                    f"You are a friendly and encouraging AI career counsellor. "
                    f"A user has expressed interest in '{user_interests_str}'. "
                    f"Based on this, a potentially suitable career is '{rec['career']}'. "
                    f"{rec['prompt_enhancer']} "
                    f"Keep the tone positive and conversational. Make it sound like a natural continuation of our chat. Do not start with 'Okay' or 'Sure'. Be concise but informative, around 100-150 words."
                )
                print(f"DEBUG: Prompt to LLM for {rec['career']}: {prompt_to_llm}")

                # --- Gemini API Call ---
                model = genai.GenerativeModel(GEMINI_MODEL)
                response = model.generate_content(prompt_to_llm)
                llm_response = response.text.strip() # Get the text content from Gemini's response

                print(f"DEBUG: LLM Response for {rec['career']}: {llm_response}")

                intro_message = f"Since you're interested in {user_interests_str}, let's talk about becoming a **{rec['career']}**!"
                dispatcher.utter_message(text=intro_message)
                dispatcher.utter_message(text=llm_response)
                dispatcher.utter_message(response="utter_ask_for_more_details")

            except Exception as e:
                print(f"CRITICAL ERROR calling Gemini or processing response for {rec['career']}: {e}")
                dispatcher.utter_message(text=f"I had a little trouble getting detailed info for {rec['career']} right now, but here's a basic idea: {rec['base_description']}")
                dispatcher.utter_message(response="utter_ask_for_more_details")
        else:
            print("DEBUG: No recommendations found after processing.")
            dispatcher.utter_message(response="utter_no_recommendation")

        print("DEBUG: ActionRecommendCareerEnhanced finished.")
        return []