import os
import time
import logging
import sys
import config
import requests
from dotenv import load_dotenv

# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("compare.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("compare")

load_dotenv()
NVIDIA_API_URL = os.getenv("INVOKE_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
if not NVIDIA_API_URL.endswith("/chat/completions"):
    NVIDIA_API_URL = NVIDIA_API_URL.rstrip("/") + "/chat/completions"
GEMMA_API_KEY = os.getenv("GEMMA_MODEL_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_MODEL_API_KEY")

try:
    from sentence_transformers import SentenceTransformer, util
    logger.info("Loading Semantic Similarity Model (all-MiniLM-L6-v2)...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Failed to load sentence-transformers: {e}")
    semantic_model = None

# Mock test data based on the original manual layout
test_chunks = [
    {
        "parent_heading": "Safety Devices",
        "text": "The incubators are equipped with the following safety features: a sample protection feature that safeguards the samples against destruction through overheating in case of controller failure; dual fuses rated at 16 amperes."
    },
    {
        "parent_heading": "Sensing and Control System",
        "text": "The PT 100-type sensor for the control of the work space temperature and for the thermal protection [1] is installed on the bottom of table-top units."
    }
]

def call_nvidia_api(prompt, model, api_key):
    if not api_key:
        return "API_KEY_MISSING"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    try:
        res = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        return res.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"ERROR: {e}"

def contextual_translate(text, parent_heading, model, api_key):
    prompt = (
        f"You are a technical translator. Translate the following text into German. "
        f"The text belongs to the section titled: '{parent_heading}'. "
        f"Do not translate brand names. Use the following glossary for technical terms: "
        f"{{'Table-top units': 'Tischgeräte', 'Floor stand units': 'Bodenstandeinheiten', 'PT 100-type sensor': 'PT 100-Sensor'}}. "
        f"Provide ONLY the translated text, without any additional explanations.\n\n"
        f"Text:\n{text}"
    )
    return call_nvidia_api(prompt, model, api_key)

def back_translate(text, model, api_key):
    prompt = f"Translate the following German text back to English. Provide ONLY the translated text.\n\nText:\n{text}"
    return call_nvidia_api(prompt, model, api_key)

def llm_judge(original, translated, api_key):
    prompt = (
        f"Rate this translation from 1 to 5 based on whether it perfectly preserves the technical instructions and context of the original text. "
        f"If it hallucinates, misses glossary terms, or is completely wrong, output 1. ONLY output the integer number (1, 2, 3, 4, or 5), nothing else.\n\n"
        f"Original English: {original}\n"
        f"German Translation: {translated}"
    )
    res = call_nvidia_api(prompt, config.LLAMA_MODEL, api_key)
    try:
        # Extract just the digit
        return int(''.join(filter(str.isdigit, res))[0])
    except:
        return 0

def calculate_similarity(text1, text2):
    if semantic_model:
        emb1 = semantic_model.encode(text1, convert_to_tensor=True)
        emb2 = semantic_model.encode(text2, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2)[0][0])
    return 0.0

class NLLBSimple:
    def __init__(self, model_name):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        logger.info(f"Loading local NLLB model {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
    def translate_to_german(self, text):
        self.tokenizer.src_lang = "eng_Latn"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        translated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id["deu_Latn"], max_length=1024)
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
    def translate_to_english(self, text):
        self.tokenizer.src_lang = "deu_Latn"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        translated_tokens = self.model.generate(**inputs, forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"], max_length=1024)
        return self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def main():
    logger.info("\n" + "="*80)
    logger.info(" ADVANCED CONTEXTUAL TRANSLATION BENCHMARK ".center(80, "="))
    logger.info("="*80 + "\n")
    
    models_to_test = [
        ("Gemma (Context-Aware)", config.GEMMA_MODEL, GEMMA_API_KEY),
        ("Llama (Context-Aware)", config.LLAMA_MODEL, LLAMA_API_KEY)
    ]
    
    try:
        nllb = NLLBSimple(config.NLLB_MODEL)
    except Exception as e:
        logger.error(f"Failed to load NLLB: {e}")
        nllb = None

    stats = {
        "Gemma (Context-Aware)": {"time": [], "sim": [], "score": []},
        "Llama (Context-Aware)": {"time": [], "sim": [], "score": []},
        "NLLB (Zero-Context)": {"time": [], "sim": [], "score": []}
    }
    
    for i, chunk in enumerate(test_chunks):
        logger.info(f"\n[{i+1}/{len(test_chunks)}] Testing Context: '{chunk['parent_heading']}'")
        logger.info(f"Original Text: {chunk['text']}\n")
        
        # 1. Evaluate Context-Aware Models (Gemma/Llama)
        for m_name, m_path, m_key in models_to_test:
            logger.info(f"Starting evaluation for {m_name}...")
            
            # Forward Translation
            logger.info(f"  -> Step 1: Forward translating to German using {m_path}...")
            t0 = time.time()
            ger_text = contextual_translate(chunk['text'], chunk['parent_heading'], m_path, m_key)
            t_forward = time.time() - t0
            
            # Back Translation
            logger.info(f"  -> Step 2: Back-translating to English...")
            t0 = time.time()
            eng_text = back_translate(ger_text, m_path, m_key)
            t_back = time.time() - t0
            
            logger.info(f"  -> Step 3: Calculating Semantic Similarity...")
            sim = calculate_similarity(chunk['text'], eng_text)
            
            logger.info(f"  -> Step 4: Running LLM-as-a-Judge...")
            score = llm_judge(chunk['text'], ger_text, LLAMA_API_KEY)
            
            stats[m_name]["time"].append(t_forward + t_back)
            stats[m_name]["sim"].append(sim)
            stats[m_name]["score"].append(score)
            
            logger.info(f"  [RESULTS] {m_name}:")
            logger.info(f"    - German: {ger_text}")
            logger.info(f"    - Back-Tr: {eng_text}")
            logger.info(f"    - Metrics -> Time: {t_forward+t_back:.2f}s | Semantic Sim: {sim:.2f} | LLM Judge: {score}/5\n")

        # 2. Evaluate Zero-Context Local Model (NLLB)
        if nllb:
            m_name = "NLLB (Zero-Context)"
            logger.info(f"Starting evaluation for {m_name}...")
            
            logger.info(f"  -> Step 1: Forward translating to German...")
            t0 = time.time()
            ger_text = nllb.translate_to_german(chunk['text'])
            t_forward = time.time() - t0
            
            logger.info(f"  -> Step 2: Back-translating to English...")
            t0 = time.time()
            eng_text = nllb.translate_to_english(ger_text)
            t_back = time.time() - t0
            
            logger.info(f"  -> Step 3: Calculating Semantic Similarity...")
            sim = calculate_similarity(chunk['text'], eng_text)
            
            logger.info(f"  -> Step 4: Running LLM-as-a-Judge...")
            score = llm_judge(chunk['text'], ger_text, LLAMA_API_KEY)
            
            stats[m_name]["time"].append(t_forward + t_back)
            stats[m_name]["sim"].append(sim)
            stats[m_name]["score"].append(score)
            
            logger.info(f"  [RESULTS] {m_name}:")
            logger.info(f"    - German: {ger_text}")
            logger.info(f"    - Back-Tr: {eng_text}")
            logger.info(f"    - Metrics -> Time: {t_forward+t_back:.2f}s | Semantic Sim: {sim:.2f} | LLM Judge: {score}/5\n")

    logger.info("\n" + "="*80)
    logger.info(" FINAL MODEL COMPARISON STATISTICS ".center(80, "="))
    logger.info("="*80)
    
    for m_name in stats:
        if not stats[m_name]["time"]:
            continue
        avg_time = sum(stats[m_name]["time"]) / len(stats[m_name]["time"])
        avg_sim = sum(stats[m_name]["sim"]) / len(stats[m_name]["sim"])
        avg_score = sum(stats[m_name]["score"]) / len(stats[m_name]["score"])
        
        logger.info(f"\n{m_name.upper()}")
        logger.info(f"  Average RTT Latency    : {avg_time:.2f} seconds")
        logger.info(f"  Semantic Integrity     : {avg_sim*100:.1f}%")
        logger.info(f"  LLM Judge Score (1-5)  : {avg_score:.1f}/5.0")
        
    logger.info("\nBenchmark complete.")

if __name__ == '__main__':
    main()
