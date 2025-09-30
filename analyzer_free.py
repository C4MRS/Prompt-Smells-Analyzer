import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import textstat
import language_tool_python
import re
import math

# -----------------------------
# Configurazione modello GPT-Neo
# -----------------------------
MODEL_NAME = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

MODEL_MAX_TOKENS = 2048  # limite GPT-Neo 125M

# -----------------------------
# Configurazioni metriche
# -----------------------------
WC_MAX = 60
FORMALITY_TARGET = 0.5
tool = language_tool_python.LanguageTool('en-US')

# -----------------------------
# Funzioni metriche
# -----------------------------
def grammar_score(prompt):
    matches = tool.check(prompt)
    return max(0, 1 - len(matches) / max(1, len(prompt.split())))

def formatting_score(prompt):
    bad_formatting = 0
    if prompt.islower() or prompt.isupper():
        bad_formatting += 1
    if not any(p in prompt for p in ['.', '?', '!']):
        bad_formatting += 1
    return max(0, 1 - bad_formatting / 2)

def clarity_score(prompt):
    readability = textstat.flesch_reading_ease(prompt)
    return min(1, max(0, readability / 100))

def pqs(prompt):
    G = grammar_score(prompt)
    F = formatting_score(prompt)
    C = clarity_score(prompt)
    return (G + F + C) / 3, G, F, C

def complexity_length_score(prompt):
    word_count = len(prompt.split())
    fog = textstat.gunning_fog(prompt)
    score = (word_count / WC_MAX + fog / 20) / 2
    return max(0, 1 - min(score, 1))

# -----------------------------
# Funzione GPT-Neo per metriche testuali
# -----------------------------
def gpt_neo_yesno(prompt, instruction):
    # Usa GPT-Neo per rispondere con Yes o No.
    # Ritorna 1.0 se Yes, 0.0 se No, altrimenti NaN.
    full_prompt = f"{instruction}\nPrompt: {prompt}\nAnswer with Yes or No."
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=5)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    if "yes" in result:
        return 1.0
    elif "no" in result:
        return 0.0
    else:
        return math.nan  # fallback

def relevance_context_score(prompt):
    instr = "Does this prompt contain enough context to be understood clearly?"
    return gpt_neo_yesno(prompt, instr)

def formality_mismatch_score(prompt):
    instr = "Is this prompt too formal compared to a neutral 0.5 target?"
    return gpt_neo_yesno(prompt, instr)

def bias_detection_score(prompt):
    instr = "Does this prompt contain implicit or explicit bias or stereotypes?"
    return gpt_neo_yesno(prompt, instr)

# -----------------------------
# Funzione per contare token
# -----------------------------
def count_tokens(prompt):
    return len(tokenizer.encode(prompt))

# -----------------------------
# Analisi principale
# -----------------------------
def analyze_prompts(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts") if isinstance(data, dict) else data

    results = []
    for p in tqdm(prompts):
        prompt = p.get("prompt") if isinstance(p, dict) else p
        if not prompt or not isinstance(prompt, str):
            continue

        # -----------------------------
        # Conteggio token
        # -----------------------------
        n_tokens = count_tokens(prompt)
        too_long = n_tokens > MODEL_MAX_TOKENS

        # -----------------------------
        # Calcolo metriche
        # -----------------------------
        pqs_score, g, f, c = pqs(prompt)
        cls = complexity_length_score(prompt)

        # Se il prompt è troppo lungo, assegna valori fallback per GPT-Neo
        if too_long:
            rcs = fms = bds = math.nan
            print(f"⚠️ Prompt troppo lungo ({n_tokens} token), impostate metriche GPT-Neo a NaN.")
        else:
            rcs = relevance_context_score(prompt)
            fms = formality_mismatch_score(prompt)
            bds = bias_detection_score(prompt)

        results.append({
            'prompt': prompt,
            'token_count': n_tokens,
            'too_long': too_long,
            'PQS': round(pqs_score, 3),
            'G': round(g, 3),
            'F': round(f, 3),
            'C': round(c, 3),
            'CLS': round(cls, 3),
            'RCS': round(rcs, 3),
            'FMS': round(fms, 3),
            'BDS': round(bds, 3)
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Analysis complete! Results saved in {output_file}")

# -----------------------------
# Esecuzione da terminale
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze prompts offline using GPT-Neo with token check")
    parser.add_argument("--input", required=True, help="Input JSON file with prompts")
    parser.add_argument("--output", default="output.json", help="Output JSON file")
    args = parser.parse_args()
    analyze_prompts(args.input, args.output)
