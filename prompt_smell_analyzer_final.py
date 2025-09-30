import os
import pandas as pd
import textstat
import json
import language_tool_python
import openai
import re
from tqdm import tqdm

# Inizializza strumenti
tool = language_tool_python.LanguageTool('en-US')

# Imposta la chiave API
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Parametri
WC_MAX = 60
FORMALITY_TARGET = 0.5  # Adatta a seconda del contesto

# Funzioni di metrica

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

def ask_openai(prompt, system_instruction):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=30
        )
        text = response.choices[0].message.content.strip()
        # Stampa per debug
        print("GPT raw output:", text)
        match = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", text)
        if match:
            val = float(match.group(0))
            # Limita sempre tra 0 e 1
            return min(1.0, max(0.0, val))
        else:
            return 0.5
    except Exception as e:
        print("OpenAI API error:", e)
        return 0.5

def relevance_context_score(prompt):
    instr = (
            "Valuta se questo prompt è chiaro e contiene abbastanza contesto per essere compreso. "
            "Rispondi SOLO con un numero tra 0 (per niente chiaro) e 1 (molto chiaro), senza testo aggiuntivo. "
            "Esempio: 0.73"
    )
    return ask_openai(prompt, instr)

def formality_mismatch_score(prompt):
    instr = (
        "Quanto è formale questo prompt? Rispondi SOLO con un numero da 0 (molto informale) "
        "a 1 (molto formale), senza testo aggiuntivo. Esempio: 0.65"
    )
    val = ask_openai(prompt, instr)
    return abs(val - FORMALITY_TARGET)

def bias_detection_score(prompt):
    instr = (
        "Questo prompt contiene bias o stereotipi impliciti o espliciti? "
        "Rispondi SOLO con un numero da 0 (nessun bias) a 1 (molto biased), senza testo aggiuntivo. "
        "Esempio: 0.2"
    )
    return ask_openai(prompt, instr)

# Analisi principale

def analyze_prompts(input_file, output_file):
    # Lettura del file JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supporta due formati: lista di prompt o dizionario con chiave "prompts"
    if isinstance(data, dict) and "prompts" in data:
        prompts = data["prompts"]
    elif isinstance(data, list):
        prompts = data
    else:
        raise ValueError("Formato JSON non valido. Deve essere una lista di prompt o un oggetto con chiave 'prompts'.")

    results = []

    for prompt in tqdm(prompts):
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        pqs_score, g, f, c = pqs(prompt)
        rcs = relevance_context_score(prompt)
        cls = complexity_length_score(prompt)
        fms = formality_mismatch_score(prompt)
        bds = bias_detection_score(prompt)

        results.append({
            'prompt': prompt,
            'PQS': round(pqs_score, 3),
            'G': round(g, 3),
            'F': round(f, 3),
            'C': round(c, 3),
            'RCS': round(rcs, 3),
            'CLS': round(cls, 3),
            'FMS': round(fms, 3),
            'BDS': round(bds, 3)
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Analisi completata. Output salvato in {output_file}")

# Esecuzione da terminale
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analisi Prompt Smells da JSON")
    parser.add_argument("--input", required=True, help="File JSON di input (lista di prompt o oggetto con chiave 'prompts')")
    parser.add_argument("--output", default="output.json", help="File JSON di output")
    args = parser.parse_args()
    analyze_prompts(args.input, args.output)
