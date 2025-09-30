# Prompt Smells Analyzer

This project analyzes text prompts to detect **common issues** such as grammatical errors, poor formatting, low clarity, excessive complexity, formality mismatches, and implicit biases.  

The script takes a JSON file containing prompts as input and outputs a JSON file with the calculated metrics.

---

## Requisiti

- Python 3.8+
- OpenAI API Key
- Dipendenze Python:

```bash
pip install pandas textstat language-tool-python openai tqdm
```

## Configuration
1. Set your OpenAI API Key as an environment variable::
- Linux/macOS
```bash
export OPENAI_API_KEY="your_api_key"
```
- Windows PowerShell
```powershell
setx OPENAI_API_KEY "your_api_key"
```
2. Prepare the input JSON file.
```json
{
  "prompts": [
    "Write a short story about a dog.",
    "Explain quantum physics in simple terms."
  ]
}
```

## Execution
From the terminal, run:
```bash
python prompt_smell_analyzer_final.py --input input.json --output output.json
```
- **--input**: path to the JSON file containing the prompts to analyze.
- **--output**: path to the output JSON file (default: output.json).

## Output
The output JSON file will contain the following metrics for each prompt:

- PQS: Prompt Quality Score (average of grammar, formatting, and clarity)
- G: grammar score
- F: formatting score
- C: clarity score
- RCS: Relevance / Context Score
- CLS: Complexity / Length Score
- FMS: Formality Mismatch Score
- BDS: Bias Detection Score

### Example:
```json
[
  {
    "prompt": "Write a short story about a dog.",
    "PQS": 0.933,
    "G": 1.0,
    "F": 1.0,
    "C": 0.8,
    "RCS": 0.95,
    "CLS": 0.85,
    "FMS": 0.1,
    "BDS": 0.0
  }
]
```

## Notes
- The analysis of RCS, FMS, and BDS requires an internet connection and a valid OpenAI API Key.
- Scores are normalized between 0 and 1, where higher values indicate higher quality or stronger presence of bias/formality depending on the metric.
- Input prompts can be either plain strings or objects with the key "prompt".
