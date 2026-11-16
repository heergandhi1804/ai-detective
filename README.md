# AI Detective

A knowledge-based mystery-solving agent (CSE518 Artificial Intelligence, Monsoon 2026).
It works out who killed Vikram Shah at Riverside Manor by combining:

- **CSP**: node consistency, AC-3 and backtracking search over culprit, weapon, location and time
- **Propositional logic**: a knowledge base checked by model enumeration (`ask` / `entails`)
- **Bayesian network**: posterior ranking of the remaining hypotheses from uncertain evidence
- **HMM**: filtering, smoothing and Viterbi for the time of the murder

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface. The solver code is copied unchanged from the notebook. |
| `AI_Detective_Project.ipynb` | Project notebook with the full derivation and tests |
| `requirements.txt` | Python dependencies |

## Run locally

Requires Python 3.12.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Deploy to Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ with GitHub.
2. Click **Create app** and choose to deploy from a GitHub repository.
3. Enter:
   - **Repository:** `<your-github-user>/ai-detective`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Open **Advanced settings** and select **Python 3.12**.
5. Click **Deploy**.

The app needs no secrets or environment variables.

## Notes

- All deductions follow the puzzle's encoded rules and modelling assumptions. They are not real-world facts.
- All 13 clues are encoded. The original case leaves 3 of the 192 hypotheses after CSP and logic.
- Clues 1, 3, 6 and 7 can be edited on the main page, and any clue can be unticked to leave it out of
  both the CSP and the KB. The CPTs and the HMM were written for the original story, so probabilities
  in a modified case are illustrative only.
