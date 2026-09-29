# 🧠 DataMind: AI Data Science Copilot

Upload any CSV/Excel file (and optional PDFs) and get automatic EDA, one-click AutoML with explainability,
and a chat agent that writes safe pandas code, draws charts, trains models and answers from documents.

## Features
- One-click demo datasets, data health score (0-100), interactive Plotly charts and correlation heatmap
- Chart builder, data-cleaning tab (drop / fill / dedupe) with CSV download and reset
- AutoML with model comparison, feature importance and a what-if predictor
- Chat agent (Claude tool calling) with suggested questions, plus downloadable HTML report

## Architecture
```
Streamlit UI -> Agent loop (LLM tool calling) -> Guardrails
                 |- get_eda_report : data-quality report
                 |- run_pandas     : AST-validated, sandboxed, read-only code
                 |- make_chart     : Plotly charts
                 |- train_model    : CV model comparison + permutation importance
                 '- search_docs    : RAG (chunk -> TF-IDF -> cosine) with citations + abstention
```

## Safety design
- LLM-written code is parsed with `ast` and rejected if it has imports, `open/exec/eval`, dunder access or file writes.
- Code runs on a **copy** of the data with a whitelist of builtins.
- Prompt-injection / destructive-request filter before the LLM is called.
- Answers must come from data or documents; RAG returns nothing below a similarity threshold, so the agent says "I don't know".

## Run it
```
pip install -r requirements.txt
python test_tools.py          # 7 checks, no API key needed
streamlit run app.py          # paste your Anthropic API key in the sidebar
```
Try `sample_data/titanic_like.csv` and `sample_data/company_policy.txt`.

## Known limits (honest)
- Sandbox is AST-based, not OS-level isolation. Use a container for production.
- RAG uses TF-IDF; next step is embeddings + a vector DB (Chroma/Qdrant).
- No time-series forecasting yet. Planned: statsmodels/Prophet tool.
- Evaluation set (30-50 questions) is the next milestone.
