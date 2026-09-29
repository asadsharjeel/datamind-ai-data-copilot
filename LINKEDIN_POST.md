# LinkedIn post (edit the [brackets] after you run it yourself)

Analyzing a new dataset takes hours. So I built an AI copilot that does it in minutes. 🧠

DataMind: upload any CSV/Excel file (plus PDFs) and:
📊 Get an automatic data-quality report (missing values, outliers, correlations)
🤖 Train and compare ML models in one click, and see what drives the prediction
💬 Ask questions in plain English and get answers, charts and the code behind them
📄 Ask questions about uploaded documents, with sources

Under the hood it's an agent: the LLM decides which tool to use, and every tool is guard-railed.

🛡️ What I focused on beyond "calling an API":
• LLM-written code is validated (AST) and run read-only on a copy of the data
• Prompt-injection and destructive-request filtering
• The bot says "I don't know" instead of inventing columns or facts
• 7 automated tests for the tools

🔧 Stack: Python, Streamlit, pandas, scikit-learn, Plotly, Claude API (tool calling), RAG

⚠️ Challenge I solved: stopping the model from running unsafe code. Blocking imports, file access and private attributes taught me how much security an AI app needs.

📈 Results: [X]% accuracy on my [N]-question test set, about $[Y] per query. (Fill in after the evaluation step.)

🔗 Live demo: [link] | Code: [GitHub link]

Next: time-series forecasting, embeddings + vector DB, and a proper evaluation dashboard.

What dataset should I test it on next?

#DataScience #GenerativeAI #RAG #MachineLearning #AgenticAI #Python #Streamlit #DataAnalytics

---
Tips: post a 60-90 second screen recording (upload -> EDA -> AutoML -> chat with a chart) as the video.
Screenshots to include: EDA tab, feature-importance chart, a chat answer with "Show the code", an architecture diagram.
Don't post the [Results] line until you have real numbers.
