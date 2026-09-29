import pandas as pd
import streamlit as st

from agent import ask
from tools.eda import quick_eda
from tools.ml import train_model
from tools.rag import DocIndex, extract_text

st.set_page_config(page_title="DataMind", page_icon="🧠", layout="wide")
st.title("🧠 DataMind: AI Data Science Copilot")
st.caption("Upload data, explore it, train ML models, and chat with it. Read-only and guard-railed.")

with st.sidebar:
    api_key = st.text_input("Anthropic API key", type="password", help="Only needed for the chat tab.")
    data_file = st.file_uploader("Dataset (CSV or Excel)", type=["csv", "xlsx"])
    doc_files = st.file_uploader("Documents for Q&A (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

if data_file is None:
    st.info("Upload a dataset in the sidebar to begin (try sample_data/titanic_like.csv).")
    st.stop()

df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)

index = None
if doc_files:
    index = DocIndex()
    for f in doc_files:
        index.add(f.name, extract_text(f))

tab1, tab2, tab3 = st.tabs(["📊 Auto-EDA", "🤖 AutoML", "💬 Chat with data"])

with tab1:
    r = quick_eda(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", r["rows"]); c2.metric("Columns", r["columns"])
    c3.metric("Duplicates", r["duplicate_rows"]); c4.metric("Numeric cols", r["numeric_columns"])
    st.dataframe(df.head())
    miss = pd.Series(r["missing_percent"]).sort_values(ascending=False)
    st.subheader("Missing values (%)")
    st.bar_chart(miss[miss > 0])
    if r["high_correlations"]:
        st.subheader("Strong correlations")
        st.dataframe(pd.DataFrame(r["high_correlations"], columns=["col A", "col B", "|corr|"]))
    if r["outlier_counts"]:
        st.subheader("Outliers (IQR rule)")
        st.json(r["outlier_counts"])

with tab2:
    target = st.selectbox("Column to predict", df.columns)
    if st.button("Train & compare models"):
        with st.spinner("Training..."):
            res = train_model(df, target)
        if "error" in res:
            st.error(res["error"])
        else:
            st.success(f"Best: {res['best_model']}  |  test {res['metric']}: {res['test_score']}")
            st.write("Cross-validation scores:", res["cv_scores"])
            if res["dropped_columns"]:
                st.caption(f"Auto-dropped columns (too empty / ID-like): {res['dropped_columns']}")
            st.subheader("What drives the prediction?")
            st.bar_chart(pd.Series(res["feature_importance"]))

with tab3:
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.write(m["text"])
            for fig in m.get("figs", []):
                st.plotly_chart(fig, use_container_width=True)
    q = st.chat_input("Ask about your data... e.g. 'average age by sex, as a bar chart'")
    if q:
        st.session_state.chat.append({"role": "user", "text": q})
        if not (api_key or __import__("os").getenv("ANTHROPIC_API_KEY")):
            st.warning("Add your Anthropic API key in the sidebar.")
        else:
            with st.spinner("Thinking..."):
                try:
                    out = ask(q, df, index, api_key=api_key)
                    st.session_state.chat.append({"role": "assistant", "text": out["answer"], "figs": out["figures"]})
                    if out["code_trace"]:
                        with st.expander("Show the code the agent ran"):
                            st.code("\n\n".join(out["code_trace"]), language="python")
                except Exception as e:
                    st.session_state.chat.append({"role": "assistant", "text": f"Error: {e}"})
        st.rerun()
