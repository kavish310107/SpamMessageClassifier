import streamlit as st
from FInalmodel import classify_SMS


EXAMPLES = {
    "Choose an example...": "",
    "Likely spam": "Congratulations! You have won a free prize. Call now to claim your reward!",
    "Likely genuine": "Hey, are we still meeting for lunch tomorrow?",
}


def load_example() -> None:
    st.session_state["message_text"] = EXAMPLES[st.session_state["selected_example"]]


st.set_page_config(
    page_title="Signal | SMS spam check",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --ink: #172722; --muted: #64746d; --paper: #f5f7f2; --line: #dce4dc; --green: #176b4b; --lime: #d9f36a; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: radial-gradient(ellipse at 92% 3%, #e6f0d7 0, transparent 31%), var(--paper); }
    .block-container { max-width: 1080px; padding-top: 2rem; padding-bottom: 4rem; }
    .brand { display:flex; align-items:center; gap:10px; font-size:14px; font-weight:700; letter-spacing:0; }
    .brand-mark { width:32px; height:32px; display:grid; place-items:center; background:var(--lime); border-radius:9px; color:var(--ink); font-size:17px; }
    .eyebrow { color:var(--green); font:500 11px 'DM Mono', monospace; text-transform:uppercase; letter-spacing:1.2px; margin:34px 0 10px; }
    .title { font-size:clamp(36px, 5vw, 58px); line-height:1.02; letter-spacing:0; font-weight:700; max-width:700px; margin:0; }
    .intro { color:var(--muted); font-size:16px; max-width:620px; line-height:1.6; margin:15px 0 27px; }
    .panel { background:rgba(255,255,255,.82); border:1px solid var(--line); border-radius:8px; padding:22px; }
    .result { border-radius:8px; padding:18px 20px; margin-top:18px; border:1px solid var(--line); background:white; }
    .result.spam { border-left:5px solid #d85d48; }
    .result.ham { border-left:5px solid var(--green); }
    .result-label { font:500 11px 'DM Mono', monospace; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
    .result-title { font-size:23px; font-weight:700; margin:5px 0 2px; }
    .small-note { color:var(--muted); font-size:13px; }
    .section-label { font:500 11px 'DM Mono', monospace; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
    div[data-testid="stMetric"] { background:white; border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
    div[data-testid="stMetricLabel"] { color:var(--muted); }
    .stTextArea textarea { border-radius:8px; background:white; font-size:15px; }
    .stButton button, .stFormSubmitButton button { border-radius:6px; font-weight:600; }
    hr { border-color:var(--line); }
    </style>
    <div class="brand"><div class="brand-mark">✉</div><span>SIGNAL / MESSAGE SCREENING</span></div>
    <div class="eyebrow">A small filter for noisy inboxes</div>
    <h1 class="title">Know what’s hiding<br>in your messages.</h1>
    <p class="intro">Paste an SMS below to check whether it looks like spam. The classifier is trained on the SMS Spam Collection using the project’s TF-IDF and Naive Bayes approach.</p>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 0.85], gap="large")
with left:
    st.markdown('<div class="section-label">Message to check</div>', unsafe_allow_html=True)
    if "message_text" not in st.session_state:
        st.session_state["message_text"] = ""
    st.selectbox(
        "Try a sample",
        list(EXAMPLES),
        key="selected_example",
        on_change=load_example,
        label_visibility="collapsed",
    )
    message = st.text_area(
        "SMS message",
        placeholder="Paste or type a message...",
        height=180,
        max_chars=5000,
        key="message_text",
        label_visibility="collapsed",
    )
    check_message = st.button("Check message", type="primary", use_container_width=True)

    if check_message:
        if not message.strip():
            st.warning("Enter a message first.")
        else:
            try:
                prediction = int(classify_SMS(message))
                category = "Spam" if prediction == 1 else "Likely genuine"
                result_class = "spam" if prediction == 1 else "ham"
                detail = "Treat links, requests for payment, and urgency with care." if prediction == 1 else "Still check unexpected links or requests before acting."
                st.markdown(
                    f'<div class="result {result_class}"><div class="result-label">Classifier result</div>'
                    f'<div class="result-title">{category}</div>'
                    f'<div class="small-note">{detail}</div></div>',
                    unsafe_allow_html=True,
                )
            except Exception as error:
                st.error(f"The project classifier could not process this message: {error}")

with right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">The model</div>', unsafe_allow_html=True)
    st.markdown("### Multinomial Naive Bayes")
    st.write("Uses the project’s saved classifier, TF-IDF vectorizer, and `transform_text` preprocessing function.")
    st.markdown("---")
    metric_a, metric_b = st.columns(2)
    metric_a.metric("Classifier", "Project model")
    metric_b.metric("Vocabulary limit", "3,000")
    st.caption("Use the result as a signal, not a guarantee.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><div class='small-note'>Built from the project dataset and preprocessing workflow.</div>", unsafe_allow_html=True)