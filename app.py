import streamlit as st
import pandas as pd
import re
from datetime import datetime

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# =========================
# APP SETUP
# =========================
st.set_page_config(page_title="Whatsapp Message Analyzer and Summarizer", layout="wide")

HF_TOKEN = "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"
MODEL_ID = "mopuriganesh/whatsapp-summarizer"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, use_auth_token=HF_TOKEN)
    return tokenizer, model


tokenizer, model = load_model()

# =========================
# PARSING & CLEANING
# =========================

regex_pattern = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s?[ap]m\s-\s([^:]+):\s(.*)$',
    re.IGNORECASE
)

emoji_re = re.compile(
    "[" 
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE
)


def parse_txt(lines):
    rows = []
    for line in lines:
        m = regex_pattern.match(line)
        if m:
            date, time_, sender, msg = m.groups()
            rows.append([date, time_, sender, msg])
        else:
            if rows:
                rows[-1][3] += " " + line.strip()
    return pd.DataFrame(rows, columns=["Date", "Time", "Sender", "Message"])


def clean_message(msg):
    if not isinstance(msg, str):
        return ""
    msg = emoji_re.sub("", msg)
    return msg.replace("<Media omitted>", "").strip()


def load_chat(file):
    fname = file.name.lower()
    if fname.endswith(".txt"):
        lines = file.read().decode("utf-8", errors="ignore").splitlines()
        df = parse_txt(lines)
    elif fname.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = [c.capitalize() for c in df.columns]
    if "Sender" not in df.columns: df["Sender"] = "Unknown"
    if "Message" not in df.columns: df["Message"] = ""
    if "Time" not in df.columns: df["Time"] = "00:00"

    df["Message"] = df["Message"].apply(clean_message)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

    df = df.dropna(subset=["Date"])
    df = df[df["Message"] != ""]

    return df


# =========================
# ANALYTICS
# =========================
def compute_stats(df):
    stats = {}
    stats["total_messages"] = len(df)

    words = re.findall(r"\b\w+\b", " ".join(df["Message"]).lower())
    stats["total_words"] = len(words)
    stats["unique_words"] = len(set(words))

    df["word_count"] = df["Message"].apply(lambda x: len(x.split()))
    longest = df.loc[df["word_count"].idxmax()]
    stats["longest"] = longest

    stats["first_date"] = df["Date"].min()
    stats["last_date"] = df["Date"].max()
    stats["chat_days"] = df["Date"].dt.date.nunique()

    stats["per_sender"] = {
        s: {"messages": len(g), "words": sum(g["word_count"])}
        for s, g in df.groupby("Sender")
    }
    return stats


def build_wordcloud(text):
    return WordCloud(width=800, height=400, background_color="white").generate(text)


# =========================
# SUMMARIZATION (CHUNK SAFE)
# =========================
def summarize_chat_for_person(df: pd.DataFrame, person_name: str) -> str:
    if df.empty:
        return "No messages to summarize for this person in the selected date range."

    # Filter only the required person's messages
    person_msgs = df[df["Sender"] == person_name]
    if person_msgs.empty:
        return "No messages from this person in the selected date range."

    # Collect messages into a single clean text
    chat_text = " ".join(person_msgs["Message"].astype(str))
    chat_text_clean = chat_text.replace("\n", " ")

    # Improved, natural summary without instruction text showing in output
    prompt = (
        f"Write a long, detailed and meaningful summary of the WhatsApp conversation "
        f"written by {person_name}. Describe:\n"
        f"- Communication purpose\n"
        f"- Plans, coordination and decisions\n"
        f"- Relationship tone (friendly / supportive / tense etc.)\n"
        f"- Key topics and important messages\n\n"
        f"Conversation:\n{chat_text_clean}"
    )

    # Tokenize and run summarization (fast!)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,  # fits model context, fast execution
    )

    output_ids = model.generate(
        **inputs,
        max_length=220,   
        min_length=100,   
        num_beams=6,
        repetition_penalty=3.0,
        no_repeat_ngram_size=3,
        length_penalty=1.8,
        early_stopping=True,
    )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()



# =========================
# UI
# =========================

st.markdown("<h1 style='text-align:center;color:#008060;'>Whatsapp Message Analyzer and Summarizer: Insights with Artificial Intelligence</h1>", unsafe_allow_html=True)

file = st.file_uploader("📂 Upload Chat File (.txt / .csv / .xlsx)")

if file:
    df = load_chat(file)

    min_d, max_d = df["Date"].min(), df["Date"].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_d, max_d])

    start, end = date_range if len(date_range) == 2 else (min_d, max_d)
    df_range = df[(df["Date"] >= pd.to_datetime(start)) & (df["Date"] <= pd.to_datetime(end))]

    stats = compute_stats(df_range)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Messages", stats["total_messages"])
    col2.metric("Words", stats["total_words"])
    col3.metric("Unique Words", stats["unique_words"])
    col4.metric("Chat Days", stats["chat_days"])

    col5, col6 = st.columns(2)
    col5.metric("First Msg", stats["first_date"].strftime("%d-%b-%Y"))
    col6.metric("Last Msg", stats["last_date"].strftime("%d-%b-%Y"))

    st.subheader(" Longest Message")
    lm = stats["longest"]
    st.info(f"{lm['Sender']} — {lm['word_count']} words")

    st.subheader("📈 Messages per Month")
    df_range["YearMonth"] = df_range["Date"].dt.to_period("M").dt.to_timestamp()
    st.line_chart(df_range.groupby("YearMonth")["Message"].count())

    st.subheader("🥧 Message Share")
    fig1, ax1 = plt.subplots()
    sender_counts = df_range["Sender"].value_counts()
    ax1.pie(sender_counts, labels=sender_counts.index, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    st.subheader("☁️ Word Cloud")
    wc = build_wordcloud(" ".join(df_range["Message"]))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.imshow(wc); ax2.axis("off")
    st.pyplot(fig2)

    st.subheader("🧠 Person-Wise Summary")
    selected = st.selectbox("Select Person", sorted(df["Sender"].unique()))
    if st.button("Generate AI Summary"):
        st.info("⏳ Generating summary... Please wait...")
        summary = summarize_chat_for_person(df_range, selected)
        st.success("Summary Generated:")
        st.write(summary)


# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align:center; font-size:16px;'>
<b>Code Executed By:</b><br>
Mopuri Ganesh Kumar Reddy - 24691F0060<br>
Matam Geetha - 24691F0062<br>
Challa Pujitha - 24691F00J1
</div>
""", unsafe_allow_html=True)
