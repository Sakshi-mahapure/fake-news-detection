import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
port_stem = PorterStemmer()
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# ----- CSS Styling -----
st.markdown("""
<style>
body {
    background-color: #f0f2f6;
}
h1.title {
    background: linear-gradient(to right, #1abc9c, #16a085);
    -webkit-background-clip: text;
    color: transparent;
    text-align: center;
    font-size: 3rem;
    font-family: 'Segoe UI', sans-serif;
}
p.subheader {
    text-align: center;
    font-size: 1.2rem;
    color: #2c3e50;
}
textarea {
    font-size: 16px !important;
    border-radius: 10px;
}
.stButton > button {
    background-color: #1abc9c;
    color: white;
    font-size: 18px;
    border-radius: 8px;
    padding: 12px 20px;
    width: 100%;
    margin-top: 10px;
}
.stButton > button:hover {
    background-color: #16a085;
    color: white;
}
.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    margin-top: 50px;
    font-size: 0.9rem;
    color: #777;
}
</style>
""", unsafe_allow_html=True)

# ----- Text Preprocessing -----
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    return ' '.join(con)

# ----- Prediction -----
def fake_news(news):
    processed_news = stemming(news)
    vector_input = vector_form.transform([processed_news])
    prediction = load_model.predict(vector_input)
    return prediction[0], processed_news

# ----- Main App -----
def main():
    # Title
    st.markdown('<h1 class="title">📰 Fake News Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Paste a news article below and find out if it\'s real or fake</p>', unsafe_allow_html=True)


    # Input
    news_input = st.text_area("📝 Enter News Content:", height=200, placeholder="Paste or write your news article here...")

    if st.button("🔍 Predict"):
        if not news_input.strip():
            st.warning("⚠️ Please enter some news content first.")
        else:
            with st.spinner("Analyzing..."):
                prediction, cleaned = fake_news(news_input)

            st.markdown("### 📊 Prediction Result")
            st.markdown('<div class="result-box">', unsafe_allow_html=True)

            if prediction == 0:
                st.success("✅ This news is likely **Reliable**.")
                st.markdown("🟢 *The content appears trustworthy and fact-based.*")
            else:
                st.error("🚫 This news is likely **Unreliable**.")
                st.markdown("🔴 *The content may be misleading or fake.*")

            st.markdown('</div>', unsafe_allow_html=True)

            # Show cleaned input
            with st.expander("🔍 View Preprocessed Text"):
                st.code(cleaned)

    st.markdown('<div class="footer">AS</div>', unsafe_allow_html=True)

# ----- Run App -----
if __name__ == "__main__":
    main()
