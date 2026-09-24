import streamlit as st
import requests

st.set_page_config(page_title="AgriGPT 🌱")

st.title("AgriGPT 🌱")
st.write("Assistant agricole intelligent")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Posez votre question")

if st.button("Envoyer") and question:
    res = requests.post(
        "http://localhost:8000/ask",
        json={"question": question}
    ).json()

    st.session_state.history.append(
        (question, res["answer"], res["sources"])
    )

for q, a, s in reversed(st.session_state.history):
    st.markdown(f"**🧑‍🌾 Question :** {q}")
    st.markdown(f"**🤖 Réponse :** {a}")
    st.markdown(f"**📚 Sources :** {', '.join(s)}")
    st.markdown("---")
