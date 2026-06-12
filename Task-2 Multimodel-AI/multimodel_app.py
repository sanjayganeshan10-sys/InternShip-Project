import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

st.set_page_config(page_title="Multi-Modal AI Engine", page_icon="👁️", layout="centered")
st.title("👁️ Advanced Multi-Modal Reasoning Engine")
st.caption("Context-aware visual analysis and evidence-based decision platform")

load_dotenv()

@st.cache_resource
def get_ai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in your .env file!")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_ai_client()

if "multimodal_history" not in st.session_state:
    st.session_state.multimodal_history = []

st.sidebar.header("📁 Visual Input Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload an image for the AI to analyze and reason over...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    active_image = Image.open(uploaded_file)
    st.sidebar.image(active_image, caption="Active Visual Context", use_container_width=True)
else:
    active_image = None
    st.sidebar.info("No image uploaded yet. The assistant will operate in text-only mode until an image is provided.")

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.multimodal_history = []
    st.rerun()

for message in st.session_state.multimodal_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Ask a question about the image or context..."):
    
    st.session_state.multimodal_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning over multimodal contents..."):
            try:
                history_context = ""
                for msg in st.session_state.multimodal_history[:-1]:
                    history_context += f"{msg['role'].capitalize()}: {msg['content']}\n"

                system_instruction = f"""
                You are an advanced multi-modal reasoning engine. Your goal is to analyze both text inputs and image structures to provide evidence-based responses.
                
                CRITICAL CORE DIRECTIVES:
                1. Contextual Memory: You must respect past dialogue history. If the user asks a follow-up question (e.g., 'What color is it?'), refer back to previous turns to identify what 'it' is.
                2. Evidence-Based Reasoning: Do not guess or hallucinate. Base your findings on visible visual data or hard facts. Point out exactly where in the image or text your proof comes from.
                3. Ambiguity & Quality Validation: If an image is uploaded but is too blurry, dark, crop-damaged, or ambiguous to answer the user's specific request confidently, do NOT make up an answer. Instead, ask intelligent follow-up questions to clarify what the user needs.
                
                Past Conversation History context:
                {history_context}
                """
                contents = [system_instruction, f"Current User Request: {user_prompt}"]
                if active_image:
                    contents.append(active_image)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                    )
                )

                ai_response = response.text
                st.markdown(ai_response)
                st.session_state.multimodal_history.append({"role": "assistant", "content": ai_response})

            except Exception as e:
                st.error(f"Multimodal Engine Error: {e}")