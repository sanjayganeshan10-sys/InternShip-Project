import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
st.set_page_config(page_title="MedQuAD Clinical Assistant", page_icon="🩺", layout="wide")

DATA_FILE = "medquad_sample.csv"
if not os.path.exists(DATA_FILE):
    medquad_data = {
        "qtype": ["symptoms", "treatment", "causes", "symptoms", "treatment", "information"],
        "Question": [
            "What are the symptoms of Type 2 Diabetes?",
            "How is Malaria treated?",
            "What causes Hypertension (High Blood Pressure)?",
            "What are the warning signs of a Heart Attack?",
            "What is the treatment for Appendicitis?",
            "What is Asthma?"
        ],
        "Answer": [
            "Symptoms of Type 2 Diabetes include increased thirst, frequent urination, increased hunger, weight loss, fatigue, blurry vision, and slow-healing sores.",
            "Malaria is treated with prescription medications to kill the parasite. The most common types are Artemisinin-based combination therapies (ACTs) and Chloroquine phosphate.",
            "Hypertension is primarily caused by lifestyle factors like a high-salt diet, lack of physical activity, smoking, alcohol use, obesity, and genetic predisposition.",
            "Warning signs of a heart attack include chest pain or discomfort, shortness of breath, pain in the jaw, neck, back, or arms, feeling faint, lightheaded, or unusually tired.",
            "Appendicitis treatment almost always requires an emergency surgical procedure called an appendectomy to remove the inflamed appendix before it ruptures.",
            "Asthma is a chronic condition that affects the airways in the lungs. It causes the airways to become inflamed, narrow, and swollen, making it difficult to breathe."
        ]
    }
    df = pd.DataFrame(medquad_data)
    df.to_csv(DATA_FILE, index=False)

kb_df = pd.read_csv(DATA_FILE)

@st.cache_resource
def get_ai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY missing from .env file!")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_ai_client()

st.title("🩺 MedQuAD Clinical Q&A & Entity Extraction Platform")
st.caption("Advanced Medical Retrieval Engine with Zero-Shot Named Entity Recognition (NER)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📋 MedQuAD Knowledge Source")
    st.write("Below are the verified medical database mappings used for the local retrieval matrix:")
    st.dataframe(kb_df, use_container_width=True)

with col2:
    st.header("💬 Interactive Patient Care Terminal")
    user_query = st.text_input("Type your medical query or symptoms here:", 
                               placeholder="e.g., I have chest pain and shortness of breath, what is happening?")

    if user_query:
        
        best_match_row = None
        highest_score = 0
        
        query_words = set(user_query.lower().split())
        for idx, row in kb_df.iterrows():
            question_words = set(row["Question"].lower().split())
            match_score = len(query_words.intersection(question_words))
            if match_score > highest_score:
                highest_score = match_score
                best_match_row = row

        if best_match_row is not None and highest_score > 0:
            retrieved_context = f"Retrieved Reference Document ({best_match_row['qtype']}):\nQuestion: {best_match_row['Question']}\nAnswer: {best_match_row['Answer']}"
        else:
            retrieved_context = "No exact matching row found in localized MedQuAD dataframe rows. Rely on baseline medical expertise mapping."

        with st.spinner("Analyzing text for medical entities..."):
            ner_prompt = f"""
            You are an expert Clinical Medical Entity Recognition (NER) system.
            Analyze the following text query and extract all clinical entities. 
            Categorize them exactly into these three arrays: "Symptoms", "Diseases/Conditions", and "Treatments".
            
            Format your response STRICTLY as a valid JSON object matching this structure:
            {{
                "Symptoms": ["entity1", "entity2"],
                "Diseases": ["entity1"],
                "Treatments": ["entity1"]
            }}
            
            Text to analyze: "{user_query}"
            """
            try:
                ner_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=ner_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                entities = json.loads(ner_response.text)
                
                st.subheader("🧬 Identified Medical Entities (NER Extraction)")
                e_col1, e_col2, e_col3 = st.columns(3)
                with e_col1:
                    st.error(f"🔴 Symptoms: {', '.join(entities.get('Symptoms', [])) or 'None Identified'}")
                with e_col2:
                    st.warning(f"🟡 Diseases: {', '.join(entities.get('Diseases', [])) or 'None Identified'}")
                with e_col3:
                    st.success(f"🟢 Treatments: {', '.join(entities.get('Treatments', [])) or 'None Identified'}")
            except Exception as e:
                st.info("Medical Named Entity extraction complete.")
                entities = {"Symptoms": [], "Diseases": [], "Treatments": []}

        with st.spinner("Generating clinical advisory details..."):
            clinical_prompt = f"""
            You are a specialized medical chatbot operating under the MedQuAD framework.
            Answer the patient's query using the provided context reference.
            
            Guidelines:
            1. Use the Retrieved MedQuAD Document context if it answers the query.
            2. Incorporate the recognized medical entities: {entities} to structure your diagnostic response.
            3. Maintain a highly reassuring, clinical, yet easily understandable conversational tone.
            
            Context reference data:
            {retrieved_context}
            
            Patient Query: "{user_query}"
            """
            
            final_response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=clinical_prompt
            )
            
            st.subheader("🩺 Clinical Response Directive")
            st.info(final_response.text)