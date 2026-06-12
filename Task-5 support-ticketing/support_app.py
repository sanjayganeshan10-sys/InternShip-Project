import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize settings
load_dotenv()
st.set_page_config(page_title="Intelligent Support Hub & Ticket CRM", page_icon="🎫", layout="wide")

# Configure Key Connection with REST transport fallback for robust network handling
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY missing from .env file!")
    st.stop()
genai.configure(api_key=api_key, transport='rest')

# --- REUSABLE PERSISTENT DATABASE ENGINE ---
# We use Streamlit session state to create a mock transactional SQL/NoSQL database behavior
if "ticket_db" not in st.session_state:
    st.session_state.ticket_db = pd.DataFrame([
        {"Ticket ID": "TCK-101", "Customer Name": "Sanjay G", "Issue Description": "Database timeout during checkout integration loop.", "Intent": "Technical Bug", "Sentiment": "Frustrated", "Priority": "High", "Status": "Open"},
        {"Ticket ID": "TCK-102", "Customer Name": "John Doe", "Issue Description": "Where can I find the API usage billing history panel?", "Intent": "Billing Inquiry", "Sentiment": "Neutral", "Priority": "Low", "Status": "Resolved"},
        {"Ticket ID": "TCK-103", "Customer Name": "Alice Smith", "Issue Description": "The web host deployment fails instantly with an environment error!", "Intent": "Deployment Issue", "Sentiment": "Urgent/Angry", "Priority": "Urgent", "Status": "Open"}
    ])

if "support_chat_history" not in st.session_state:
    st.session_state.support_chat_history = []

# --- ADVANCED STRUCTURAL ANALYTICS PARSER ---
def analyze_query_with_ai(customer_name, text_query):
    """
    Executes real-time structural intent recognition, sentiment extraction,
    and automatic ticketing priority calculation.
    """
    analysis_prompt = f"""
    You are an elite customer support operations analyst. Analyze the following customer support message.
    You must output your findings strictly as a valid JSON object containing exactly four keys:
    "intent" (e.g., Technical Bug, General Inquiry, Billing Inquiry, Feature Request),
    "sentiment" (e.g., Happy, Neutral, Frustrated, Angry),
    "priority" (Low, Medium, High, Urgent),
    "response" (A polite, ultra-professional customer support response resolving the issue or informing them a ticket is being created).

    Customer Query: "{text_query}"
    
    Respond ONLY with the raw JSON object structure. No extra text, no markdown codeblock wrappers.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        ai_output = model.generate_content(analysis_prompt, request_options={"timeout": 15.0}).text.strip()
        
        # Clean any accidental code block wrappers if the model outputs them
        if ai_output.startswith("```json"):
            ai_output = ai_output.split("```json")[1].split("```")[0].strip()
        elif ai_output.startswith("```"):
            ai_output = ai_output.split("```")[1].split("```")[0].strip()
            
        return json.loads(ai_output)
    except Exception as e:
        # Robust localized analytical backup parser if network connection timeouts occur
        lower_txt = text_query.lower()
        intent = "General Inquiry"
        sentiment = "Neutral"
        priority = "Medium"
        
        if "error" in lower_txt or "fail" in lower_txt or "bug" in lower_txt:
            intent, sentiment, priority = "Technical Bug", "Frustrated", "High"
        elif "money" in lower_txt or "bill" in lower_txt or "price" in lower_txt:
            intent, sentiment, priority = "Billing Inquiry", "Neutral", "Medium"
            
        return {
            "intent": intent,
            "sentiment": sentiment,
            "priority": priority,
            "response": "Thank you for reaching out. We have logged your request into our engineering database and our triage team is investigating."
        }

# --- APPLICATION INTERFACE VIEW LAYER ---
st.title("🎫 Intelligent Support CRM & Ticket Automation Hub")
st.caption("Production-Grade Intent Recognition, Sentiment Triage, and Dynamic Agent Control Systems")

# Setup clean side-by-side management view splits
tab1, tab2 = st.tabs(["👥 Customer Portal Interface", "🛠️ Agent Dashboard Control Panel"])

# --- TAB 1: FRONT-END CUSTOMER INTAKE PORTAL ---
with tab1:
    st.header("💬 Helpdesk Communication Console")
    st.write("Submit an execution query or system issue below. Our AI triage layer will automatically classify, respond, and open active engineering tracking numbers if needed.")
    
    with st.form("customer_intake_form", clear_on_submit=True):
        c_name = st.text_input("Your Full Name:", placeholder="e.g., Sanjay G")
        c_query = st.text_area("How can we assist you today?", placeholder="Type your system error message, billing issue, or feature request here...")
        submit_btn = st.form_submit_button("Transmit Support Request")
        
        if submit_btn:
            if not c_name or not c_query:
                st.warning("Please fill out both fields before submitting your ticket.")
            else:
                with st.spinner("Analyzing operational metrics via CRM intake engine..."):
                    # Process AI analysis array
                    parsed_insights = analyze_query_with_ai(c_name, c_query)
                    
                    # Generate unique ticket ID sequence
                    new_id = f"TCK-{100 + len(st.session_state.ticket_db) + 1}"
                    
                    # Build new tracking row payload
                    new_row = {
                        "Ticket ID": new_id,
                        "Customer Name": c_name,
                        "Issue Description": c_query,
                        "Intent": parsed_insights["intent"],
                        "Sentiment": parsed_insights["sentiment"],
                        "Priority": parsed_insights["priority"],
                        "Status": "Open"
                    }
                    
                    # Persist record straight into transactional database state
                    st.session_state.ticket_db = pd.concat([st.session_state.ticket_db, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Append transaction logs to screen chat history view
                    st.session_state.support_chat_history.append((c_name, c_query, parsed_insights))
                    st.success(f"🎉 Request Tracked Successfully! Automated System Reference: **{new_id}**")

    # Display active communication conversation cards below
    if st.session_state.support_chat_history:
        st.write("### ⏱️ Your Recent Portal Submissions")
        for name, query, insights in reversed(st.session_state.support_chat_history):
            with st.container(border=True):
                st.write(f"👤 **Customer Name:** {name}")
                st.info(f"**Query Submitted:** {query}")
                
                # Render NLP metadata pill chips
                col_p1, col_p2, col_p3 = st.columns(3)
                col_p1.metric("Recognized Intent", insights["intent"])
                col_p2.metric("Extracted Sentiment", insights["sentiment"])
                col_p3.metric("System Priority Tier", insights["priority"])
                
                st.write("🤖 **AI Operations Agent Response:**")
                st.write(insights["response"])

# --- TAB 2: BACK-END AGENT TICKETING CONTROL DASHBOARD ---
with tab2:
    st.header("🛠️ Engineering Support Queue CRM Grid")
    st.write("Real-time telemetry oversight panel for sorting, filtering, and updating transactional support states.")
    
    # Render analytic summary metric ribbons
    total_tickets = len(st.session_state.ticket_db)
    open_tickets = len(st.session_state.ticket_db[st.session_state.ticket_db["Status"] == "Open"])
    resolved_tickets = len(st.session_state.ticket_db[st.session_state.ticket_db["Status"] == "Resolved"])
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Tickets Logged", total_tickets)
    m_col2.metric("Active Open Incidents", open_tickets, delta=f"{open_tickets} pending", delta_color="inverse")
    m_col3.metric("Resolved Incidents", resolved_tickets)
    
    st.write("---")
    
    # Filter Controls row
    st.write("### 🔍 Advanced Queue Sorting Engine")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        status_filter = st.selectbox("Filter Data View by Operational Status:", ["All Records", "Open", "Resolved"])
    with filter_col2:
        priority_filter = st.selectbox("Filter Data View by Priority Tier:", ["All Priorities", "Urgent", "High", "Medium", "Low"])
        
    # Build filtered slice array
    display_df = st.session_state.ticket_db.copy()
    if status_filter != "All Records":
        display_df = display_df[display_df["Status"] == status_filter]
    if priority_filter != "All Priorities":
        display_df = display_df[display_df["Priority"] == priority_filter]
        
    # Output live interactive database matrix
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.write("---")
    
    # Interactive Ticket Status Update Form Layer
    st.write("### 🔄 Ticket Incident State Triage Processor")
    st.caption("Select a system reference tracking ID below to update its operational completion marker:")
    
    with st.form("status_update_form"):
        col_up1, col_up2 = st.columns(2)
        with col_up1:
            ticket_to_update = st.selectbox("Target Ticket ID Identifier:", st.session_state.ticket_db["Ticket ID"].tolist())
        with col_up2:
            new_status_val = st.selectbox("Assign New Operational Status:", ["Open", "Resolved"])
            
        update_btn = st.form_submit_button("Commit Status Transition to Database")
        if update_btn:
            # Locate target structural match row index mapping array
            idx = st.session_state.ticket_db[st.session_state.ticket_db["Ticket ID"] == ticket_to_update].index
            if len(idx) > 0:
                st.session_state.ticket_db.at[idx[0], "Status"] = new_status_val
                st.toast(f"Database mutation completed successfully! {ticket_to_update} transitioned to {new_status_val}.", icon="💾")
                st.rerun()