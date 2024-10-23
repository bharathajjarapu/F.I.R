import os
import io
import streamlit as st
from groq import Groq
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.tools import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

required_env_vars = ["GROQ_API_KEY", "TAVILY_API_KEY"]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

FIR_PROMPT = """You are an AI assistant specializing in Indian law enforcement. Your task is to generate a First Information Report (FIR) based on the incident details provided. Ensure you include only sections and acts based on the information given.

Chain of Thought:
1. Analyze the incident details thoroughly.
2. Identify the key elements of the crime or complaint.
3. Determine the sections and acts that apply to the incident.
4. Organize the information into a standard Indian FIR format using Markdown.
5. Review and refine the FIR for clarity and accuracy.

Do not fill the report with hallucinations or irrelevant information. Focus on providing a factual and concise account of the incident. IF you dont know the answer, leave it with N/A.

Use the following Markdown format for the FIR:

# FIRST INFORMATION REPORT

1. **District:** [District Name]     **P.S.:** [Police Station Name]     **Year:** [Year]     **FIR No.:** [FIR Number]

2. **Acts and Sections:**
   - [List relevant Acts and Sections]

3. **Occurrence of offence:**
   - **Date:** [Date of Incident]
   - **Time:** [Time of Incident]
   - **Information received at P.S.:**
     - **Date:** [Date of FIR]
     - **Time:** [Time of FIR]

4. **Type of Information:** [Written/Oral]

5. **Place of Occurrence:** [Address/Location of Incident]

6. **Complainant/Informant Details:**
   - **Name:** [Complainant's Name]
   - **Father's/Husband's Name:** [Father's/Husband's Name]
   - **Date/Year of Birth:** [DOB/Year]
   - **Nationality:** [Nationality]
   - **Address:** [Complainant's Address]

7. **Details of known/suspected/unknown accused:**
   [List of accused persons with available details]

8. **Reasons for delay in reporting (if any):**
   [Reasons if applicable]

9. **Particulars of properties stolen/involved:**
   [Details of stolen/involved properties if applicable]

10. **Total value of properties stolen/involved:**
    [Total value if applicable]

11. **Inquest Report / U.D. case No. (if any):**
    [Inquest Report details if applicable]

12. **First Information contents:**
    [Detailed description of the incident as reported by the complainant/informant]

13. **Action taken:** [Brief description of initial action taken]

15. **Date and time of dispatch to the court:** [Date and Time]


Keep the FIR professional, precise, and factual. Do not include any unnecessary details or speculation."""

MAIN_PROMPT = """You are an AI assistant specializing in Indian law enforcement. Your primary task is to analyze incident details and generate appropriate sections and acts for First Information Reports (FIRs).

Your task is to create a report that includes:
1. A list of appropriate sections and acts based on the incident details.
2. A brief explanation for each section and act, stating why it's applicable.
3. Any additional legal considerations or recommendations for the investigating officer.

Chain of Thought:
1. Carefully review the incident details provided.
2. Identify the key elements of the crime or complaint.
3. Determine the most relevant sections of the Indian Penal Code (IPC) and other applicable acts.
4. For each identified section or act, provide a concise explanation of its relevance to the case.
5. Consider any special circumstances or aggravating factors that might invoke additional sections.
6. Compile the final report in a clear, organized format.

Keep the report professional, precise, and easy to understand. Focus on providing accurate legal information without unnecessary elaboration.

Important: Do not provide any legal advice or predict case outcomes. Your role is to provide factual information about applicable laws and sections based on the incident details."""

QA_PROMPT = """You are an AI assistant specializing in Indian law enforcement. Your task is to provide accurate and helpful answers to queries about legal matters, particularly those related to First Information Reports (FIRs) and applicable sections of Indian law.

You have access to:
1. A legal knowledge base (FAISS index)
2. Internet search results (Tavily)

Chain of Thought:
1. Analyze the user's question carefully.
2. Identify which resources are most relevant to answer the question.
3. Search the legal knowledge base for applicable laws, sections, and precedents.
4. Use the internet search tool if additional current information is needed.
5. Synthesize the information from all sources.
6. Formulate a clear, concise, and accurate answer.

Important guidelines:
1. Provide factual information about laws and procedures, not legal advice.
2. If asked about specific sections or acts, explain their general application without interpreting for specific cases.
3. For questions about FIR filing procedures, provide general guidelines followed by Indian law enforcement.
4. If unsure about any information, clearly state that and suggest consulting with a legal professional.
5. Maintain objectivity and avoid speculation about case outcomes.

Ensure your response is precise and relevant to Indian law and police procedures. Remind users that your answers are for informational purposes only, not professional legal advice."""


class FIRAssistant:
    def __init__(self):
        try:
            self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )
            self.tavily_tool = TavilySearchResults()
            self.embeddings = FastEmbedEmbeddings()
            self.law_qa = self._setup_law_qa()
            self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            self.agents = {
                "main": self._create_agent(MAIN_PROMPT),
                "fir": self._create_agent(FIR_PROMPT),
                "qa": self._create_agent(QA_PROMPT),
            }
        except Exception as e:
            st.error(f"Error initializing FIRAssistant: {str(e)}")
            raise

    def _create_agent(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _setup_law_qa(self):
        try:
            db = FAISS.load_local(
                "faiss_index", self.embeddings, allow_dangerous_deserialization=True
            )
            return RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=db.as_retriever()
            )
        except FileNotFoundError:
            st.error(
                "FAISS index not found. Please ensure the 'faiss_index' directory exists and contains the necessary files."
            )
            raise
        except Exception as e:
            st.error(f"Error setting up law QA: {str(e)}")
            raise

    def generate_fir(self, incident_details: str) -> str:
        try:
            with st.spinner("Analyzing incident details..."):
                main_input = {
                    "input": f"Analyze the following incident details and provide appropriate sections and acts:\n{incident_details}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                legal_analysis = self.agents["main"].invoke(main_input)

            with st.spinner("Generating FIR..."):
                fir_input = {
                    "input": f"Generate a First Information Report based on:\nIncident details: {incident_details}\nLegal analysis: {legal_analysis}",
                    "chat_history": self.memory.load_memory_variables({})[
                        "chat_history"
                    ],
                }
                fir = self.agents["fir"].invoke(fir_input)

            self.memory.save_context({"input": incident_details}, {"output": fir})
            return fir
        except Exception as e:
            st.error(f"Error generating FIR: {str(e)}")
            return f"An error occurred while generating the FIR: {str(e)}"

    def chat(self, query: str) -> str:
        try:
            chat_input = {
                "input": query,
                "chat_history": self.memory.load_memory_variables({})["chat_history"],
            }

            response = self.agents["qa"].invoke(chat_input)

            self.memory.save_context({"input": query}, {"output": response})
            return response
        except Exception as e:
            st.error(f"Error in chat: {str(e)}")
            return f"An error occurred during the chat: {str(e)}"

    def transcribe_audio(self, audio_data, language):
        try:
            audio_bytes = audio_data["bytes"]
            audio_io = io.BytesIO(audio_bytes)
            audio_io.name = "audio.webm"

            transcription = self.groq_client.audio.transcriptions.create(
                file=audio_io, model="whisper-large-v3", language=language
            )
            return transcription.text
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None


def setup_page():
    st.set_page_config(
        page_title="FIR Assistant",
        page_icon=":police_car:",
        layout="wide",
        menu_items=None,
    )
    st.markdown(
        """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main .block-container {
        max-width: 100%;
        padding-top: 0.1rem;
        padding-bottom: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stRadio > div {
        flex-direction: row;
        align-items: center;
    }
    .stRadio > div > label {
        margin-right: 15px;
    }
    .stSelectbox {
        max-width: 150px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def setup_header():
    header = st.columns([5, 1, 1, 1, 1])
    header[0].write("F.I.R.A - FIR Assistant")
    fir_gen_btn = header[1].button("FIR Generation", use_container_width=True)
    legal_qa_btn = header[2].button("Legal Q&A", use_container_width=True)

    languages = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
    }
    selected_language = header[4].selectbox("", list(languages.keys()), index=0)

    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "Text"

    input_switch_btn = header[3].button(
        "Switch to " + ("Text" if st.session_state.input_mode == "Voice" else "Voice"),
        use_container_width=True,
    )

    if input_switch_btn:
        st.session_state.input_mode = (
            "Voice" if st.session_state.input_mode == "Text" else "Text"
        )

    return fir_gen_btn, legal_qa_btn, selected_language, languages


def fir_generation_page(agent, selected_language, languages):
    st.markdown(
        "<h2 style='text-align: center;'>FIR Generation</h2>", unsafe_allow_html=True
    )

    if st.session_state.input_mode == "Text":
        incident_details = st.text_area("Enter the incident details:", height=200)
    else:
        audio_data = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            key="audio_recorder",
        )
        if audio_data is not None:
            st.audio(audio_data["bytes"])
            st.write("Transcribing audio...")
            incident_details = agent.transcribe_audio(
                audio_data, languages[selected_language]
            )
            if incident_details:
                st.write("Transcription:", incident_details)
            else:
                st.error("Failed to transcribe audio. Please try again.")

    if st.button("Generate FIR"):
        if incident_details:
            fir = agent.generate_fir(incident_details)
            st.session_state.fir_generated = True
            st.session_state.fir_content = fir
            st.markdown(fir)
        else:
            st.warning(
                "Please enter or record incident details before generating the FIR."
            )


def legal_qa_page(agent):
    st.markdown(
        "<h2 style='text-align: center;'>Legal Q&A</h2>", unsafe_allow_html=True
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask your legal question here...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            response = agent.chat(query)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    setup_page()
    fir_gen_btn, legal_qa_btn, selected_language, languages = setup_header()

    if "current_page" not in st.session_state:
        st.session_state.current_page = "fir_generation"

    if fir_gen_btn:
        st.session_state.current_page = "fir_generation"
    elif legal_qa_btn:
        st.session_state.current_page = "legal_qa"

    if "agent" not in st.session_state:
        try:
            st.session_state.agent = FIRAssistant()
        except Exception as e:
            st.error(f"Failed to initialize FIRAssistant: {str(e)}")
            return

    st.session_state.setdefault("fir_generated", False)
    st.session_state.setdefault("fir_content", "")
    st.session_state.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": "Welcome to the FIR Assistant. How can I help you today?",
            }
        ],
    )

    if st.session_state.current_page == "fir_generation":
        fir_generation_page(st.session_state.agent, selected_language, languages)
    elif st.session_state.current_page == "legal_qa":
        legal_qa_page(st.session_state.agent)


if __name__ == "__main__":
    main()
