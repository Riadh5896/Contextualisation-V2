import streamlit as st
from langchain.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer, util
import fpdf
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings
from PIL import Image
import time
import uuid
import setup_environment
import environment
import os


#setup_environment.setup_environment()

st.set_page_config(page_title="Contextualisation-V2",
                       page_icon=":books:")

page_bg_img = """
<style>
background-color: blue;
background-size: cover;
}
</style>
"""
# Add CSS for positioning the text input at the bottom
st.markdown(
    """
    <style>
    .bottom-input-container {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        background-color: white;
        z-index: 1000;  /* Toujours visible au-dessus des autres éléments */
        padding: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

bot_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px;">
    <div style="flex-shrink: 0; margin-right: 10px;">
        <img src="https://png.pngtree.com/png-clipart/20190924/original/pngtree-agent-icon-for-your-project-png-image_4856258.jpg"
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div style="background-color: #f1f1f1; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

user_template = '''
<div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-end;">
    <div style="flex-shrink: 0; margin-left: 10px;">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-q-characters-character-alphabet-letter-36051.png?f=webp&w=512" 
             style="max-height: 50px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div style="background-color: #007bff; color: white; padding: 10px; border-radius: 10px; max-width: 75%; word-wrap: break-word; overflow-wrap: break-word;">
        {msg}
    </div>
</div>
'''

button_style = """
<style>
    .small-button {
        display: inline-block;
        padding: 5px 10px;
        font-size: 12px;
        color: blue;
        background-color: #007bff;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-right: 5px;
    }
    .small-button:hover {
        background-color: #0056b3;
    }
</style>
"""



# Cache the model loading using @st.cache_resource
@st.cache_resource
def load_llama_model():
    """Load the Ollama Llama model."""
    return Ollama(model="llama3.2")

# Cache the embeddings model

@st.cache_resource
def load_embeddings_model():
    """Load the HuggingFace Embeddings model."""
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Function to process pre-uploaded CSV files with caching
@st.cache_data
def process_preloaded_csv_files(file_paths):
    """Process pre-uploaded CSV files, split documents, and initialize a vector database retriever."""
    split_docs = []
    for file_path in file_paths:
        loader = CSVLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512,
            chunk_overlap=256,
            disallowed_special=(),
            separators=["\n\n", "\n", " "]
        )
        split_docs.extend(splitter.split_documents(documents))

    embeddings_model = load_embeddings_model()  # Use cached embeddings model
    db = FAISS.from_documents(split_docs, embeddings_model)
    return db.as_retriever()


def calculate_similarity_score(answer: str, context_docs: list) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    context_docs = [doc.page_content for doc in context_docs]
    # Encode the answer and context documents
    answer_embedding = model.encode(answer, convert_to_tensor=True)
    context_embeddings = model.encode(context_docs, convert_to_tensor=True)

    # Calculate cosine similarities
    similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)

    # Return the maximum similarity score from the context documents
    max_score = similarities.max().item() 
    if max_score < 0.55:
        return max_score, "Je ne peux pas trouver de réponse précise dans les documents. Vous devriez vérifier sur Internet."
    return max_score


# Function to get the conversation chain
def get_conversation_chain(retriever):
    llama_model = load_llama_model()  # Use cached Ollama model

    contextualize_q_system_prompt = (
        "Given the chat history and the latest user question, "
        "provide a response that directly addresses the user's query based on the provided documents. "
        "Do not rephrase the question or ask follow-up questions. "
        "Please respond in French only."
        "Always respond in French"
        "If the awnser is not presented in the data just say that he need to check on the internet"
        "If the question is not clear ask for more clarifications "
        "If the question is only one or two words ask for more details"
        "If the user is greeting you, just greet back."
        "When appropriate, provide the answer in the form of bullet points each one in a diffrent ligne." 
        "Never respond with jokes or humor, even if explicitly asked to."
        "If asked what you do, say that you are here to help them with the Belgian civil code."
        "If a question is unrelated to the Belgian civil code, state that you are only equipped to assist with matters related to the Belgian civil code."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llama_model, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "As a personal chat assistant, provide accurate and relevant information based on the provided document in 2-6 sentences. "
        "Answer should be limited to 50 words and 2-3 sentences. Do not prompt to select answers or ask standalone questions. "
        "Always respond in French."
        "If the awnser is not presented in the data just say that he need to check on the internet"
        "If the question is not clear ask for more clarifications "
        "If the question is only one or two words ask for more details"
        "If no relevant information is found, state that the user should check online. "
        "If the question is unclear or too short, ask for more details. "
        "If the question seems too broad or unrelated to the documents, ask for clarification. "
        "If the input contains only a few words, request a more detailed query."
        "If the user is greeting you, just greet back."
        "When appropriate, provide the answer in the form of bullet points." 
        "Never respond with jokes or humor, even if explicitly asked to."
        "If asked what you do, say that you are here to help them with the Belgian civil code."
        "If a question is unrelated to the Belgian civil code, state that you are only equipped to assist with matters related to the Belgian civil code."
        "Always respond in French"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llama_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()

        max_history_length = 3  # Keep only the last 3 messages
        history = store[session_id]
        history.messages = history.messages[-max_history_length:]  # Truncate history if needed

        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    

    return conversational_rag_chain

from fpdf import FPDF
import tempfile

def generate_pdf(chat_history):
    if not chat_history or not isinstance(chat_history, list):
        raise ValueError("chat_history is empty or not in the correct format.")

    try:
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Add conversation to the PDF
        for message in chat_history:
            user_msg = message.get('user', '(Message utilisateur manquant)')
            bot_msg = message.get('bot', '(Message bot manquant)')

            # Replace unsupported characters (Optional, if needed)
            user_msg = user_msg.replace("•", "-")
            bot_msg = bot_msg.replace("•", "-")

            # Add content to the PDF
            pdf.set_text_color(0, 0, 255)  # User message color: Blue
            pdf.multi_cell(0, 10, f"Utilisateur: {user_msg}")

            pdf.set_text_color(255, 0, 0)  # Bot message color: Red
            pdf.multi_cell(0, 10, f"Bot: {bot_msg}")
            pdf.ln()  # Add space between conversations

        # Save PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_pdf_path = temp_file.name
            pdf.output(temp_pdf_path)

        # Read binary data from the file
        with open(temp_pdf_path, "rb") as pdf_file:
            pdf_output = pdf_file.read()

        return pdf_output

    except Exception as e:
        print(f"Error generating PDF: {e}")
        return None





def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="Logo_du_Ministère_de_la_Cybersécurité_et_du_Numérique.svg.png", width=350, height=150)
st.sidebar.image(my_logo)


st.sidebar.markdown(
    "**Description du projet:**"
)
st.sidebar.write(
    "Ce projet a pour but de développer un système d'IA capable de comprendre et d'appliquer le code civil belge. "
    "Pour y parvenir, nous utilisons un modèle RAG qui permet d'adapter un grand modèle de langage (LLM) au contexte juridique spécifique du code civil belge."
)
# Add the dataset link to the sidebar
st.sidebar.markdown(
    "**Dataset:**"
)
st.sidebar.write(
    "Le jeu de données utilisé pour ce projet est disponible sur Kaggle: [Belgian Statutory Article Retrieval Dataset (BSAR)](https://www.kaggle.com/datasets/thedevastator/belgian-statutory-article-retrieval-dataset-bsar)"
)

#Informe when a function is loaded 
#with st.spinner('Chargement..'):
  #      time.sleep(5)
#st.success("Terminer!")


st.title("Bienvenue chez votre agent de contextualisation :books:")

with st.popover(":mega: Note d'information"):
    st.markdown("Nous vous remercions de votre participation ! Veuillez noter que ce projet est encore en cours de développement, "
    "et de nouvelles améliorations seront disponibles prochainement.\n\n"
    "Pour une expérience optimale, assurez-vous d'utiliser le mode clair. Si votre interface est sombre, appuyez sur les "
    "trois petits points en haut à droite, puis sélectionnez Paramètres et choisissez le mode clair."
                )



# Preloaded CSV files
PRELOADED_CSV_FILES = ["data/articles.csv"]  # Replace with actual file paths
# st.sidebar.header("Preloaded CSV Files")
# st.sidebar.write(f"Processing {len(PRELOADED_CSV_FILES)} preloaded CSV files:")

session_id = st.session_state.get("session_id", str(uuid.uuid4()))

#Show the loaded dataset
#for file_name in PRELOADED_CSV_FILES:
   # st.sidebar.write(f"- `{file_name}`")

# Process preloaded CSV files and initialize retriever
retriever = process_preloaded_csv_files(PRELOADED_CSV_FILES)

# Initialize the conversation chain
conversational_chain = get_conversation_chain(retriever)
st.session_state.conversational_chain = conversational_chain

# Chat input
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.markdown('<div class="bottom-input-container">', unsafe_allow_html=True)
user_input = st.text_input(
    "Bonjour :raised_hand_with_fingers_splayed: comment puis-je vous assister ?",
    key="user_input",
    help="Veuillez entrer votre requête",
    placeholder="Veuillez entrer votre requête...",
)
st.markdown('</div>', unsafe_allow_html=True)
# Layout for the buttons in a single row (horizontal alignment)
cols = st.columns([1, 1])  # Create two equal columns for buttons

# Render "Submit" button
with cols[0]:
    if st.button("Soumettre"):
          if user_input:
            if len(user_input.split()) < 3:
                st.warning("Veuillez poser une question plus détaillée avec plus de mots.")
        
            else:
                st.session_state.session_id = session_id # Static session ID for this demo; make it dynamic if needed
                conversational_chain = st.session_state.conversational_chain
                response = conversational_chain.invoke({"input": user_input}, config={"configurable": {"session_id": session_id}})
                context_docs = response.get('context', [])
                st.session_state.chat_history.append({"user": user_input, "bot": response['answer'], "context_docs": context_docs})
          else:
            st.warning("Veuillez entrer une question.")

# Render "New Chat" button
with cols[1]:
    if st.button("Nouvelle conversation"):
        st.session_state.chat_history = []  # Clear chat history



# Display chat history
if st.session_state.chat_history:
    for index, message in enumerate(st.session_state.chat_history):
        # Render the user message using the template
        st.markdown(user_template.format(msg=message['user']), unsafe_allow_html=True)
        
        # Render the bot message using the bot template
        st.markdown(bot_template.format(msg=message['bot']), unsafe_allow_html=True)

        # Initialize session state for each message
        if f"show_docs_{index}" not in st.session_state:
            st.session_state[f"show_docs_{index}"] = False
        if f"similarity_score_{index}" not in st.session_state:
            st.session_state[f"similarity_score_{index}"] = None
        

        # Layout for the buttons in a single row (horizontal alignment)
        cols = st.columns([1, 1])  # Create two equal columns for buttons

        # Render "Show Source Docs" button
        with cols[0]:
            if st.button(f"Afficher/Masquer les documents sources", key=f"toggle_{index}"):
                # Toggle the visibility of source documents for this message
                st.session_state[f"show_docs_{index}"] = not st.session_state[f"show_docs_{index}"]

        # Render "Answer Relevancy" button
        with cols[1]:
            if st.button(f"Calculer la pertinence de la réponse", key=f"relevancy_{index}"):
                if st.session_state[f"similarity_score_{index}"] is None:
                    score = calculate_similarity_score(message['bot'], message['context_docs'])
                    st.session_state[f"similarity_score_{index}"] = score
    
        # Check if source documents should be shown
        if st.session_state[f"show_docs_{index}"]:
            with st.expander("Source Documents"):
                for doc in message.get('context_docs', []):
                    st.write(f"Source: {doc.metadata['source']}")
                    st.write(doc.page_content)

        # Display similarity score if available
        if st.session_state[f"similarity_score_{index}"] is not None:
            st.write(f"Score de similarité: {st.session_state[f'similarity_score_{index}']:.2f}")
         # Add a download button
        #st.write("Chat History Debug:", st.session_state.chat_history)
        # Add a download button
        pdf_data = generate_pdf(st.session_state.chat_history)

        if pdf_data:
            st.download_button(
                label="Télécharger la conversation en PDF",
                data=pdf_data,
                file_name="historique_de_la_conversation.pdf",
                mime="application/pdf",
                key=f"download_pdf_{index}"  # Unique key
            )
        else:
            st.error("Erreur lors de la génération du PDF. Veuillez réessayer.")
