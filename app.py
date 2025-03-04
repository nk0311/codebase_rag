import streamlit as st
import os
# from github import Github
# from git import Repo
from sentence_transformers import SentenceTransformer
# from langchain_pinecone import PineconeVectorStore
# from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import time
import numpy as np
from dotenv import load_dotenv

# # Initialize Pinecone
# pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
# pinecone_index = pc.Index("codebase-rag")

# # Initialize OpenAI client
# client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=st.secrets["GROQ_API_KEY"]
# )

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("codebase-rag")

# Initialize OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Perform rag
def perform_rag(query, namespace):
   # Embed the query
   raw_query_embedding = get_huggingface_embeddings(query)

   # Find the top_matches
   top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=5, include_metadata=True, namespace=namespace)
 
   # Get the list of retrieved texts
   contexts = [item['metadata']['text'] for item in top_matches['matches']]

   # Augment the query with contexts retrieved
   augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

   # Modify the prompt below as needed to improve the response quality
   system_prompt = f"""You are a Senior Software Engineer with 20 years of experience, specializing in Typescript and Python.


   Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response.

   If there are points, format it and make sure each point starts at a new line. Let's think step by step. Verify step by step.
   """


   llm_response = client.chat.completions.create(
       model="llama-3.1-8b-instant",
       messages=[
           {"role": "system", "content": system_prompt},
           {"role": "user", "content": augmented_query}
       ]
   )


   return llm_response.choices[0].message.content


# List of embedded GitHub repos
repos = [
    "https://github.com/CoderAgent/SecureAgent",
    "https://github.com/nk0311/ai_code_editor",
    "https://github.com/nk0311/customer_churn_ml"
]

# Streamed response emulator
def response_generator(prompt, repo):
    response = perform_rag(prompt, repo)
    for word in response.split("\n"):
        # Yield each part, ensuring newlines are preserved and streaming happens
        yield word + "\n"
        time.sleep(0.05)

# Streamlit UI

# Main UI
st.title("</> Codebase RAG </>")

# Sidebar
st.sidebar.title("</> Codebase RAG </>")
st.sidebar.title("🔎 About")
st.sidebar.info(
    "Codebase RAG answers your questions on a specific codebase using RAG (Retrieval Augmented Generation)."
)


# Add selected_repo as a key to session state
if "selected_repo" not in st.session_state:
    st.session_state.selected_repo = None
# Initialize messsages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar.expander("Select Github Repo"):  
    # Initially, no repo selected
    selected_repo = st.selectbox("Select a Repository to perform RAG:", ["Select a repository"] + repos)    

    st.write(f"You have selected the repository: {selected_repo}")

    # Check if the repository selection has changed
    if selected_repo != st.session_state.selected_repo:
        # Update the session state with the new repository
        st.session_state.selected_repo = selected_repo
        # Clear chat messages
        st.session_state.messages = []
        # TODO: keep messages in session state and display them when going back to a previously selected repo

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about the codebase:"):
    # Save the user message
    st.session_state.messages.append({"role": "user", "content": prompt})  
    with st.chat_message("user"):
        st.markdown(prompt)
      

    # Get response from the backend
    # with st.spinner("Fetching response..."):
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # stream = response_generator(prompt, selected_repo)
        response = st.write_stream(response_generator(prompt, selected_repo))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})