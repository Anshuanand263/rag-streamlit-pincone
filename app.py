from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
import os

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.environ["PINECONE_INDEX"]

genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="RETRIEVAL_QUERY")
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash"

)
query_rewriter_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
system_instruction = """
You are a query rewriting expert. Based on the provided chat history, 
rephrase the current user question into a complete, standalone question 
that can be understood without the chat history.
Only output the rewritten question and nothing else.
"""
)
vector_store = PineconeVectorStore(embedding=embeddings, index=index)




st.set_page_config(page_title="StanBot", page_icon="🤖")
st.title("StanBot  ")


def transform_query(question, history_messages):
    
    contents = []
    for m_sg in history_messages:
        role = "user" if m_sg["role"] == "user" else "assistant"
        contents.append({
            "role": role,
            "parts": [{"text": m_sg["content"]}]
        })

    
    contents.append({
        "role": "user",
        "parts": [{"text": question}]
    })



    response = query_rewriter_model.generate_content(
        contents=contents
    )

    return response.text.strip()
# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


query = st.chat_input("Ask StanBot something...")

if query:
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking (Gemini)..."):
            try:
                rewritten_query = transform_query(query, st.session_state.messages)
                with st.expander(" Rewritten Query"):
                    st.markdown(rewritten_query)
                
                results = vector_store.similarity_search(rewritten_query,k=5)
                context = "\n\n".join([doc.page_content for doc in results])

               
                with st.expander(" Retrieved Context"):
                    st.markdown(context)

                
                rag_prompt = f"""
                Context:
                {context}

                Question:
                {rewritten_query}
                 You have to behave like a Data Structure and Algorithm Expert.
    You will be given a context of relevant information and a user question.
    Your task is to answer the user's question based ONLY on the provided context.
    If the answer is not in the context, you must say "I could not find the answer in the provided document.
    Keep your answers clear, concise, and educational.
                
                """

               
                response = model.generate_content(rag_prompt)


                answer = response.text
                st.markdown(answer)

               
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"An error occurred: {e}")