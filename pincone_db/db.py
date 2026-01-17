from time import sleep
from langchain_community.document_loaders import PyPDFLoader

file_path = "./Dsa.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

genai.configure(api_key=GEMINI_API_KEY)

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

print(f" Created {len(docs)} chunks")





from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = os.environ["PINECONE_API_KEY"]
index = os.environ["PINECONE_INDEX"]
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
BATCH_SIZE = 32
for i in range(0, len(docs), BATCH_SIZE):
    vector_store.add_documents(docs[i:i + BATCH_SIZE])
    sleep(60)

print("Data embedded & stored in pineconedb")