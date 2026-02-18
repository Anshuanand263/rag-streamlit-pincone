from time import sleep
from pdf2image import convert_from_path

# Convert each page of the PDF into an image
pages = convert_from_path("ordinances.pdf")
import pytesseract

text = ""
for page in pages:
    text += pytesseract.image_to_string(page)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(text)

# documentation for chunking
from langchain_community.document_loaders import TextLoader

# Load text file
loader = TextLoader("output.txt",encoding="utf-8")
documents = loader.load()



from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

print(f" Created {len(docs)} chunks")


from langchain_google_genai import GoogleGenerativeAIEmbeddings
genai.configure(api_key=GEMINI_API_KEY)

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