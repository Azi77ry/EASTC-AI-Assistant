from flask import Flask, render_template, request, jsonify
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv

# ------------------ CONFIG ------------------
app = Flask(__name__)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCJsrmp1NCs3Q9NZ5E4J21irqyDSXO7THI"  # <--- Replace with your key

DOCS_FOLDER = "docs"
VECTOR_STORE_PATH = "vector_store_faiss"

# ------------------ LOAD DOCUMENTS ------------------
all_docs = []

def load_local_docs():
    print("📂 Loading local documents...")
    local_docs = []
    for file_name in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file_name)
        try:
            if file_name.endswith(".txt"):
                docs = TextLoader(file_path, encoding="utf-8").load()
                local_docs.extend(docs)
                print(f"✅ Loaded text file: {file_name}")
            elif file_name.endswith(".pdf"):
                docs = PyPDFLoader(file_path).load()
                local_docs.extend(docs)
                print(f"✅ Loaded PDF file: {file_name}")
        except Exception as e:
            print(f"❌ Error loading {file_name}: {str(e)}")
    return local_docs

def scrape_eastc_website():
    print("🌐 Scraping EASTC website...")
    urls = [
        "https://www.eastc.ac.tz",
        "https://www.eastc.ac.tz/about",
        "https://www.eastc.ac.tz/academic-programmes",
        "https://www.eastc.ac.tz/admissions",
        "https://www.eastc.ac.tz/students",
        "https://www.eastc.ac.tz/news",
        "https://www.eastc.ac.tz/contact",
        "https://www.eastc.ac.tz/index.php?r=site%2Ffee"
    ]
    web_docs = []
    for url in urls:
        try:
            docs = WebBaseLoader(url).load()
            for doc in docs:
                doc.metadata.update({
                    "type": "website",
                    "url": url,
                    "date_scraped": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                })
            web_docs.extend(docs)
            print(f"✅ Scraped: {url}")
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
    return web_docs

print("🚀 Preparing EASTC knowledge base...")
all_docs.extend(load_local_docs())
all_docs.extend(scrape_eastc_website())

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(all_docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

if os.path.exists(VECTOR_STORE_PATH):
    print("📦 Loading existing vector store...")
    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
else:
    print("💾 Creating new vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

prompt_template = """
You are EASTC-AI, a friendly and professional assistant for the Eastern Africa Statistical Training Centre.

Use this context to answer accurately:
Website Info:
{web_context}

Document Info:
{document_context}

Question:
{question}
"""
PROMPT = PromptTemplate(
    input_variables=["web_context", "document_context", "question"],
    template=prompt_template
)

def get_enhanced_context(question):
    relevant_docs = retriever.get_relevant_documents(question)
    web_docs = [d for d in relevant_docs if d.metadata.get("type") == "website"]
    doc_docs = [d for d in relevant_docs if d.metadata.get("type") != "website"]

    web_context = "\n\n".join([d.page_content for d in web_docs]) or "No website info."
    doc_context = "\n\n".join([d.page_content for d in doc_docs]) or "No document info."
    return web_context, doc_context

def enhanced_qa_chain(question):
    web_context, doc_context = get_enhanced_context(question)
    prompt = PROMPT.format(
        web_context=web_context,
        document_context=doc_context,
        question=question
    )
    response = llm.invoke(prompt)
    return response.content

# ------------------ FLASK ROUTES ------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input.strip():
        return jsonify({"response": "Please type a question."})
    
    answer = enhanced_qa_chain(user_input)
    return jsonify({"response": answer})

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(debug=True)
