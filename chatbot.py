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
app = Flask(__name__, static_folder='.', static_url_path='')

load_dotenv()

# Get API key from environment variable (set in Render dashboard)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

DOCS_FOLDER = "docs"
VECTOR_STORE_PATH = "vector_store_faiss"

# ------------------ LOAD DOCUMENTS ------------------
all_docs = []

def load_local_docs():
    print("📂 Loading local documents...")
    local_docs = []
    if not os.path.exists(DOCS_FOLDER):
        print("📁 Docs folder not found, creating empty folder")
        os.makedirs(DOCS_FOLDER, exist_ok=True)
        return local_docs
        
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

# Initialize knowledge base
print("🚀 Preparing EASTC knowledge base...")
all_docs.extend(load_local_docs())
all_docs.extend(scrape_eastc_website())

if all_docs:
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
else:
    print("⚠️ No documents found, chatbot will use general knowledge only")
    retriever = None

llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

prompt_template = """
You are EASTC-AI, a friendly yet professional assistant created to support students, staff, and visitors of the Eastern Africa Statistical Training Centre (EASTC).

🎯 **INFORMATION PRIORITY:**
1. **LIVE WEBSITE DATA** - Most current information
2. **STRUCTURED DOCUMENTS** - Official policies, fee structures
3. **GENERAL INSTITUTIONAL INFO** - Basic program details
4. **GENERAL KNOWLEDGE** - Only when specific info is unavailable

📚 **AVAILABLE CONTEXT:**

**💰 FEE & FINANCIAL INFORMATION:**
{document_context}

**🎓 ACADEMIC PROGRAM INFORMATION:**
{document_context}

**🌐 LIVE WEBSITE UPDATES:**
{web_context}

📊 **Source Types Available:**
- 🌐 **Live Website**: Current news, admissions, academic programmes, events
- 📄 **Local Documents**: Policies, regulations, forms, historical information
- 💡 **General Knowledge**: Educational context when specific info is missing

---
📚 **CONTEXT FROM AVAILABLE SOURCES:**

**🌐 LIVE WEBSITE INFORMATION (Most Current):**
{web_context}

**📄 LOCAL DOCUMENT INFORMATION (Official Policies):**  
{document_context}

---
❓ **USER QUESTION:**
{question}

💬 **RESPONSE GUIDELINES:**
- ✅ **If website data exists**: Prioritize and cite it as current information
- 📄 **If only local documents**: Mention they're from official records
- 🔄 **If conflicting info**: Prefer website data as more current
- ❌ **If no info**: Politely guide to check website or contact relevant office
- 🎯 **Always**: Be clear, factual, and cite your sources when possible
- **For fee questions**: Provide exact amounts, currency, and breakdown if available
- **For program questions**: Include duration, requirements, and key features
- **Cite your sources**: Mention if from fee structure, website, or general info
- **Be specific**: Use exact numbers, dates, and program names
- **If incomplete**: Guide user to official channels for missing details

**Format preferences:**
- Use bullet points for lists and deadlines
- Include dates and timelines when available
- Mention if information is from current website or archived documents
- Provide actionable next steps when appropriate
- Use bullet points for lists and fee breakdowns
- Include currency (TZS/USD) for all financial amounts
- Highlight deadlines and important dates
- Separate different types of information clearly

🤖 **Assistant Response (friendly, current, factual):**
"""

PROMPT = PromptTemplate(
    input_variables=["web_context", "document_context", "question"],
    template=prompt_template,
)

def get_enhanced_context(question):
    if not retriever:
        return "No local knowledge base available.", "No document info."
        
    relevant_docs = retriever.get_relevant_documents(question)
    web_docs = [d for d in relevant_docs if d.metadata.get("type") == "website"]
    doc_docs = [d for d in relevant_docs if d.metadata.get("type") != "website"]

    web_context = "\n\n".join([d.page_content for d in web_docs]) or "No website info."
    doc_context = "\n\n".join([d.page_content for d in doc_docs]) or "No document info."
    return web_context, doc_context

def enhanced_qa_chain(question):
    if retriever:
        web_context, doc_context = get_enhanced_context(question)
        prompt = PROMPT.format(
            web_context=web_context,
            document_context=doc_context,
            question=question
        )
    else:
        # Fallback to general response if no knowledge base
        prompt = f"""You are EASTC AI Assistant. Answer this question about Eastern Africa Statistical Training Centre: {question}
        If you don't have specific information, guide the user to check the official website at https://www.eastc.ac.tz"""
    
    response = llm.invoke(prompt)
    return response.content

# ------------------ FLASK ROUTES ------------------

@app.route("/")
def index():
    return app.send_static_file('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input.strip():
        return jsonify({"response": "Please type a question."})
    
    try:
        answer = enhanced_qa_chain(user_input)
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"response": "Sorry, I'm experiencing technical difficulties. Please try again later."})

# ------------------ RUN APP ------------------
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))