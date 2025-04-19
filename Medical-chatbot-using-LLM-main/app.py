from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

load_dotenv()

# Configuration
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize components
def initialize_components():
    try:
        # Load embedding model
        embeddings = download_hugging_face_embeddings()
        
        # Connect to Pinecone
        index_name = "medicalchatbot"
        docsearch = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        
        # Initialize LLM
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0.4,
            max_tokens=500
        )
        
        # Define prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
    
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        raise

# Initialize at startup
rag_chain = initialize_components()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        msg = data.get("msg", "").strip()
        if not msg:
            return jsonify({"error": "Empty message"}), 400

        print("User input:", msg)
        response = rag_chain.invoke({"input": msg})
        print("Response generated")
        
        # Return JUST the answer string
        return response["answer"]
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return "Sorry, I encountered an error while processing your request.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)