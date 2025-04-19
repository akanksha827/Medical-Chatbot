# Medical-chatbot-using-LLM
```bash
pip install -r requirements.txt
```
After installing the nesccesary requirements we need to create the folder structure for the project to be running and storing various components of files so we run the list_of_files which is been created in **template.py** 
``` bash
python template.py
```
# System Diagram
![System Diagram](research/Architecture%20Diagram/system_diagram_REU_drawio.png)

# (1) Document Loading
## Medical Book 
The **Gale Encyclopedia of Medicine (GEM)** and **Essentials of Medical Microbiology (EMM)** complement each other in providing comprehensive medical information. Together, these books cover thousands of diseases, disorders, and medical conditions, each following a structured, systemized approach to make information accessible for learners and professionals alike.

### Combined Features for Building a Chatbot:
1. **Broad Disease Coverage**: GEM provides detailed descriptions of medical conditions, while EMM focuses on microbiological aspects of diseases. Together, they can form a vast knowledge base, catering to a variety of queries.

2. **Structured Approach**: Both books use standardized formats, such as overviews, symptoms, diagnosis, and treatment, enabling consistent and reliable chatbot responses.

3. **Accessibility**: Content is written in a clear, concise manner, suitable for general users and healthcare professionals, which aligns with user-friendly chatbot interactions.

4. **Visual Aids**: Inclusion of diagrams, tables, and charts in both books enhances the user's understanding and can be leveraged for multimedia responses in the chatbot.

5. **Expert-Reviewed Content**: High-quality, evidence-based information ensures the chatbot's credibility and trustworthiness.
# (2) Splitting
## Extracting and Splitting Text from PDFs for Easy Processing
To extract text from a PDF and split it into manageable chunks for further processing, we use tools like PyPDFLoader and RecursiveCharacterTextSplitter. Here’s an explanation of how this is done step by step:

### Step 1: Loading the PDF
1) PyPDFLoader is a utility that facilitates reading and extracting text content from PDF files.
2) The loader reads the PDF document page by page, extracting the textual content while handling encoding issues that PDFs often have.
3) It essentially transforms the PDF’s raw text into a Python-friendly format, like strings or structured data.

### Step 2: Splitting Text into Chunks
After loading the text from the PDF, the extracted content is often a single block of text or split only by pages. This needs to be broken down further into smaller, more manageable chunks for processing, especially when dealing with NLP tasks or vector embeddings.

1. Recursive Splitting Logic:

- The splitter divides the text into chunks based on a predefined maximum character limit.
- It uses a hierarchical approach, first attempting to split on larger boundaries (e.g., paragraphs) and, if necessary, progressively splitting on smaller boundaries (e.g., sentences, words).

2. Preserving Context:

- By prioritizing splitting at logical boundaries (like paragraphs or sentences), it preserves the natural flow of the text. This ensures that the chunks remain coherent and contextually meaningful.

3. Configurable Parameters:

- Chunk Size: The maximum number of characters allowed in a single chunk.
- Chunk Overlap: To ensure context continuity between chunks, overlapping content (e.g., the last few sentences of one chunk appearing at the start of the next) can be configured.

4. Handling Edge Cases:

- For PDFs with special formatting or non-linear text flow (e.g., columns, tables), the splitter ensures that these structures are maintained as well as possible during splitting.

### Step 3: Resulting Text Chunks
The output of this process is a list of text chunks, where:

- Each chunk is coherent and respects logical boundaries.
- Chunk sizes are optimized for downstream processing, such as embedding generation or feeding into NLP models.
- Overlaps, if configured, ensure no loss of context between consecutive chunks.

# (3) Storage
### Step 1: Embedding Generation
- What Are Embeddings?
  - Embeddings are dense vector representations of text, where each vector encodes the semantic meaning of the corresponding text. These vectors exist in a high-dimensional space and enable computational systems to understand text contextually.

- Purpose:
  - To transform text chunks into numerical representations that capture their meaning.
  - These embeddings facilitate similarity comparisons, as semantically similar chunks will have vectors that are close to each other in the vector space.

- Process:
  - A pretrained language model (e.g., from Hugging Face) is used. These models are trained on vast amounts of text data and specialize in capturing language semantics.
  - Each chunk of text is passed through the embedding model, which outputs a fixed-size vector (e.g., 384 dimensions for some models).

- Outcome: 
  - Each chunk of text is represented as a unique embedding—a vector that encapsulates its semantic meaning.
### Step 2: Storing Embeddings in Pinecone
- What is Pinecone?
  - Pinecone is a vector database designed for managing, searching, and retrieving embeddings. It is optimized for high-dimensional vector storage and retrieval operations.

- Purpose:
  - To store the embeddings for efficient retrieval based on similarity.
  - Enable advanced operations like nearest neighbor search, which finds embeddings (and therefore text chunks) that are most similar to a given query embedding.

- Process:
  - Pinecone is initialized with an index that organizes and stores the embeddings.
  - Along with embeddings, metadata such as the original chunk of text and its position in the document are stored. This metadata is crucial for linking retrieved embeddings back to the original document or its context.
  - Once uploaded, the Pinecone database indexes these embeddings in such a way that it can perform fast similarity searches using techniques like Approximate Nearest Neighbor (ANN) search.

- Outcome: 
  - The vector database contains all the embeddings, mapped to their respective text chunks. It serves as a highly optimized storage and retrieval system.

# (4) Retrieval of Text Using RAG Chain and Its Creation
### What is RAG?
- Core Idea: 
  - Instead of relying solely on a language model’s pretrained knowledge, RAG incorporates relevant external information during the response generation process.
  - It retrieves chunks of text from a knowledge base or document database (e.g., Pinecone vector database) and combines them with the input query to generate a context-aware response.
  
- Advantages:
  - Ensures responses are grounded in factual and up-to-date data.
  - Reduces reliance on static model training by dynamically retrieving information.

  ### Steps in RAG Chain
#### 1. Query Embedding
- What Happens?:
  - When a query is received, it is converted into an embedding using the same model used during the storage phase.
  - This ensures that the query is represented in the same high-dimensional space as the stored embeddings.

#### 2. Vector Search
- What Happens?:
  - The query embedding is passed to the vector database (e.g., Pinecone), where a similarity search is performed.
  - Pinecone identifies the top N most similar embeddings (text chunks) based on cosine similarity or other distance metrics.
- Outcome:
  - A set of relevant text chunks that are most semantically similar to the query.

#### 3. Combine with Query
- What Happens?:
  - The retrieved text chunks are combined with the query as context.
  - This extended context ensures that the language model has all the necessary information to generate a meaningful and accurate response.
- Example Context:
  - Query: "What are the symptoms of diabetes?"
  - Retrieved Chunks: Relevant passages from medical documents about diabetes symptoms.
  - Combined Input: "Question: What are the symptoms of diabetes? Context: [retrieved chunk 1], [retrieved chunk 2]".

# (5) Output Generation 

### How does LLM work?
#### 1. Input Preparation:  
   - The query and the retrieved chunks of relevant text are combined into a structured format.  
   - Example format:  
      
     Question: [User Query]  
     Context: [Chunk 1], [Chunk 2], [Chunk 3]  
     
   - This ensures that the language model can understand the relationship between the query and the provided context.
#### 2. Model Encoding:  
   - The combined input is tokenized (converted into a sequence of numbers representing words or subwords) and passed to the LLaMA-2 model.  
   - LLaMA-2 encodes the input to recognize the semantic relationships between the query and the context, focusing on relevant details within the chunks.
#### 3. Attention Mechanism:  
   - LLaMA-2 uses a self-attention mechanism to weigh the importance of different parts of the input.  
   - It determines which context chunks or phrases are most relevant to answering the query.  
   - This ensures the generated response is grounded in the retrieved text rather than relying solely on the model's prior knowledge.
#### 4. Response Decoding:  
   - Using its decoder module, LLaMA-2 generates a human-like response by predicting the most probable next token (word/subword) iteratively.  
   - It ensures fluency and coherence while adhering to the information provided in the context.  
#### 5. Grounded Output Generation:  
   - The final response is heavily influenced by the retrieved context, ensuring accuracy and relevance.  
   - If the query cannot be answered based on the context, LLaMA-2 is likely to generate a response acknowledging the lack of sufficient information.

## How Everything works together? - Architecture Diagram
![Architecture Diagram](research/Architecture%20Diagram/LLm-Page-2.jpg)

## Front-End Integration
![Front-End output](research/Architecture%20Diagram/output-front-end.png)
