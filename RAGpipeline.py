
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class RAGPipeline:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
    
    def load_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from PDF")
        return documents
    
    def split_text(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        return embeddings
    
    def store_in_vectordb(self, chunks, embeddings):
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        print("Created vector database")
        return self.vectorstore
    
    def setup_retriever(self, k=4):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return self.retriever
    
    def setup_llm(self):
        llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        return llm
    
    def create_prompt_template(self):
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""
        )
        return prompt
    
    def build_chain(self, llm, prompt):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain ready!")
        return self.rag_chain
    
    def query(self, question):
        if not self.rag_chain:
            raise Exception("Pipeline not set up yet!")
        
        answer = self.rag_chain.invoke(question)
        return answer
    
    def setup(self, pdf_path):
        print("Setting up RAG pipeline...")
        
        docs = self.load_pdf(pdf_path)
        chunks = self.split_text(docs)
        embeddings = self.create_embeddings()
        self.store_in_vectordb(chunks, embeddings)
        self.setup_retriever()
        llm = self.setup_llm()
        prompt = self.create_prompt_template()
        self.build_chain(llm, prompt)
        
        print("Pipeline setup complete!\n")

def main():

    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file")
        return
    
    pdf_path = input("Enter the path to your PDF file: ")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' not found")
        return

    rag = RAGPipeline(api_key)
    rag.setup(pdf_path)
    
    print("\nAsk questions about your document (type 'quit' to exit):")
    while True:
        question = input("\nYour question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question.strip():
            continue
        
        answer = rag.query(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    
    main()