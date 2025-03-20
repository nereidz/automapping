from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text


load_dotenv()

DB_URL = "postgresql://postgres:sid@localhost:5432/test"
engine = create_engine(DB_URL)

def fetch_pdf_path(clientid: int, layoutid: int):
    query = text("""
     select dictionary_location from data_dictionary 
        WHERE clientid = :clientid AND layoutid = :layoutid
    """)
    with engine.connect() as connection:
        result = connection.execute(query, {"clientid": clientid, "layoutid": layoutid}).fetchone()
    
    return result[0] if result else None

def get_source_target_fields(clientid: int, layoutid: int):
    source_query = text("""
        SELECT fieldnames 
        FROM source_layout
        WHERE clientid = :clientid AND layoutid = :layoutid
    """)
    
    target_query = text("""
        SELECT fieldnames 
        FROM target_layout
        WHERE clientid = :clientid AND layoutid = :layoutid
    """)
    
    with engine.connect() as connection:
        source_fields = [row[0] for row in connection.execute(source_query, {"clientid": clientid, "layoutid": layoutid}).fetchall()]
        target_fields = [row[0] for row in connection.execute(target_query, {"clientid": clientid, "layoutid": layoutid}).fetchall()]
    
    return source_fields, target_fields

def map_source_to_target(clientid: int, layoutid: int):

    pdf_path = fetch_pdf_path(clientid, layoutid)
    if not pdf_path:
        return {"error": "No PDF found for the given clientid and layoutid"}


    loader = PyPDFLoader(pdf_path)
    data = loader.load()

  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)


    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":10})
    retrieved_docs = retriever.get_relevant_documents("what is this document about")
    print(len(retrieved_docs))
    source_fields, target_fields = get_source_target_fields(clientid, layoutid)

    if not source_fields or not target_fields:
        return {"error": "No source or target fields found for the given clientid and layoutid"}

  
    system_prompt = """
    You are an expert in data transformation and SQL generation. Your job is to generate a valid SQL query to transform and load data into the target table.
    Analyze the retrieved context to determine transformation logic where applicable.There are  multiple lookup tables in the context which you have to use in join conditions.

    Context:
    {context}

    Source fields:
    {source_fields}

    Target fields:
    {target_fields}

    Return only the SQL query:
    "INSERT INTO target_table (target_field_1, target_field_2) 
        SELECT transformation_logic(source_field_1), transformation_logic(source_field_2) FROM source_table;"
    """


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

  
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0, max_tokens=5000)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)


    input_data = {
        "input": "Generate a SQL query to transfer data from source to target while applying transformation logic extracted from the context.There are  multiple lookup tables in the context which you have to use in join conditions.",
        "source_fields": ", ".join(source_fields),
        "target_fields": ", ".join(target_fields)
    }

    response = rag_chain.invoke(input_data)

    return response["answer"]

