from dotenv import load_dotenv
import os
import json
import pandas as pd
import glob
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import shutil
import time

load_dotenv()

#EMBEDDINGS MODEL  
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
)
  
if os.path.exists(os.getenv("DATABASE_LOCATION")):
    shutil.rmtree(os.getenv("DATABASE_LOCATION"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"), 
)



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

#LOADING TEXT FILES INSTEAD OF JSON   
data_folder = os.getenv("DATASET_STORAGE_FOLDER") or "data/"
text_files = glob.glob(os.path.join(data_folder, "*.txt"))

file_content = []

for file_path in text_files:
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
        file_content.append({
            "url": file_path,
            "title": os.path.basename(file_path).replace(".txt", ""),
            "raw_text": text
        })

#CHUNKING, EMBEDDING AND INGESTION   
for line in file_content:
    print(f" Processing: {line['title']}")

    
    texts = text_splitter.create_documents(
        [line['raw_text']],
        metadatas=[{"source": line['url'], "title": line['title']}]
    )

    uuids = [str(uuid4()) for _ in range(len(texts))]

    
    vector_store.add_documents(documents=texts, ids=uuids)

print(" All documents have been chunked and embedded successfully!")
