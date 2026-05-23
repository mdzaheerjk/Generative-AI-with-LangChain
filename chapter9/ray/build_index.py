"""Build and save FAISS index from LangChain or LangGraph documentation.

This module provides functionality to build a FAISS vector index from documentation
websites using Ray for parallel processing. It includes preprocessing, embedding,
and checkpointing capabilities for efficient index creation.

To force a rebuild: Delete the faiss_index directory.

Example:
    Basic usage:
        $ python build_langchain_index.py
    
    Custom URL:
        index = build_index("https://langchain-ai.github.io/langgraph/")
"""

import ray
import numpy as np
import pickle
import os
from typing import List, Optional
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from utils import clean_html_content

# Initialize Ray
ray.init(include_dashboard=True, dashboard_host="0.0.0.0")


@ray.remote
def preprocess_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Preprocess documents by splitting them into smaller chunks.
    
    Args:
        docs: List of documents to process.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of document chunks.
    """
    print(f"Preprocessing batch of {len(docs)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Generated {len(chunks)} chunks")
    return chunks


@ray.remote
def embed_chunks_with_progress(
    chunks: List[Document], 
    batch_id: int, 
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FAISS:
    """Embed a batch of document chunks and create a FAISS index.
    
    Args:
        chunks: List of document chunks to embed.
        batch_id: Identifier for this batch (for progress tracking).
        model_name: Name of the embedding model to use.
        
    Returns:
        FAISS index containing the embedded chunks.
    """
    print(f"[Batch {batch_id}] Starting embedding of {len(chunks)} chunks")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    result = FAISS.from_documents(chunks, embeddings)
    print(f"[Batch {batch_id}] Completed embedding")
    return result

def build_index(
    base_url: str = "https://python.langchain.com/docs/tutorials/",
    batch_size: int = 10,
    max_depth: int = 2,
    embedding_batch_size: int = 500,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_dir: str = "faiss_index",
    checkpoint_dir: str = "embedding_checkpoints"
) -> FAISS:
    """Build and save a FAISS index from documentation website.
    
    This function loads documentation from a website, preprocesses it into chunks,
    embeds the chunks using a specified model, and saves the resulting FAISS index.
    Includes checkpointing to resume from interruptions.
    
    Args:
        base_url: Base URL to scrape documentation from. Defaults to LangChain tutorials.
                 Alternative: "https://langchain-ai.github.io/langgraph/" for LangGraph docs.
        batch_size: Number of documents to process in each preprocessing batch.
        max_depth: Maximum depth to crawl from the base URL.
        embedding_batch_size: Number of chunks to embed in each parallel batch.
        model_name: HuggingFace model name for embeddings.
        index_dir: Directory to save the final FAISS index.
        checkpoint_dir: Directory to save intermediate checkpoints.
        
    Returns:
        The constructed FAISS index.
        
    Example:
        # Build index from LangChain tutorials (default)
        index = build_index()
        
        # Build index from LangGraph documentation
        index = build_index("https://langchain-ai.github.io/langgraph/")
        
        # Custom configuration
        index = build_index(
            base_url="https://python.langchain.com/docs/how_to/",
            batch_size=5,
            max_depth=1
        )
    """
    # Create directories
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check for cached chunks first
    chunks_file = os.path.join(checkpoint_dir, "chunks.pkl")
    if os.path.exists(chunks_file):
        print("Loading cached chunks...")
        with open(chunks_file, 'rb') as f:
            all_chunks = pickle.load(f)
        print(f"Loaded {len(all_chunks)} cached chunks")
    else:
        print(f"Loading documentation from {base_url}")
        loader = RecursiveUrlLoader(
            base_url,
            max_depth=max_depth,
            prevent_outside=True
        )
        docs = loader.load()
        print(f"Loaded {len(docs)} documents")
        
        # Preprocess in parallel with smaller batches
        chunks_futures = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            chunks_futures.append(preprocess_documents.remote(batch))
        
        print("Waiting for preprocessing to complete...")
        all_chunks = []
        for chunks in ray.get(chunks_futures):
            all_chunks.extend(chunks)
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Save chunks for future use
        print("Saving chunks checkpoint...")
        with open(chunks_file, 'wb') as f:
            pickle.dump(all_chunks, f)
    
    # Check if FAISS index already exists
    index_file = os.path.join(index_dir, "index.faiss")
    if os.path.exists(index_file):
        print(f"Loading existing FAISS index from '{index_dir}'...")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        index = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded existing index with {index.index.ntotal} vectors")
        return index

    print("No existing index found, proceeding with embedding...")

    # Split into embedding batches
    chunk_batches = []
    for i in range(0, len(all_chunks), embedding_batch_size):
        chunk_batches.append(all_chunks[i:i + embedding_batch_size])
    
    print(f"Starting parallel embedding with {len(chunk_batches)} batches of ~{embedding_batch_size} chunks each...")
    index_futures = [
        embed_chunks_with_progress.remote(batch, i, model_name) 
        for i, batch in enumerate(chunk_batches)
    ]
    
    # Get results with progress tracking
    indices = []
    for i, future in enumerate(index_futures):
        result = ray.get(future)
        indices.append(result)
        print(f"Completed {i+1}/{len(index_futures)} embedding batches")
    
    # Merge indices
    print("Merging indices...")
    index = indices[0]
    for idx in indices[1:]:
        index.merge_from(idx)
    
    # Save the index
    print(f"Saving index to '{index_dir}'...")
    index.save_local(index_dir)
    print(f"Index saved successfully! Contains {index.index.ntotal} vectors")
    
    return index

if __name__ == "__main__":
    """Main execution block with example usage patterns."""
    
    # Example 1: LangChain tutorials (smaller, faster - recommended for testing)
    print("Building index from LangChain tutorials...")
    index = build_index(
        base_url="https://python.langchain.com/docs/tutorials/",
        batch_size=10,
        max_depth=2
    )
    
    # Example 2: LangGraph documentation (alternative)
    # print("Building index from LangGraph documentation...")
    # index = build_index(
    #     base_url="https://langchain-ai.github.io/langgraph/",
    #     batch_size=5,
    #     max_depth=1
    # )
    
    # Example 3: LangChain how-to guides (larger dataset)
    # index = build_index(
    #     base_url="https://python.langchain.com/docs/how_to/",
    #     batch_size=10,
    #     max_depth=2
    # )
    
    # Test the index with a sample query
    print("\nTesting the index:")
    test_queries = [
        "How can I build a chatbot with LangChain?",
        "What is retrieval augmented generation?", 
        "How do I use document loaders?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = index.similarity_search(query, k=2)
        for i, doc in enumerate(results):
            # Clean the content for readable display
            clean_content = clean_html_content(doc.page_content, max_length=150)
            print(f"  Result {i + 1}:")
            print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"    Content: {clean_content}")

    print(f"\nIndex building complete! Saved to 'faiss_index' directory")
    print(f"Total vectors in index: {index.index.ntotal}")
