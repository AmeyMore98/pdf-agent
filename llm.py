"""LLM functionality for the PDF Agent."""

import os
from typing import List, Dict, Any
from openai import OpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-4o-mini"

# Global client - loaded once
_openai_client = None

def get_openai_client():
    """Get the OpenAI client, initializing it if necessary."""
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY environment variable is required")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]], model: str = DEFAULT_MODEL) -> str:
    """Generate an answer using OpenAI based on query and retrieved chunks."""
    client = get_openai_client()

    # Prepare context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(f"[Source {i} - {chunk['document_path']} (page {chunk['page']})]\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    # Create the user prompt - task-specific content
    prompt = f"""CONTEXT FROM DOCUMENTS:
        {context}

        QUESTION: {query}

        Please analyze the above context and provide a comprehensive answer following your instructions.
    """

    system_prompt = """You are an expert research analyst specializing in document analysis. Your core capabilities:
        ANALYSIS APPROACH:
        - Synthesize information across multiple sources to form complete insights
        - Identify patterns, connections, and relationships between different pieces of information
        - Use concrete details, numbers, examples, and direct quotes when available
        - Think step-by-step through complex questions

        RESPONSE STANDARDS:
        - Start with a direct, clear answer to the question
        - Structure responses with clear sections for complex topics
        - Be honest about information gaps and limitations

        QUALITY MARKERS:
        - Evidence-based conclusions
        - Cross-source synthesis
        - Logical reasoning chains
        - Precise citations
        - Professional analysis depth
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        return response.choices[0].message.content

    except Exception as e:
        raise Exception(f"Error generating answer: {e}")

def rag_search(search_function, query: str, limit: int = 5, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Perform Retrieval-Augmented Generation: search + generate answer."""

    print("🔍 Searching document chunks...")

    # Step 1: Retrieve relevant chunks using the provided search function
    retrieved_chunks = search_function(query, limit)

    if not retrieved_chunks:
       return {
            'query': query,
            'retrieved_chunks': [],
            'answer': "No relevant documents found to answer your question.",
            'sources': []
        }

    print(f"✅ Found {len(retrieved_chunks)} relevant chunks")
    print("🤖 Generating AI answer...")

    # Step 2: Generate answer using LLM
    answer = generate_answer(query, retrieved_chunks, model)

    # Step 3: Prepare sources list
    sources = []
    for chunk in retrieved_chunks:
        source_info = {
            'document_path': chunk['document_path'],
            'page': chunk['page'],
            'similarity': chunk['similarity'],
            'preview': chunk['preview']
        }
        if source_info not in sources:  # Avoid duplicates
            sources.append(source_info)

    return {
        'query': query,
        'retrieved_chunks': retrieved_chunks,
        'answer': answer,
        'sources': sources
    }
