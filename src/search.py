import numpy as np
from rank_bm25 import BM25Okapi
import chromadb
import logging

def search_in_chroma(collection, query: str, top_k: int = 5):
    """
    Perform a search in ChromaDB and return top_k results.
    """
    logging.info(f"Performing ChromaDB search for query: '{query}'")
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances']
    )
    logging.info(f"ChromaDB search returned {len(results['documents'][0])} results.")
    return results

def bm25_search(bm25_model, corpus, query, top_k=5):
    """
    Performs a BM25 search on the corpus using a pre-built model.
    """
    logging.info(f"Performing BM25 search for query: '{query}'")
    tokenized_query = query.split(" ")
    doc_scores = bm25_model.get_scores(tokenized_query)
    
    # Get the top_k results
    top_n = np.argsort(doc_scores)[::-1][:top_k]
    
    results = []
    for i in top_n:
        results.append({
            'corpus_id': i,
            'score': doc_scores[i]
        })
    logging.info(f"BM25 search returned {len(results)} results.")
    return results

def reciprocal_rank_fusion(results, k=60):
    """
    Combines search results using Reciprocal Rank Fusion.
    """
    fused_scores = {}
    for doc_id, score in results:
        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0
        fused_scores[doc_id] += 1 / (k + score)
    
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return reranked_results

def hybrid_search(collection, bm25_model, corpus, metadatas, query, top_k=5):
    """
    Performs a hybrid search using BM25 and ChromaDB, combining the results.
    """
    logging.info("Performing hybrid search...")
    
    # Get results from ChromaDB
    chroma_results = search_in_chroma(collection, query, top_k=top_k)
    
    # Get results from BM25
    bm25_results = bm25_search(bm25_model, corpus, query, top_k=top_k)
    
    # Combine the results using RRF
    
    # For ChromaDB, the 'documents' list contains the results for each query.
    # Since we have one query, we take the first element.
    chroma_docs = chroma_results['documents'][0]
    
    # Create a mapping from corpus index to document metadata
    corpus_map = {i: {'source': meta['source']} for i, meta in enumerate(metadatas)}

    # Prepare results for RRF
    rrf_results = []
    
    # Add Chroma results
    for i, doc in enumerate(chroma_docs):
        # Find the corpus index of the document
        try:
            corpus_id = list(corpus).index(doc)
            rrf_results.append((corpus_id, i)) # Using rank as score
        except ValueError:
            continue

    # Add BM25 results
    for result in bm25_results:
        rrf_results.append((result['corpus_id'], result['score']))

    fused_results = reciprocal_rank_fusion(rrf_results)
    
    # Get the top_k results
    top_results = []
    for doc_id, score in fused_results[:top_k]:
        top_results.append({
            'document': corpus[doc_id],
            'metadata': corpus_map.get(doc_id, {'source': 'Unknown'}),
            'score': score
        })
        
    logging.info(f"Hybrid search returned {len(top_results)} results.")
    return top_results