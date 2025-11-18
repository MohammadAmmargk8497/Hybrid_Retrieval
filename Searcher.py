

def search_documents(query, model, index, embeddings, top_k=5):
    # Generate embedding for the query
    query_embedding = model.encode(query).reshape(1, -1).astype('float32')
    
    # Search for the nearest neighbors
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the corresponding filenames
    results = []
    for i in range(top_k):
        filename = embeddings[indices[0][i]][0]  # Get the filename of the nearest document
        score = distances[0][i]  # Get the distance (lower is more similar)
        results.append((filename, score))
    
    return results

def search_in_chroma(collection, query, model, top_k=5):
    # Encode the query into an embedding
    query_embedding = model.encode(query)
    
    # Perform the search in Chroma DB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    print("Search Results (Debugging):")
    print(results) 
    
    return results