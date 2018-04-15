import pickle

out = pickle.load( open( "embedding_raw.pkl", "rb" ) )


def map_word_to_pretrained_embedding(embeddings_index, word_to_id):
    vocab_size = len(word_to_id)
    embedding_matrix = np.zeros((vocab_size, 50))
    
    for word, i in word_to_id.items():
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
    return embedding_matrix

X_embeddings = map_word_to_pretrained_embedding(out, input_lang.word2index)
