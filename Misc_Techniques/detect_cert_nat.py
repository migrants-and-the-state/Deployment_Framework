from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle 


def load_pickle_file(file_path):
    ''' 
    Load up a pickle file
    '''
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-6)


def init_text_model():
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return model

def init_image_embeddings(image_embeddings_path):
    ''' 
    Loads up Image Embeddings
    '''
    all_img_embeddings = load_pickle_file(image_embeddings_path)
    image_embeddings = list()
    for i in range(len(all_img_embeddings)):
        image_embeddings.append(all_img_embeddings[i][1])
    image_embeddings = np.array(image_embeddings)
    return image_embeddings

def init_text_embeddings(text_embeddings_path):
    ''' 
    Loads up Text Embeddings (vectorized outputs of ocr)
    '''
    text_embeddings = load_pickle_file(text_embeddings_path)
    return text_embeddings    


def load_template_embeddings(pkl_path="./Misc_techniques/cert_nat_img_text.pkl"):
    ''' 
    Loads Pkl file containing template image and text embedding of certificate of naturalization
    '''
    cert_nat_image, cert_nat_text = load_pickle_file(pkl_path)
    return cert_nat_image, cert_nat_text


def compute_cosine_similarity_scores_from_pkls(text_list, template_vector, vectors=None, model=None, mode='text_model'):
    ''' 
    Computes Cosine Similarity 
    '''
    
    if model is not None:
        if mode == "text_model":
            # assume text_list contains text
            embeddings = model.encode(text_list)
            embeddings_array = np.array(embeddings)

        elif  mode == 'image_model':
            # assume text_list contains urls
            output, features = model.inference(text_list[0])
            embeddings_array = np.array(features)
        specific_doc_vector_array = np.array(template_vector).reshape(1, -1)
        similarity_scores = cosine_similarity(specific_doc_vector_array, embeddings_array)

    elif mode == 'pkl' and type(vectors) is not None:
        # assume text list actually just indices 

        indices = text_list
        embeddings_array = vectors[indices]
        embeddings_array = normalize_vectors(embeddings_array)
        specific_doc_vector_array = np.array(template_vector).reshape(1, -1)

        similarity_scores = cosine_similarity(specific_doc_vector_array, embeddings_array)

    
    return similarity_scores

def verify_cert_nat(text_cosine_sim, image_cosine_sim):

    if np.where(image_cosine_sim + text_cosine_sim > 1.7, 1, 0)[0][0] == 1:
        return 1
    else:
        return 0 
