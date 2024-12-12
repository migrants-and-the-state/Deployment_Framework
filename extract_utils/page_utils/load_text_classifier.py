# Contains all code for text models and support utils
import pandas as pd
from tqdm import tqdm
import os 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle 
from extract_utils.page_utils.load_image_classifier import Pretrained_Image_Classifier


class TextProcessing_Utils:
    def __init__(self, csv_path, formatted_ocr_path):
        
        self.csv_path = csv_path
        self.output_csv = pd.read_csv(self.csv_path)
        self.formatted_ocr_path = formatted_ocr_path

    def load_pickle_file(self, file_path):
        ''' 
        Load up a pickle file
        '''
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def normalize_vectors(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-6)


    def init_text_image_model(self, image_model_path, device = 'cuda', model_inf_map =  {0:'form',1:'letter',2:'photograph',3:'misc'}):
        self.text_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
        self.image_model = Pretrained_Image_Classifier(model_path = image_model_path, 
                                          model_inf_map = model_inf_map,
                                          csv_path = self.csv_path,
                                          device=device)


    def load_template_embeddings(self, pkl_path="./Misc_Techniques/cert_nat_img_text.pkl"):
        ''' 
        Loads Pkl file containing template image and text embedding of certificate of naturalization
        '''
        self.cert_nat_image, self.cert_nat_text = self.load_pickle_file(pkl_path)


    def compute_cosine_similarity_scores_from_pkls(self, input, template_vector, vectors=None, model=None, mode='text_model'):
        ''' 
        Computes Cosine Similarity 
        TODO: Add more info to aid debugging
        '''
        
        if model is not None:
            if mode == "text_model":
                # assume input contains text
                embeddings = model.encode(input)
                embeddings_array = np.array(embeddings)

            elif  mode == 'image_model':
                # assume input contains urls
                output, features = model.inference(input[0])
                embeddings_array = np.array(features)
            
            specific_doc_vector_array = np.array(template_vector).reshape(1, -1)
            similarity_scores = cosine_similarity(specific_doc_vector_array, embeddings_array)

        return similarity_scores

    def verify_cert_nat(self, text_cosine_sim, image_cosine_sim):

        if np.where(image_cosine_sim + text_cosine_sim > 1.7, 1, 0)[0][0] == 1:
            return True
        else:
            return False

    def detect_cert_nat(self, text, row):
        url = row['full_jpg']

        image_cosine_sim = self.compute_cosine_similarity_scores_from_pkls([url], self.cert_nat_image, None, self.image_model, mode='image_model')
        text_cosine_sim = self.compute_cosine_similarity_scores_from_pkls([text], self.cert_nat_text, None, self.text_model, mode='text_model')

        return self.verify_cert_nat(text_cosine_sim, image_cosine_sim)

    def run_inference_on_csv(self, support_func, col_name, col_datatype='object', init_val=pd.NA, batch_size=20):
        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = init_val

        for index, row in tqdm(self.output_csv.iterrows()):
            if row[col_name] != init_val:
                continue
            
            # read the text file
            file_id = row['afile_id']
            if not os.path.exists(os.path.join(self.formatted_ocr_path,f"{file_id}.txt")) or row['ms_doctype_v1'] in ['photograph','misc']: # skip if path doesn't exist
                print(f"Skipping {file_id}")
                continue 
            
            try:
                with open(os.path.join(self.formatted_ocr_path, f"{file_id}.txt"), 'r', encoding='utf-8') as f:
                    text = f.read()
                output = support_func(text, row)
                
                if output is not None:

                    self.output_csv.at[index, col_name] = output
                    changes_count += 1

                    if changes_count % batch_size == 0:
                        self.output_csv[col_name] = self.output_csv[col_name].astype(col_datatype)
                        self.output_csv.to_csv(self.csv_path, index=False)
                        # print(f"Saved batch of {batch_size} changes")
                else:
                    print(f"Failed to process row {index}")

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        if changes_count > 0:
            self.output_csv[col_name] = self.output_csv[col_name].astype(col_datatype)
            self.output_csv.to_csv(self.csv_path, index=False)
            print(f"Text Inference completed. Total changes: {changes_count}")




def detect_g325av5(text, row):
    ''' 
    Note: This code is written on a bunch of rules assuming the OCR is accurate AND that the ocr outputs are all concatenated
    Better algorithms can be used
    '''
    
    if "G-325" in text:
        if len(text) >= 1980 and len(text) < 3300:
            if ("G-325A" in text[:400] or "G-325A" in text[-200:] \
                or "G-325 A" in text[:400] or "G-325 A" in text[-200:] \
                or "325" in text[-200:]) \
                and ("I-130" not in text[-200:] and "Form I" not in text[-200:] and "Form 1" not in text[-200:]):
                return True
            else:
                return False
        else:
            return False
    return False
