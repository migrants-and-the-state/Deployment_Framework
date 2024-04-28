import json 
import torch
from tqdm import tqdm
from metadata_extraction.run_llm_pipeline import Extraction
from classification_models.image.load_classifier import Pretrained_Image_Classifier
from misc_techniques import detect_cert_nat
from misc_techniques.detect_G325A import detect_g325av5 
from misc_techniques import flair_extract
import pandas as pd 
import os

df = pd.read_csv("df_combined.csv")
# os.mkdir('/Users/ajay/Documents/Oncampus/TrainingFramework/metadata_outputs/') # make directory in scratch

cert_nat_image, cert_nat_text = detect_cert_nat.load_template_embeddings()
image_embeddings = detect_cert_nat.init_image_embeddings("/scratch/ag8172/data/parallel_results.pkl")
text_embeddings = detect_cert_nat.init_text_embeddings("/scratch/ag8172/data/text_embeddings_textract.pkl")

model_class = Pretrained_Image_Classifier("linear_layer.pth")
ext_class = Extraction("./metadata_extraction/configs/fewshot_v2.cfg", "./metadata_extraction/configs/examples_g325a_2.yml")

model_inf_map = {0:'form',1:'letter',2:'photograph',3:'misc'}


''' 
model_inf_map = {0:'form',1:'letter',2:'photo',3:'misc'}
for each url:
        run inference, get class
        {'document_type': model_inf_map[output]}
        if doc not photograph or misc:
                get text       
                check certnat
                if certnat:
                        {'is_cert_nat':1}
                elif g325a:
                        {'is_g325a':True}
                        invoke llm
                        {'llm_<attribute>':}

                run generalextraction with text
                {<attribute>}
                
'''
error_log_path = "/scratch/ag8172/error_log.txt"
error_noocr_path = "/scratch/ag8172/no_ocr_log.txt"
for i in tqdm(df.index.tolist()):
        try:
                document_wise_result = dict()
                url = df.iloc[i]['url']
                filename = str(i) + "_" + "_".join(df.iloc[i]['url'].split('/')[5].split('_')[1:]) + ".json"
                filename = os.path.join("/scratch/ag8172/metadata_outputs/", filename)
                output, features = model_class.inference(url)
                document_wise_result['document_type'] = model_inf_map[torch.argmax(output).item()]

                if document_wise_result['document_type'] not in ['photograph','misc']:
                        # get text
                        # text was concatenated strings obtained from aws textract
                        if not isinstance(text,float):
                                text = df.iloc[i]['Detected Text']
                                
                                image_cosine_sim = detect_cert_nat.compute_cosine_similarity_scores_from_pkls([i], cert_nat_image, image_embeddings, mode='pkl')
                                text_cosine_sim = detect_cert_nat.compute_cosine_similarity_scores_from_pkls([i], cert_nat_text, text_embeddings, mode='pkl')
                                is_cert_nat = detect_cert_nat.verify_cert_nat(text_cosine_sim, image_cosine_sim)
                                if is_cert_nat:
                                        document_wise_result['is_cert_naturalization'] = True
                                else:
                                        document_wise_result['is_cert_naturalization'] = False 
                                
                                is_g325a = detect_g325av5(text)
                                document_wise_result['g325a_attributes'] = dict()
                                if is_g325a:
                                        document_wise_result['is_g325a'] = True
                                        output = ext_class.extract_metadata(f"""{text}""")
                                        document_wise_result['g325a_attributes'] = dict()
                                        for k,v in output.items():
                                                document_wise_result['g325a_attributes']['LLM_'+k] = v

                                else:
                                        document_wise_result['is_g325a'] = False
                                
                                output = flair_extract.extract_country_years(text)
                                document_wise_result['countries'] = output['countries']
                                document_wise_result['years'] = output['years']
                        else:
                                with open(error_noocr_path, "a") as error_log_file:
                                        error_log_file.write(f"{i}, OCR Content Absent\n")
                with open(filename, "w") as json_file:
                        json.dump(document_wise_result, json_file)
        except Exception as e:
                with open(error_log_path, "a") as error_log_file:
                        error_log_file.write(f"{i}, {str(e)}\n")
                print("Exception occurred for", i, e)

        

