# code containing extraction with llm based models 
import os
import torch
from PIL import Image, ImageFile
from transformers import AutoModel, AutoTokenizer
from io import BytesIO
import requests
from tqdm import tqdm
import pandas as pd
import re


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VLLM_Extractor:
    def __init__(self, model_name, device, csv_path):

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16).eval().cuda()
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(":::: LOADED MODEL SUCCESSFULLY")

        self.csv_path = csv_path
        self.output_csv = pd.read_csv(self.csv_path)


    def inference_llm(self, url, prompt, text_postprocess_fn=None):

        
        # Retrieve the image from the URL
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        
        
            # Define the question to ask
            question = f"""{prompt}"""
            msgs = [{'role': 'user', 'content': [image, question]}]
            
            # Generate the answer using the model
            answer = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                temperature=0.0,
                do_sample=False
            )

            if text_postprocess_fn:
                # Process the answer to standardize gender format
                answer = text_postprocess_fn(answer)

            return answer 

        except Exception as e:
            print(f"Error retrieving or processing image at {url}: {e}")
            return None

    def run_inference_on_csv(self, prompt, text_postprocess_fn, col_name, batch_size=20):
        ''' 
        very general script to run vllm inference for a given prompt and store the results into a csv on a particular column

        '''


        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = None 

        for index, row in tqdm(self.output_csv.iterrows()):
            if pd.notna(row[col_name]):
                continue
            
            image_url = row['full_jpg']
            try:
                output = self.inference_llm(image_url, prompt, text_postprocess_fn)
                
                if output is not None:

                    self.output_csv.at[index, col_name] = output
                    changes_count += 1

                    if changes_count % batch_size == 0:
                        self.output_csv.to_csv(self.csv_path, index=False)
                        # print(f"Saved batch of {batch_size} changes")
                else:
                    print(f"Failed to process row {index}")

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        if changes_count > 0:
            self.output_csv.to_csv(self.csv_path, index=False)
            print(f"Image Model Inference completed. Total changes: {changes_count}")







# all post procssing functions go here
def process_gender(text):
    '''
    Custom Function for processing Gender
    '''
    # Normalize the text to lowercase for easier matching
    text = text.strip().lower()
    
    # Define patterns for Male, Female, or any specific gender
    if re.fullmatch(r'male|m', text):
        return "male"
    elif re.fullmatch(r'female|f', text):
        return "female"
    elif re.fullmatch(r'non-binary|genderqueer|transgender|other|trans', text):  # Extendable for other specified genders
        return text.lower()
    elif re.fullmatch(r'\w+', text) and text not in {"male", "female", "m", "f"}:  # Return other single words
        return text.lower()
    else:
        return "N/A"  # Default to N/A for anything not specifically matched


def process_form_title(text):
    # Extract texts within double quotes in the answer and join them with commas
    extracted_texts = re.findall(r'"(.*?)"', text)
    formatted_answer = ", ".join(extracted_texts)
    return formatted_answer