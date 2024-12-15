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
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

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

    def run_inference_on_csv(self, prompt, text_postprocess_fn, col_name,col_datatype='object', init_val=None, batch_size=20):
        ''' 
        very general script to run vllm inference for a given prompt and store the results into a csv on a particular column

        '''


        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = init_val 

        for index, row in tqdm(self.output_csv.iterrows()):
            if row[col_name] != init_val or row['ms_doctype_v1'] in ['photograph','misc']: # if row value already has something or if its a photograph or misc
                continue
            
            image_url = row['full_jpg']
            try:
                output = self.inference_llm(image_url, prompt, text_postprocess_fn)
                
                if output is not None:

                    self.output_csv.at[index, col_name] = output
                    changes_count += 1

                    if changes_count % batch_size == 0:
                        self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
                        # print(f"Saved batch of {batch_size} changes")
                else:
                    print(f"Failed to process row {index}")

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        if changes_count > 0:
            self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
            print(f"Image Model Inference completed. Total changes: {changes_count}")

# To be merged with vllm extractor in the future
class Qwen_Extractor:

    def __init__(self, csv_path,formatted_ocr_path):

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map='auto'
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", max_pixels = 1280*25*25)

        print(":::: LOADED MODEL SUCCESSFULLY")

        self.csv_path = csv_path
        self.formatted_ocr_path = formatted_ocr_path
        self.output_csv = pd.read_csv(self.csv_path)


    def inference_llm_certnat_complexion(self, url, prompt, text_postprocess_fn=None):
        """ 
        Extract Complexion with an LLM.
        Instead can also obtain it from the OCR straight.
        """
        
        # Retrieve the image from the URL
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise data extractor. Only return the exact requested information without any additional text or explanations. DO NOT USE PHOTOGRAPHS for anything racial. Expand any US State abbreviations."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://dctn4zjpwgdwdiiy5odjv7o2se0bqgjb.lambda-url.us-east-1.on.aws/iiif/3/og-2023-kc-nara_A10517306_0000/full/max/0/default.jpg",  # Replace with template image path
                        },
                        {"type": "text", "text": "Extract ONLY the complexion from the image text. Do not confuse it with color of eyes. Return N.A if not present."},                        
                    ],
                },
                {
                    "role": "assistant",
                    "content": "medium"  # Example output based on the template image
                },
                # Main example: Input for the primary task
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{url}",
                        },
                        {"type": "text", "text": "Extract ONLY the complexion from the image text. Do not confuse it with color of eyes. Return N.A if not present."},
                    ],
                }
                

            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
    
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=256,top_p=1, top_k=1)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
                    

        except Exception as e:
            print(f"Error retrieving or processing image at {url}: {e}")
            return None


        if text_postprocess_fn:
            # Process the answer to standardize gender format
            answer = text_postprocess_fn(output_text)

        return answer 

    def run_inference_on_csv_complexion(self, prompt, text_postprocess_fn, col_name,col_datatype='object', init_val=None, batch_size=20):
        ''' 
        very general script to run vllm inference for a given prompt and store the results into a csv on a particular column

        '''


        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = init_val 

        for index, row in tqdm(self.output_csv.iterrows()):
            if row[col_name] != init_val or row['is_cert_naturalization'] is False: 
                continue
            
            image_url = row['full_jpg']
            file_id = row['afile_id']
            try:
                if os.path.exists(os.path.join(self.formatted_ocr_path, f"{file_id}.txt")):

                    with open(os.path.join(self.formatted_ocr_path, f"{file_id}.txt"), 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    text = 'complex'

                if 'complex' in text.lower(): # for complexion
                    output = self.inference_llm_certnat_complexion(image_url, prompt, text_postprocess_fn)
                else:
                    output = "N.A"
                
                if output is not None:

                    self.output_csv.at[index, col_name] = output
                    changes_count += 1

                    if changes_count % batch_size == 0:
                        self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
                        # print(f"Saved batch of {batch_size} changes")
                else:
                    print(f"Failed to process row {index}")

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        if changes_count > 0:
            self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
            print(f"Image Model Inference completed. Total changes: {changes_count}")


    def inference_llm_certnat(self, url, prompt, text_postprocess_fn=None):
        """ 
        Extract Complexion with an LLM.
        Instead can also obtain it from the OCR straight.
        """
        
        # Retrieve the image from the URL
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise data extractor. Only return the exact requested information without any additional text or explanations. DO NOT USE PHOTOGRAPHS for anything racial. Expand any US State abbreviations."
                },
                # Main example: Input for the primary task
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{url}",
                        },
                        {"type": "text", "text": f"{prompt}"},
                    ],
                }
                

            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
    
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=256,top_p=1, top_k=1)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
                    

        except Exception as e:
            print(f"Error retrieving or processing image at {url}: {e}")
            return None


        if text_postprocess_fn:
            # Process the answer to standardize format
            answer = text_postprocess_fn(answer)

        return answer 

    def run_inference_on_csv(self, prompt, text_postprocess_fn, col_name,col_datatype='object', init_val=None, batch_size=20):
        ''' 
        very general script to run vllm inference for a given prompt and store the results into a csv on a particular column

        '''


        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = init_val 

        for index, row in tqdm(self.output_csv.iterrows()):
            if row[col_name] != init_val or row['is_cert_naturalization'] is False: 
                continue
            
            image_url = row['full_jpg']
            try:
                
                output = self.inference_llm_certnat(image_url, prompt, text_postprocess_fn)
                
                if output is not None:

                    self.output_csv.at[index, col_name] = output
                    changes_count += 1

                    if changes_count % batch_size == 0:
                        self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
                        # print(f"Saved batch of {batch_size} changes")
                else:
                    print(f"Failed to process row {index}")

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        if changes_count > 0:
            self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
            print(f"Image Model Inference completed. Total changes: {changes_count}")


    def inference_llm_g325_occupation(self, url, prompt, text_postprocess_fn=None):
        """ 
        Extract occupation with an LLM.
        """
        
        # Retrieve the image from the URL
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise data extractor. Only return the exact requested information. DO NOT USE PHOTOGRAPHS for anything racial. Return N.A if the value is not present"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://dctn4zjpwgdwdiiy5odjv7o2se0bqgjb.lambda-url.us-east-1.on.aws/iiif/3/og-2023-kc-nara_A10706964_0010/full/max/0/default.jpg",  # Replace with template image path
                        },
                        {"type": "text", "text":  "Extract ALL the values from the occupation column ONLY while reading it top to bottom in the image. Return N.A if empty"}

                    ],
                },
                {
                    "role": "assistant",
                    "content": "N.A"  # Example output based on the template image
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "https://dctn4zjpwgdwdiiy5odjv7o2se0bqgjb.lambda-url.us-east-1.on.aws/iiif/3/og-2023-kc-nara_A11638538_0020/full/max/0/default.jpg",
                        },
                        {"type": "text", "text":  "Extract ALL the values from the occupation column ONLY while reading it top to bottom in the image. Return N.A if empty"}

                    ],
                },
                {
                    "role": "assistant",
                    "content": "Mechanical Eng, Retired"  # Example output based on the template image
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{url}",
                        },
                        {"type": "text", "text":  "Extract ALL the values from the occupation column ONLY while reading it top to bottom in the image. Return N.A if empty"}
                    ],
                }
                

            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
    
            # Inference: Generation of the output
            generated_ids = self.model.generate(**inputs, max_new_tokens=256,top_p=1, top_k=1)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
                    

        except Exception as e:
            print(f"Error retrieving or processing image at {url}: {e}")
            return None


        if text_postprocess_fn:
            # Process the answer to standardize gender format
            answer = text_postprocess_fn(output_text)

        return answer 
    
        

    def run_inference_on_csv_g325a(self, prompt, text_postprocess_fn, col_name,col_datatype='object', init_val=None, batch_size=20):
        ''' 
        very general script to run vllm inference for a given prompt and store the results into a csv on a particular column

        '''


        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = init_val 

        for index, row in tqdm(self.output_csv.iterrows()):
            if row[col_name] != init_val or row['is_cert_naturalization'] is False: 
                continue
            
            image_url = row['full_jpg']
            file_id = row['afile_id']
            try:

                if os.path.exists(os.path.join(self.formatted_ocr_path, f"{file_id}.txt")):
                    with open(os.path.join(self.formatted_ocr_path, f"{file_id}.txt"), 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    text = ''
                
                output = self.inference_llm_g325_occupation(image_url, prompt, text_postprocess_fn)
                if 'mechanical eng' not in text.lower() and 'mechanical eng' in output:
                    output = ['n.a']
                if output is not None:

                    self.output_csv.at[index, col_name] = output
                    changes_count += 1

                    if changes_count % batch_size == 0:
                        self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
                        # print(f"Saved batch of {batch_size} changes")
                else:
                    print(f"Failed to process row {index}")

            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue

        if changes_count > 0:
            self.output_csv.to_csv(self.csv_path, index=False).astype(col_datatype)
            print(f"Image Model Inference completed. Total changes: {changes_count}")


# all post procssing functions go here
def process_occupation(text):
    text = text.lower().split(',')
    if len(text) == 2 and text[-1] == 'retired':
        text = text[:1]
    text = list(set(text))

    return text

def process_certnat_complexion(text):
    return "N.A" if text.lower() not in ["fair", "medium", "dark"] else text.lower()

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