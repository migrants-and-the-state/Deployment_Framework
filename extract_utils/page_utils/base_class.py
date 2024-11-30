import pandas as pd
import os
import json
import boto3
import requests
from tqdm import tqdm
import time 

class FileUtils:
    def __init__(self, filepath, output_path, url_column):
        self.csv_file = pd.read_csv(filepath)
        self.output_path = output_path
        self.url_column = url_column

    def get_basic_data(self):
        '''
        Extracts id, afile_id, page_index, and full_jpg from the specified URL column,
        and writes the results into a new or existing CSV at output_path.
        '''
        # Check if the output CSV already exists
        if os.path.exists(self.output_path):
            print(f"Output CSV already exists at {self.output_path}.")
            return

        # Prepare the extracted data
        data = []
        for url in tqdm(self.csv_file[self.url_column],desc="extracting from urls"):
            try:
                # Extract required parts from the URL
                parts = url.split("/")
                id = parts[5].split("-")[-1].split("_")[1]
                afile_id = "_".join(parts[5].split("-")[-1].split("_")[1:])
                page_index = parts[5].split("-")[-1].split("_")[-1]
                full_jpg = url

                # Append data to the list
                data.append({
                    "id": id,
                    "afile_id": afile_id,
                    "page_index": page_index,
                    "full_jpg": full_jpg
                })

            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue

        # Create a DataFrame and save it to the output CSV
        output_df = pd.DataFrame(data)
        output_df.to_csv(self.output_path, index=False)
        print(f"Extracted data has been written to {self.output_path}.")



class OCR_Utils:
    def __init__(self, raw_ocr_path, formatted_ocr_path, csv_path):
        self.raw_ocr_path = raw_ocr_path # anything straight from textract
        self.formatted_ocr_path = formatted_ocr_path # ocr post formatting
        if not os.path.exists(self.raw_ocr_path):
            os.mkdir(self.raw_ocr_path)
        if not os.path.exists(self.formatted_ocr_path):
            os.mkdir(self.formatted_ocr_path)
        self.csv_path = csv_path
        self.output_csv = pd.read_csv(csv_path)
        self.textract = boto3.client('textract', region_name='us-east-1')
        self.s3 = boto3.client('s3')
        self.bucket_name = 'mats-datadump'
        
        
    def upload_image_to_s3(self, image_url):
        # Download the image
        response = requests.get(image_url)
        image_key = image_url.split('/')[5]  # Assume the image file name is the last segment of the URL

        # Upload to S3
        self.s3.put_object(Bucket=self.bucket_name, Key=image_key, Body=response.content)
        return image_key

    def delete_image_from_s3(self, bucket_name, object_key):
        s3 = boto3.client('s3')
        s3.delete_object(Bucket=bucket_name, Key=object_key)
        # print(f"Deleted {object_key} from {bucket_name}")

    def dump_ocr(self, file_id, raw_output, formatted_output):
        # Dump raw OCR output as JSON
        json_path = f"{self.raw_ocr_path}/{file_id}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(raw_output, f, ensure_ascii=False, indent=2)
        
        # Dump formatted output as text file
        txt_path = f"{self.formatted_ocr_path}/{file_id}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        return json_path, txt_path


    def invoke_textract(self, image_key):
        response = self.textract.detect_document_text(
            Document={'S3Object': {'Bucket': self.bucket_name, 'Name': image_key}}
        )
        # response = textract.analyze_document(
        #     Document={'S3Object': {'Bucket': bucket_name, 'Name': image_key},
        #     },FeatureTypes=[
        #     'TABLES','FORMS','SIGNATURES','LAYOUT',
        # ]
        # )
        return response 


    def detect_text_only(self, response):
        # Extract text from blocks
        text_blocks = [block['Text'] for block in response['Blocks'] if block['BlockType'] == 'LINE']
        detected_text = ' '.join(text_blocks)
        return detected_text


    def detect_text_from_s3(self, image_key, text_only = False):
        # Call DetectDocumentText API
        response = self.invoke_textract(image_key = image_key)

        # Extract text and its position from blocks
        text_data = []
        detected_text = list()
        for block in response['Blocks']:
            if block['BlockType'] == 'LINE':
                text = block['Text']
                detected_text.append(text)
                if not text_only:
                    bounding_box = block.get('Geometry', {}).get('BoundingBox', {})
                    polygon = block.get('Geometry', {}).get('Polygon', [])
                    text_data.append({
                        'Text': text,
                        'BoundingBox': bounding_box,
                        'Polygon': polygon
                    })

        return ' '.join(detected_text), text_data, response

    def invoke_textract_on_url(self, image_url):
        '''
        1. gets image url
        2. uploads to s3
        3. invokes textract and saves the results locally
        4. deletes files from s3 (for storage concerns)
        '''

        image_key = self.upload_image_to_s3(image_url)
        detected_text, text_coords, response = self.detect_text_from_s3(image_key=image_key, text_only=False)
        self.delete_image_from_s3(self.bucket_name, image_key)
        return detected_text, text_coords, response


    def invoke_ocr_serial(self, url, file_id):
        '''
        Invoke textract on a url,
            Get file_id
            Save file_id.json, file_id.txt
        '''

        try:
            time.sleep(0.2)
            detected_text, text_coords, response = self.invoke_textract_on_url(url)
            raw_text_path, formatted_text_path = self.dump_ocr(file_id, response, detected_text)
            return raw_text_path, formatted_text_path
        except Exception as e:
            print(file_id, "Skipped over error",e)
            return None, None

    def call_ocr_on_csv(self, batch_size=20):
        # Check if OCR columns exist, if not create them
        if 'text_ocr' not in self.output_csv.columns:
            self.output_csv['text_ocr'] = None
        if 'raw_ocr' not in self.output_csv.columns:
            self.output_csv['raw_ocr'] = None
        
        changes_count = 0
        
        # Iterate through rows where OCR hasn't been done yet
        for index, row in tqdm(self.output_csv.iterrows()):
            # Skip if both OCR columns are already filled
            if pd.notna(row['text_ocr']) and pd.notna(row['raw_ocr']):
                continue
                
            # Get image URL from full_jpg column
            image_url = row['full_jpg']
            file_id = row['afile_id']
            
            try:
                # Run OCR and get file paths
                raw_text_path, formatted_text_path = self.invoke_ocr_serial(image_url, file_id)
                
                if raw_text_path and formatted_text_path:
                    # Update DataFrame with file paths
                    self.output_csv.at[index, 'raw_ocr'] = raw_text_path
                    self.output_csv.at[index, 'text_ocr'] = formatted_text_path
                    changes_count += 1
                    
                    # Save CSV after batch_size number of changes
                    if changes_count % batch_size == 0:
                        self.output_csv.to_csv(self.csv_path, index=False)
                        print(f"Saved batch of {batch_size} changes")
                        
                    # print(f"Processed row {index}")
                else:
                    print(f"Failed to process row {index}")
                    
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                continue
        
        # Final save of CSV if there are any remaining changes
        if changes_count > 0:
            self.output_csv.to_csv(self.csv_path, index=False)
            print(f"OCR processing completed. Total changes: {changes_count}")