# Contains all code for text models and support utils
import pandas as pd
from tqdm import tqdm
import os 
class TextProcessing_Utils:
    def __init__(self, csv_path, formatted_ocr_path):
        
        self.csv_path = csv_path
        self.output_csv = pd.read_csv(self.csv_path)
        self.formatted_ocr_path = formatted_ocr_path

    def load_text_model(self):
        pass
    

    def run_inference_on_csv(self, support_func, col_name, col_datatype='object', init_val=pd.NA, batch_size=20):
        changes_count = 0
        if col_name not in self.output_csv.columns:
            self.output_csv[col_name] = init_val

        for index, row in tqdm(self.output_csv.iterrows()):
            if row[col_name] != init_val:
                continue
            
            file_id = row['afile_id']
            if not os.path.exists(os.path.join(self.formatted_ocr_path,f"{file_id}.txt")): # skip if path doesn't exist
                print(f"Skipping {file_id}")
                continue 
            
            try:
                with open(os.path.join(self.formatted_ocr_path, f"{file_id}.txt"), 'r', encoding='utf-8') as f:
                    text = f.read().strip()
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
            print(f"Image Model Inference completed. Total changes: {changes_count}")



def detect_g325av5(text, row):
    ''' 
    Note: This code is written on a bunch of rules assuming the OCR is accurate AND that the ocr outputs are all concatenated
    Better algorithms can be used
    '''
    if row['ms_doctype_v1'] in ['form','letter']:
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