import pandas as pd
import os
from tqdm import tqdm

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
