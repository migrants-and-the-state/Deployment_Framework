''' 
Workflow:
1. take the fully generated csv
2. From that generate the metadata for each afile 
    Metadata are:
        id: Afilenumber
        label: same 
        page_count: filter df and do a len()
        sex: apply the old sex logic
        form_titles: get all form_titles and put it into a list  
'''

import pandas as pd 


class AfileUtils:

    def __init__(self, csv_path, output_path):
        '''
        save as csv under output_path
        '''
        self.afile_csv = pd.read_csv(csv_path)
        self.output_path = output_path


    def process_afile_data(self):
        # Initialize an empty DataFrame to store the results
        processed_data = pd.DataFrame()

        # 1. Unique IDs
        processed_data['id'] = self.afile_csv['id'].unique()

        # 2. 'forms available' column
        forms_available = self.afile_csv.groupby('id')['ms_form_title_llm_v1'].apply(
            lambda x: [form for form in x.dropna().unique()]
        )
        processed_data['forms available'] = processed_data['id'].map(forms_available)

        # 3. 'sex' column - prioritizing male/female, else None
        def determine_gender(group):
            counts = group.value_counts()
            if 'male' in counts:
                return 'male'
            if 'female' in counts:
                return 'female'
            return counts.idxmax() if not counts.empty else None

        gender_counts = self.afile_csv.groupby('id')['ms_sex_llm_v1'].apply(determine_gender)
        processed_data['sex'] = processed_data['id'].map(gender_counts)

        # 4. 'page_count' column
        page_count = self.afile_csv.groupby('id')['full_jpg'].apply(len)
        processed_data['page_count'] = processed_data['id'].map(page_count)

        processed_data.to_csv(self.output_path)
        print(f"Output successfully stored in {self.output_path}")

