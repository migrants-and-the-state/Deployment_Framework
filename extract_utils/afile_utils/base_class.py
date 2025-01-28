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
            # Filter the group to include only rows where ms_doctype_v1 is 'form' or 'letter'
            filtered_group = group[group['ms_doctype_v1'].isin(['form', 'letter'])]
            
            if filtered_group.empty:
                return None

            counts = filtered_group['ms_sex_llm_v1'].value_counts()
            
            if counts.empty:
                return None
            
            max_count = counts.max()
            max_values = counts[counts == max_count].index.tolist()

            # Prioritize 'male' or 'female' if they are in max_values
            if 'male' in max_values and 'female' in max_values:
                return None
            if 'male' in max_values:
                return 'male'
            if 'female' in max_values:
                return 'female'

            # If no 'male' or 'female' prioritization, return any one of the max_values
            return max_values[0]  # Arbitrary choice

        gender_counts = self.afile_csv.groupby('id')['ms_sex_llm_v1'].apply(determine_gender)
        processed_data['sex'] = processed_data['id'].map(gender_counts)

        # 4. 'page_count' column
        page_count = self.afile_csv.groupby('id')['full_jpg'].apply(len)
        processed_data['page_count'] = processed_data['id'].map(page_count)

        processed_data.to_csv(self.output_path, index=False)
        print(f"Output successfully stored in {self.output_path}")

