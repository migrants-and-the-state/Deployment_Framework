from pathlib import Path
from typing import Optional

import typer
from wasabi import msg

from spacy_llm.util import assemble


class Extraction:
    ''' 
    Extraction Class is a wrapper around spacy-llm's code 
    '''

    def __init__(self, config_path, examples_path):
        ''' 
        Loads LLM Model with config and example
        '''
        msg.text(f"Loading config from {config_path}", show=False)
        self.nlp = assemble(
            config_path,
            overrides={}
            if examples_path is None
            else {"paths.examples": str(examples_path)},
        )

    def extract_metadata(self, text):
        ''' 
        input: Text
        Output: {entity1: [text1, text2], entity2: [text1, text2]}
        '''

        doc = self.nlp(text)
        [(ent.text, ent.label_) for ent in doc.ents]
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = set()
            entities[ent.label_].add(ent.text)
        for k,v in entities.items():
            entities[k] = list(v)
        
        return entities

ext_class = Extraction("./configs/fewshot_v2.cfg", "./configs/examples_g325a_2.yml")
output = ext_class.extract_metadata("""
FORM G-325 A REV. 4-1-67 Screened by 9/18/2023 FORM APPROVED T/C 24QUDGET BUREAU NO. 43-R436 BIOGRAPHIC UNITED STATES DEPARTMENT OF JUSTICE JAN 15 1969 INFORMATION Immigration and Naturalization Service (FAMILY NAME) (FIRST NAME) (MIDDLE NAME) MALE BIRTHDATE(MO-DAY-YR.) NATIONALITY DATE ALIEN REGISTRATION NO. WONG, BEN DONG FEMALE 1/22/1814 CHINESE (IF ANY A12 250 430 ALL OTHER NAMES USED CITY AND COUNTRY OF BIRTH SOCIAL SECURITY NO. (IF ANY) WONG, GUCK ON CANTON , CHINA 553-36-2579 FAMILY NAME FIRST NAME DATE, CITY AND COUNTRY OF BIRTH (IF KNOWN) CITY AND COUNTRY OF RESIDENCE FATHER WONG WEE DON DIED MOTHER (MAIDEN NAME) TOM GIM YEP - CANADA (RESIDENCE) was SPOUSE (IF NONE, so STATE) FAMILY NAME FIRST NAME BIRTHDATE CITY & COUNTRY OF BIRTH DATE OF MARRIAGE PLACE OF MARRIAGE (FOR WIFE, GIVE MAIDEN NAME) FONG SUEY GiN 7/28/15 CANTON, CHINA, CHINA FORMER SPOUSES (IF NONE, so STATE) FAMILY NAME (FOR WIFE. GIVE MAIDEN NAME) FIRST NAME BIRTHDATE DATE & PLACE OF MARRIAGE DATE AND PLACE OF TERMINATION OF MARRIAGE NONE APPLICANT'S RESIDENCE LAST FIVE YEARS. LIST PRESENT ADDRESS FIRST. FROM TO STREET AND NUMBER CITY PROVINCE OR STATE COUNTRY MONTH YEAR MONTH YEAR 1416 310 AVENUE LOS ANGELES CALIF. U.S.A. SEPT. 52 PRESENT TIME LAST FOREIGN RESIDENCE OF MORE THAN ONE YEAR IF NOT SHOWN ABOVE. (INCLUDE ALL INFORMATION REQUESTED ABOVE.) APPLICANT'S EMPLOYMENT LAST FIVE YEARS. (IF NONE, so STATE) LIST PRESENT EMPLOYMENT FIRST. FROM TO FULL NAME AND ADDRESS OF EMPLOYER OCCUPATION MONTH YEAR MONTH YEAR SELF-EMPLOYED 1800 W-8 ST. LAUNDRY JULY 41 PRESENT TIME LAST OCCUPATION ABROAD IF NOT SHOWN ABOVE. (INCLUDE ALL INFORMATION REQUESTED ABOVE.) THIS FORM IS SUBMITTED IN CONNECTION WITH APPLICATION FOR: IF YOUR NATIVE ALPHABET IS IN OTHER THAN ROMAN LETTERS. WRITE YOUR NAME IN NATURALIZATION ADJUSTMENT OF STATUS YOUR NATIVE ALPHABET IN THIS SPACE: OTHER (SPECIFY): PENALTIES: SEVERE PENALTIES ARE PROVIDED BY LAW FOR KNOWINGLY AND WILLFULLY FALSIFYING 11/6/68 OR CONCEALING A MATERIAL FACT. DATE Wing (SIGNATURE Ben OF APPLICANT OR Dong PETITIONER) COMPLETE THIS BOX (FAMILY NAME) (GIVEN NAME) (MIDDLE NAME) (ALIEN REGISTRATION NUMBER) WONG Ben Dong A12 250 430 (OTHER AGENCY USE) (INS USE) (daughter REFER no DATA in KE HUANG law (nee CH'EN) GEN-hsia LOS WHICH WAS SENT THE Los angeles (All-94-27a) etc. T/C - 249 11/10/69 (DATE) OFFICE OF INS ON local basis investigative fi : To check FB1 this reply is result of check of FILL records, request arrest to Identification Divident Dre submitted Water are necessary for positive Finger dis, 2) Rec. Br. FORM G-325A
""")

print(output)