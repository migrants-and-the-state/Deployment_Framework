''' 
Extract Countries and Years 
Runs
'''
from fuzzywuzzy import process
from country_list import countries_for_language
import re 
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
country_names = {name: code for code, name in countries_for_language('en')}
threshold = 90
pattern_yyyy = r'\b\d{4}\b'


def extract_actual_dates(date_list):
    actual_dates = []

    for date_str in date_list:
        # Regular expression pattern to match date formats (MM/DD/YYYY or MM DD YYYY)
        date_pattern = r'(?:(?:0?[1-9]|1[0-2])/(?:0?[1-9]|1\d|2\d|3[01])/\d{4})|(?:(?:0?[1-9]|1[0-2]) (?:0?[1-9]|1\d|2\d|3[01]) \d{4})'

        # Check if the current string matches the date pattern
        if re.match(date_pattern, date_str):
            actual_dates.append(date_str)

    return actual_dates

def remove_non_dates(date_list):
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'  # Pattern for MM/DD/YYYY or MM/DD/YY

    return [date_str for date_str in date_list if re.match(date_pattern, date_str)]


def extract_country_years(text):
    ''' 
    Use Flair's ner with regex to extract years, countries, dates
    '''
    if text is None:
        return [], [], [], [], []

    files_dates = []
    files_places = []
    files_cardinals = []
    countries_per_page = set()
    years_per_page = set()

    sentence = Sentence(text)
    tagger.predict(sentence)  # Assuming tagger is defined elsewhere

    for entity in sentence.get_spans('ner'):
        if entity.tag == 'GPE':
            files_places.append(entity.text)
            closest = process.extractOne(entity.text, country_names.keys())
            if closest[1] > threshold:
                countries_per_page.add(closest[0])

        elif entity.tag == 'CARDINAL':
            files_cardinals.append(entity.text)

        elif entity.tag == 'DATE':
            files_dates.append(entity.text)
            found_years = re.findall(pattern_yyyy, entity.text)
            years_per_page.update(found_years)

    # print("printing dates",files_dates)
    files_dates = remove_non_dates(files_dates)
    return dict(dates = files_dates, 
                cardinals  =files_cardinals, 
                places = files_places, 
                countries = list(countries_per_page),
                years = list(years_per_page))
