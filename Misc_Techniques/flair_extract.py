''' 
Extract Countries and Years 
Runs
'''
from fuzzywuzzy import process
from country_list import countries_for_language
import re 
from flair.data import Sentence
from flair.models import SequenceTagger
import torch

# load tagger
tagger = SequenceTagger.load("flair/ner-english-ontonotes-large").to('cuda' if torch.cuda.is_available() else 'cpu')
country_names = {name: code for code, name in countries_for_language('en')}
threshold = 90
pattern_yyyy = r'\b\d{4}\b'
countries_list = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua & Deps', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina', 'Burundi', 'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Central African Rep', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Congo (Democratic Rep)', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Timor', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland (Republic)', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Korea North', 'Korea South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'St Kitts & Nevis', 'St Lucia', 'Saint Vincent & the Grenadines', 'Samoa', 'San Marino', 'Sao Tome & Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Togo', 'Tonga', 'Trinidad & Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe']


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
            closest = process.extractOne(entity.text, countries_list)
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
