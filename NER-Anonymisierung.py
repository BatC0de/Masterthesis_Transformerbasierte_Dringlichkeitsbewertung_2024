import spacy
import pandas as pd
import re
import unicodedata
from datetime import datetime
from tqdm import tqdm 

# Laden der Daten
start_time = datetime.now()
nlp = spacy.load("de_dep_news_trf")
df = pd.read_csv("bspdaten.csv", sep=";", encoding="utf-8")

#Problematische Zeichen aus dem Text entfernen
def remove_unwanted_characters(text):
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüßÄÖÜ0123456789 .,-;:!?@\"'")
    normalized_text = unicodedata.normalize('NFKD', text)
    filtered_text = ''.join([c for c in normalized_text if c in allowed_chars])
    return filtered_text

#anonymisierung mit regex und NER
def anonymize_texts(texts):
    docs = nlp.pipe(texts, batch_size=50)
    anonymized_texts = []
    
    for doc in docs:
        try:
            text = doc.text
            for ent in doc.ents:
                if ent.label_ in ["PER", "LOC", "ORG", "GPE", "DATE", "MONEY"]:
                    text = text.replace(ent.text, "[ANONYM]")
            
            patterns = [
                r'\b[A-ZÄÖÜ][a-zäöüß]+\s[A-ZÄÖÜ][a-zäöüß]+\b',
                r'\b\d{2}\.\d{2}\.\d{4}\b',
                r'\b\d{5}\s?[A-ZÄÖÜa-zäöüß]+\b',
                r'\b(?:\d{3,4}\s?\d{6,7}|(?:\+49\s?|0)[1-9]\d{1,10}|(?:\+49\s?|0)[1-9](?:\s?\d{1,4}){1,4})\b',
                r'\b(?:\+?(\d[\d\-. ]+)?(?:\(\d+\)[\-. ]?)?\d[\d\-. ]+\d)\b',
                r'\b([+][ ]?[1-9][0-9][ ]?[\-]?[ ]?|[(]?[0][ ]?)[0-9]{3,4}[\-\/ ]?[ ]?[1-9][\-0-9 ]{6,16}\b',
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,8}\b',
                r'\b\d{9}\b',
                r'\b\d{11}\b',
                r'\b\d{8,10}\b',
                r'\b\d{16}\b',
                r'\b\d{2,3}-\d{6,7}\b',
                r'\b[a-zA-Z]{2}[0-9]{2}\s?[a-zA-Z0-9]{4}\s?[0-9]{4}\s?[0-9]{3}(?:[a-zA-Z0-9]\s?[a-zA-Z0-9]{0,4}\s?[a-zA-Z0-9]{0,4}\s?[a-zA-Z0-9]{0,4}\s?[a-zA-Z0-9]{0,3})?\b',
                r'\bDE( ?\d){20}\b',
                r'\b([0-9A-ZÜÄÖ][0-9A-ZÄÖÜa-züöäß&\+\-\.]{0,}[0-9A-ZÄÖÜa-züöäß\-] ?)(?:&| |\+|\-){0,3}([0-9A-ZÜÄÖ][0-9A-ZÄÖÜa-züöäß&\+\-\.]{0,}[0-9A-ZÄÖÜa-züöäß\-] ?){0,5}(e\. K\.|eG|UG(?: \(haftungsbeschränkt\))?|Limited|Stiftung|(?:g|Inv)?AG(?: \(haftungsbeschränkt\))?|(?:g|G)?mbH(?: (&|und) Co(\.)? KG(aA)?)?|GbR|OHG|PartG(?: mbB)?|K(?:g|G)(aA)?|e\.V\.|Stille Gesellschaft|Partenreederei|(?:GmbH|AG) & Co. OHG|Eigenbetrieb|Einzelunternehmen|KöR|VVag|REIT-AG)',
                r'\b\d{5}\b'
            ]
            for pattern in patterns:
                text = re.sub(pattern, '[ANONYM]', text)
            text = remove_unwanted_characters(text)
            anonymized_texts.append(text)
        except Exception as e:
            print(f"Fehler bei der Verarbeitung: {doc.text} - Fehler: {e}")
            anonymized_texts.append("[FEHLER]")
    return anonymized_texts

batch_size = 100
num_batches = (len(df) + batch_size - 1) // batch_size

anonymized_messages = []
for i in tqdm(range(num_batches), desc="Anonymisierung", ncols=100):
    batch_texts = df['message'][i * batch_size : (i + 1) * batch_size].tolist()
    anonymized_messages.extend(anonymize_texts(batch_texts))

df['message'] = anonymized_messages
df.to_csv("anon_messages.csv", index=False, encoding="utf-8-sig")
end_time = datetime.now()
num_records = len(df)
processing_time = end_time - start_time

print(f"Anonymisierung abgeschlossen. {num_records} Datensätze wurden in {processing_time} verarbeitet.")
