# Blackoffer_Assignment
import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
from textblob import TextBlob
import nltk
nltk.download('punkt')

# Read URLs from input.xlsx
input_df = pd.read_excel('input.xlsx', names=['URL'])
urls = input_df['URL'].tolist()

# Create a directory to save extracted article text
if not os.path.exists('extracted_text'):
    os.makedirs('extracted_text')

# Web scraping and data extraction
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract article title and text
    article_title = soup.title.text.strip()
    article_text = ' '.join([p.text for p in soup.find_all('p')])
    
    # Save extracted article text in a text file
    url_id = url.split('/')[-2]  # Extract the URL_ID from the URL
    with open(f'extracted_text/{url_id}.txt', 'w', encoding='utf-8') as file:
        file.write(article_title + '\n' + article_text)

# Data analysis and computation of variables
output_data = []
for url in urls:
    url_id = url.split('/')[-2]
    with open(f'extracted_text/{url_id}.txt', 'r', encoding='utf-8') as file:
        article_text = file.read()

    blob = TextBlob(article_text)

    # Compute variables
    positive_score = blob.sentiment.polarity
    negative_score = -positive_score
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    sentences = nltk.sent_tokenize(article_text)
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    words = nltk.word_tokenize(article_text)
    total_words = len(words)
    complex_word_count = sum(1 for word in words if len(word) > 6)
    avg_word_length = sum(len(word) for word in words) / total_words
    syllable_count = sum(nltk.syllable_count(word) for word in words)
    avg_words_per_sentence = total_words / len(sentences)
    personal_pronouns = sum(1 for word, tag in blob.tags if tag == 'PRP')
    percentage_complex_words = (complex_word_count / total_words) * 100
    fog_index = 0.4 * (avg_words_per_sentence + percentage_complex_words)

    output_data.append([url_id, positive_score, negative_score, polarity_score, subjectivity_score,
                        avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence,
                        complex_word_count, total_words, syllable_count, personal_pronouns, avg_word_length])

# Create output DataFrame
output_df = pd.DataFrame(output_data, columns=['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                                               'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',
                                               'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
                                               'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                                               'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS',
                                               'AVG WORD LENGTH'])

# Write output DataFrame to Output Data Structure.xlsx
output_df.to_excel('Output Data Structure.xlsx', index=False)
