{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "22d52352",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "from bs4 import BeautifulSoup\n",
    "import requests\n",
    "from textblob import TextBlob\n",
    "import nltk\n",
    "nltk.download('punkt')\n",
    "\n",
    "# Read URLs from input.xlsx\n",
    "input_df = pd.read_excel('input.xlsx', names=['URL'])\n",
    "urls = input_df['URL'].tolist()\n",
    "\n",
    "# Create a directory to save extracted article text\n",
    "if not os.path.exists('extracted_text'):\n",
    "    os.makedirs('extracted_text')\n",
    "\n",
    "# Web scraping and data extraction\n",
    "for url in urls:\n",
    "    response = requests.get(url)\n",
    "    soup = BeautifulSoup(response.content, 'html.parser')\n",
    "    \n",
    "    # Extract article title and text\n",
    "    article_title = soup.title.text.strip()\n",
    "    article_text = ' '.join([p.text for p in soup.find_all('p')])\n",
    "    \n",
    "    # Save extracted article text in a text file\n",
    "    url_id = url.split('/')[-2]  # Extract the URL_ID from the URL\n",
    "    with open(f'extracted_text/{url_id}.txt', 'w', encoding='utf-8') as file:\n",
    "        file.write(article_title + '\\n' + article_text)\n",
    "\n",
    "# Data analysis and computation of variables\n",
    "output_data = []\n",
    "for url in urls:\n",
    "    url_id = url.split('/')[-2]\n",
    "    with open(f'extracted_text/{url_id}.txt', 'r', encoding='utf-8') as file:\n",
    "        article_text = file.read()\n",
    "\n",
    "    blob = TextBlob(article_text)\n",
    "\n",
    "    # Compute variables\n",
    "    positive_score = blob.sentiment.polarity\n",
    "    negative_score = -positive_score\n",
    "    polarity_score = blob.sentiment.polarity\n",
    "    subjectivity_score = blob.sentiment.subjectivity\n",
    "    sentences = nltk.sent_tokenize(article_text)\n",
    "    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)\n",
    "    words = nltk.word_tokenize(article_text)\n",
    "    total_words = len(words)\n",
    "    complex_word_count = sum(1 for word in words if len(word) > 6)\n",
    "    avg_word_length = sum(len(word) for word in words) / total_words\n",
    "    syllable_count = sum(nltk.syllable_count(word) for word in words)\n",
    "    avg_words_per_sentence = total_words / len(sentences)\n",
    "    personal_pronouns = sum(1 for word, tag in blob.tags if tag == 'PRP')\n",
    "    percentage_complex_words = (complex_word_count / total_words) * 100\n",
    "    fog_index = 0.4 * (avg_words_per_sentence + percentage_complex_words)\n",
    "\n",
    "    output_data.append([url_id, positive_score, negative_score, polarity_score, subjectivity_score,\n",
    "                        avg_sentence_length, percentage_complex_words, fog_index, avg_words_per_sentence,\n",
    "                        complex_word_count, total_words, syllable_count, personal_pronouns, avg_word_length])\n",
    "\n",
    "# Create output DataFrame\n",
    "output_df = pd.DataFrame(output_data, columns=['URL_ID', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',\n",
    "                                               'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH',\n",
    "                                               'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',\n",
    "                                               'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',\n",
    "                                               'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS',\n",
    "                                               'AVG WORD LENGTH'])\n",
    "\n",
    "# Write output DataFrame to Output Data Structure.xlsx\n",
    "output_df.to_excel('Output Data Structure.xlsx', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c175bc3c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "18e745de",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ae2a8432",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}