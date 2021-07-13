import contractions
import spacy
import unidecode
from bs4 import BeautifulSoup
from word2number import w2n


nlp = spacy.load('en_core_web_sm')

# exclude words from spacy stopwords list
deselect_stop_words = ['no', 'not']
for w in deselect_stop_words:
    nlp.vocab[w].is_stop = False


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


def text_preprocessing(text, accented_chars=True, contractions=True,
                       convert_num=True, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True,
                       stop_words=True):
    """preprocess text with default option set to true for all steps"""
    if remove_html:  # remove html tags
        text = strip_html_tags(text)
    if extra_whitespace:  # remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars:  # remove accented characters
        text = remove_accented_chars(text)
    if contractions:  # expand contractions
        text = expand_contractions(text)
    if lowercase:  # convert all characters to lowercase
        text = text.lower()

    doc = nlp(text)  # tokenise text

    clean_text = []

    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words and token.is_stop and token.pos_ != 'NUM':
            flag = False
        # remove punctuations
        if punctuations and token.pos_ == 'PUNCT' and flag:
            flag = False
        # remove special characters
        if special_chars and token.pos_ == 'SYM' and flag:
            flag = False
        # remove numbers
        if remove_num and (token.pos_ == 'NUM' or token.text.isnumeric()) and flag:
            flag = False
        # convert number words to numeric numbers
        if convert_num and token.pos_ == 'NUM' and flag:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization and token.lemma_ != "-PRON-" and flag:
            edit = token.lemma_
        # append tokens edited and not removed to list
        if edit != "" and flag:
            clean_text.append(edit)
    return clean_text
