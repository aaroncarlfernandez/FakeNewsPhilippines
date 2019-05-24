import re
from math import sqrt
import nltk as nlp
import textstat
from string import punctuation
from collections import Counter
from nltk.tokenize import word_tokenize
import dict_extract
import os

abs_path = os.path.dirname(os.path.abspath(__file__))
parse, category_names = dict_extract.load_token_parser(os.path.join(abs_path,'dictionary.dic'))

def preprocess(text):
  # if the input is HTML, force-add full stops after these tags
  fullStopTags = ['li', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dd']
  for tag in fullStopTags:
    text = re.sub(r'</' + tag + '>', '.', text)
  text = re.sub(r'<[^>]+>', '', text)  # strip out HTML
  text = re.sub(r'[,:;()\-]', ' ', text)  # replace commas, hyphens etc (count as spaces)
  text = re.sub(r'[\.!?]', '.', text)  # unify terminators
  text = re.sub(r'^\s+', '', text)  # strip leading whitespace
  text = re.sub(r'[ ]*(\n|\r\n|\r)[ ]*', ' ', text)  # replace new lines with spaces
  text = re.sub(r'([\.])[\. ]+', '.', text)  # check for duplicated terminators
  text = re.sub(r'[ ]*([\.])', '. ', text)  # pad sentence terminators
  text = re.sub(r'\s+', ' ', text)  # remove multiple spaces
  text = re.sub(r'\s+$', '', text);  # strip trailing whitespace
  return text

def letter_count(text):
    text = preprocess(text)
    newText = re.sub(r'[^A-Za-z]+', '', text)
    return len(newText)

def sentence_count(text):
  text = preprocess(text)
  return max(1, len(re.sub(r'[^\.!?]', '', text)))

def word_count(text):
    text = preprocess(text)
    return 1 + len(re.sub(r'[^ ]', '', text))  # space count + 1 is word count

def avg_words_per_sentence(text):
  text = preprocess(text)
  return 1.0 * word_count(text) / sentence_count(text)

def total_syllables(text):
    text = preprocess(text)
    words = text.split()
    return sum([syllable_count(w) for w in words])

def avg_syllables_per_word(text):
  text = preprocess(text)
  num_words = word_count(text)
  words = text.split()
  num_syllables = sum([syllable_count(w) for w in words])
  return 1.0 * num_syllables / num_words

def six_letter_word_count(text, use_proper_nouns=True):
    text = preprocess(text)
    num_long_words = 0;
    num_words = word_count(text)
    words = text.split()
    for word in words:
      if len(word) >= 6:
        if use_proper_nouns or word[:1].islower():
          num_long_words += 1
    return num_long_words

def three_syllable_word_count(text, use_proper_nouns=True):
  text = preprocess(text)
  num_long_words = 0;
  num_words = word_count(text)
  words = text.split()
  for word in words:
    if syllable_count(word) >= 3:
      if use_proper_nouns or word[:1].islower():
        num_long_words += 1
  return num_long_words

def percent_three_syllable_words(text, use_proper_nouns=True):
    text = preprocess(text)
    return 100.0 * three_syllable_word_count(text, use_proper_nouns) / word_count(text)

def syllable_count(word):
  word = word.lower()
  # remove non-alphanumeric characters
  word = re.sub(r'[^a-z]', '', word)
  word_bits = re.split(r'[^aeiouy]+', word)
  num_bits = 0
  for wb in word_bits:
    if wb != '':
      num_bits += 1
  return max(1, num_bits)

def flesch_kincaid_ease(text):
    text = preprocess(text)
    return 206.835 - (1.015 * avg_words_per_sentence(text)) - (84.6 * avg_syllables_per_word(text))

def flesch_kincaid_grade(text):
  text = preprocess(text)
  return (0.39 * avg_words_per_sentence(text)) + (11.8 * avg_syllables_per_word(text)) - 15.59

def gunning_fog(text):
    text = preprocess(text)
    return 0.4 * (avg_words_per_sentence(text) + percent_three_syllable_words(text, False))

def coleman_liau(text):
  text = preprocess(text)
  return (5.89 * letter_count(text) / word_count(text)) - (0.3 * sentence_count(text) / word_count(text)) - 15.8

def ari(text):
    text = preprocess(text)
    return (4.71 * letter_count(text) / word_count(text)) + (0.5 * word_count(text) / sentence_count(text)) - 21.43

def smog(text):
  text = preprocess(text)
  return 1.043 * sqrt((three_syllable_word_count(text) * (30.0 / sentence_count(text))) + 3.1291)

def dcrs(data):
    return textstat.dale_chall_readability_score(data)

def get_TTR(data):
  # Remove all special characters using regex
  data = re.sub(r'[^\w]', ' ', data)
  # Convert data to lowercase
  data = data.lower()
  # Tokenize the data to get word list
  tokens = nlp.word_tokenize(data)
  # Count all token and store in dictionary
  types = nlp.Counter(tokens)

  # Return Type-Token Ratio
  return (len(types) / len(tokens)) * 100

def get_allcapswordcount(data):
    return sum(map(str.isupper, data.split()))

def extract_features(text):
  tokenized_data = word_tokenize(text)
  return Counter(category for token in tokenized_data for category in parse(token))

def get_function(lwc_scores):
    return lwc_scores['function']

def get_pronoun(lwc_scores):
  return lwc_scores['pronoun']

def get_personalpronoun(lwc_scores):
    return lwc_scores['ppron']

def get_FirstPersonSingular(lwc_scores):
  return lwc_scores['i']

def get_FirstPersonPlural(lwc_scores):
    return lwc_scores['we']

def get_SecondPerson(lwc_scores):
  return lwc_scores['you']

def get_ThirdPersonSingular(lwc_scores):
    return lwc_scores['shehe']

def get_ThirdPersonPlural(lwc_scores):
  return lwc_scores['they']

def get_ImpersonalPronoun(lwc_scores):
    return lwc_scores['ipron']

def get_article(lwc_scores):
  return lwc_scores['article']

def get_Prepositions(lwc_scores):
    return lwc_scores['prep']

def get_AuxiliaryVerbs(lwc_scores):
  return lwc_scores['auxverb']

def get_CommonAdverbs(lwc_scores):
    return lwc_scores['adverb']

def get_Conjunctions(lwc_scores):
  return lwc_scores['conj']

def get_Negations(lwc_scores):
    return lwc_scores['negate']

def get_CommonVerbs(lwc_scores):
    return lwc_scores['verb']

def get_CommonAdjectives(lwc_scores):
  return lwc_scores['adj']

def get_Comparisons(lwc_scores):
    return lwc_scores['compare']

def get_Interrogatives(lwc_scores):
  return lwc_scores['interrog']

def get_ConcreteFigures(lwc_scores):
    return lwc_scores['number']

def get_Quantifiers(lwc_scores):
  return lwc_scores['quant']

def get_AffectiveProcesses(lwc_scores):
    return lwc_scores['affect']

def get_PositiveEmotion(lwc_scores):
  return lwc_scores['posemo']

def get_Achievement(lwc_scores):
    return lwc_scores['achieve']

def get_NegativeEmotion(lwc_scores):
  return lwc_scores['negemo']

def get_Anxiety(lwc_scores):
    return lwc_scores['anx']

def get_Anger(lwc_scores):
  return lwc_scores['anger']

def get_Sadness(lwc_scores):
    return lwc_scores['sad']

def get_CognitiveProcesses(lwc_scores):
  return lwc_scores['cogproc']

def get_Insight(lwc_scores):
    return lwc_scores['insight']

def get_Causation(lwc_scores):
  return lwc_scores['cause']

def get_Discrepancy(lwc_scores):
    return lwc_scores['discrep']

def get_Tentative(lwc_scores):
  return lwc_scores['tentat']

def get_Fillers(lwc_scores):
    return lwc_scores['filler']

def get_Certainty(lwc_scores):
  return lwc_scores['certain']

def get_Differentiation(lwc_scores):
    return lwc_scores['differ']

def get_InformalLanguage(lwc_scores):
  return lwc_scores['informal']

def get_SwearWords(lwc_scores):
    return lwc_scores['swear']

def get_InternetSlang(lwc_scores):
  return lwc_scores['netspeak']

def get_SexualWords(lwc_scores):
    return lwc_scores['sexual']

def get_Nonfluencies(lwc_scores):
  return lwc_scores['nonflu']

def get_PastFocus(lwc_scores):
    return lwc_scores['focuspast']

def get_PresentFocus(lwc_scores):
  return lwc_scores['focuspresent']

def get_FutureFocus(lwc_scores):
    return lwc_scores['focusfuture']

def get_punc_counter(data):
  return Counter(c for line in data for c in line if c in punctuation)

def get_allpunc(punct_scores):
  punct_sum = 0
  for c in punctuation:
    punct_sum = punct_sum + punct_scores[c]
  return punct_sum

def get_Periods(punct_scores):
    return punct_scores['.']

def get_Commas(punct_scores):
  return punct_scores[',']

def get_Colons(punct_scores):
    return punct_scores[':']

def get_Semicolons(punct_scores):
  return punct_scores[';']

def get_QuestionMarks(punct_scores):
    return punct_scores['?']

def get_ExclamationMarks(punct_scores):
  return punct_scores['!']

def get_Dashes(punct_scores):
    return punct_scores['-']

def get_Apostrophes(punct_scores):
  return punct_scores['\'']

def get_Parentheses(punct_scores):
    parentheses_sum = punct_scores['('] + punct_scores[')']
    return parentheses_sum

def get_OtherPunctuations(punct_scores):
  otherpunct_sum = 0
  for c in punctuation:
    if c in "\"#$%&*+/<=>@[\]^_`{|}~’":
      otherpunct_sum = otherpunct_sum + punct_scores[c]
  return otherpunct_sum
