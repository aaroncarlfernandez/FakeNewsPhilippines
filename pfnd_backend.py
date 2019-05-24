from newspaper import Article, Config
import re
import feature_engineering as fe
import pickle
import os
from sklearn.externals.joblib import dump, load

abs_path = os.path.dirname(os.path.abspath(__file__))

def clean_data(data,headline,source,tag):

  # Remove whitespace and newlines
  cleaned_data = data.strip().replace("\n", " ").replace("\r", " ")

  # Force strip HTML meta-data, new lines, multiple spaces, leading and trailing spaces
  cleaned_data = re.sub(r'<[^>]+>', '', cleaned_data)
  cleaned_data = re.sub(r'^\s+', '', cleaned_data)
  cleaned_data = re.sub(r'[ ]*(\n|\r\n|\r)[ ]*', ' ', cleaned_data)
  cleaned_data = re.sub(r'\s+', ' ', cleaned_data)
  cleaned_data = re.sub(r'\s+$', '', cleaned_data)

  # Remove headline from news content
  if tag == 'C':
    cleaned_data = cleaned_data.replace(headline,"")

  # Remove certain strings according to expected news sources semantic structure
  if 'manilatimes' in source:
    cleaned_data = re.sub(r'home / News / Nation /  ', '', cleaned_data)
  if 'inquirer' in source:
    cleaned_data = re.sub(r'ADVERTISEMENT ', '', cleaned_data)
    cleaned_data = re.sub(r'—[A-Za-z0-9].{1,30} Read Next LATEST STORIES MOST READ', '', cleaned_data)
  if 'mb.com.ph' in source:
    cleaned_data = re.sub(r'[0-9A-Za-z].*Tweet ', '', cleaned_data)
  if 'thinkingpinoy' in source:
    cleaned_data = re.sub(r'Thinking Pinoy: ', '', cleaned_data)
  if 'getrealphilippines' in source:
    cleaned_data = re.sub(r'Share [0-9].{1,5} Shares ', '', cleaned_data)
    cleaned_data = re.sub(r' print Spread it! Tweet More Email WhatsApp', '', cleaned_data)
  if 'duterte.today' in source:
    cleaned_data = re.sub(r' Your thoughts? Leave your comments below', '', cleaned_data)
  if 'hotnewsphil' in source:
    cleaned_data = re.sub(r'ADVERTISEMENT ', '', cleaned_data)
    cleaned_data = re.sub(r'Source: [#@A-Za-z0-9].{1,100}', '', cleaned_data)
  if 'newsmediaph' in source:
    cleaned_data = re.sub(r' Sponsor [#@A-Za-z0-9].{1,1000}', '', cleaned_data)
    cleaned_data = re.sub(r'Source: [#@A-Za-z0-9].{1,100}', '', cleaned_data)
    cleaned_data = re.sub(r'Loading... ', '', cleaned_data)

  return cleaned_data

def credibility(prediction):
  if prediction == 0:
    return 'Credible'
  else:
    return 'Not Credible'

def transform_to_array(computed_features, head_cont_flag):

  """ Rearrange features according to how they were scaled, I haven't thought about how I would code this web application
  when I was training the models, hence instead of re-training the models so that I could have a better and less verbose
  python code for this project, I'll just handle it on these codes, sorry :)"""

  x = []

  if head_cont_flag == 'B':
    x.append(computed_features['cont_fkg'])
    x.append(computed_features['cont_fre'])
    x.append(computed_features['cont_cli'])
    x.append(computed_features['cont_ari'])
    x.append(computed_features['cont_dcrs'])
    x.append(computed_features['cont_gfi'])
    x.append(computed_features['cont_smog'])
    x.append(computed_features['head_word_count'])
    x.append(computed_features['cont_word_count'])
    x.append(computed_features['head_syllables_count'])
    x.append(computed_features['cont_syllables_count'])
    x.append(computed_features['head_sentence_count'])
    x.append(computed_features['cont_sentence_count'])
    x.append(computed_features['head_words_per_sentence'])
    x.append(computed_features['cont_words_per_sentence'])
    x.append(computed_features['head_long_word_count'])
    x.append(computed_features['cont_long_word_count'])
    x.append(computed_features['cont_diff_word_count'])
    x.append(computed_features['cont_diff_word_count'])
    x.append(computed_features['head_TTR'])
    x.append(computed_features['cont_TTR'])
    x.append(computed_features['head_allcaps_word_count'])
    x.append(computed_features['cont_allcaps_word_count'])
    x.append(computed_features['head_func_word'])
    x.append(computed_features['cont_func_word'])
    x.append(computed_features['head_pronoun'])
    x.append(computed_features['cont_pronoun'])
    x.append(computed_features['head_pers_pronoun'])
    x.append(computed_features['cont_pers_pronoun'])
    x.append(computed_features['head_fps_pronoun'])
    x.append(computed_features['cont_fps_pronoun'])
    x.append(computed_features['head_fpp_pronoun'])
    x.append(computed_features['cont_fpp_pronoun'])
    x.append(computed_features['head_sec_pronoun'])
    x.append(computed_features['cont_sec_pronoun'])
    x.append(computed_features['head_tps_pronoun'])
    x.append(computed_features['cont_tps_pronoun'])
    x.append(computed_features['head_tpp_pronoun'])
    x.append(computed_features['cont_tpp_pronoun'])
    x.append(computed_features['head_impersonal_pronoun'])
    x.append(computed_features['cont_impersonal_pronoun'])
    x.append(computed_features['head_article'])
    x.append(computed_features['cont_article'])
    x.append(computed_features['head_prepositions'])
    x.append(computed_features['cont_prepositions'])
    x.append(computed_features['head_aux_verbs'])
    x.append(computed_features['cont_aux_verbs'])
    x.append(computed_features['head_common_adverbs'])
    x.append(computed_features['cont_common_adverbs'])
    x.append(computed_features['head_conjunctions'])
    x.append(computed_features['cont_conjunctions'])
    x.append(computed_features['head_negations'])
    x.append(computed_features['cont_negations'])
    x.append(computed_features['head_common_verbs'])
    x.append(computed_features['cont_common_verbs'])
    x.append(computed_features['head_common_adjectives'])
    x.append(computed_features['cont_common_adjectives'])
    x.append(computed_features['head_comparisons'])
    x.append(computed_features['cont_comparisons'])
    x.append(computed_features['head_interrogatives'])
    x.append(computed_features['cont_interrogatives'])
    x.append(computed_features['head_concrete_figures'])
    x.append(computed_features['cont_concrete_figures'])
    x.append(computed_features['head_quantifiers'])
    x.append(computed_features['cont_quantifiers'])
    x.append(computed_features['head_affect_process'])
    x.append(computed_features['cont_affect_process'])
    x.append(computed_features['head_pos_emotion'])
    x.append(computed_features['cont_pos_emotion'])
    x.append(computed_features['head_achievement'])
    x.append(computed_features['cont_achievement'])
    x.append(computed_features['head_neg_emotion'])
    x.append(computed_features['cont_neg_emotion'])
    x.append(computed_features['head_anxiety'])
    x.append(computed_features['cont_anxiety'])
    x.append(computed_features['head_anger'])
    x.append(computed_features['cont_anger'])
    x.append(computed_features['head_sadness'])
    x.append(computed_features['cont_sadness'])
    x.append(computed_features['head_cognitive_process'])
    x.append(computed_features['cont_cognitive_process'])
    x.append(computed_features['head_insight'])
    x.append(computed_features['cont_insight'])
    x.append(computed_features['head_causation'])
    x.append(computed_features['cont_causation'])
    x.append(computed_features['head_discrepancy'])
    x.append(computed_features['cont_discrepancy'])
    x.append(computed_features['head_tentative'])
    x.append(computed_features['cont_tentative'])
    x.append(computed_features['head_fillers'])
    x.append(computed_features['cont_fillers'])
    x.append(computed_features['head_certainty'])
    x.append(computed_features['cont_certainty'])
    x.append(computed_features['head_differentiation'])
    x.append(computed_features['cont_differentiation'])
    x.append(computed_features['head_informal_language'])
    x.append(computed_features['cont_informal_language'])
    x.append(computed_features['head_swear_words'])
    x.append(computed_features['cont_swear_words'])
    x.append(computed_features['head_internet_slang'])
    x.append(computed_features['cont_internet_slang'])
    x.append(computed_features['head_sexual_words'])
    x.append(computed_features['cont_sexual_words'])
    x.append(computed_features['head_nonfluencies'])
    x.append(computed_features['cont_nonfluencies'])
    x.append(computed_features['head_past_focus'])
    x.append(computed_features['cont_past_focus'])
    x.append(computed_features['head_present_focus'])
    x.append(computed_features['cont_present_focus'])
    x.append(computed_features['head_future_focus'])
    x.append(computed_features['cont_future_focus'])
    x.append(computed_features['head_punctuations'])
    x.append(computed_features['cont_punctuations'])
    x.append(computed_features['head_periods'])
    x.append(computed_features['cont_periods'])
    x.append(computed_features['head_commas'])
    x.append(computed_features['cont_commas'])
    x.append(computed_features['head_colons'])
    x.append(computed_features['cont_colons'])
    x.append(computed_features['head_semicolons'])
    x.append(computed_features['cont_semicolons'])
    x.append(computed_features['head_question_marks'])
    x.append(computed_features['cont_question_marks'])
    x.append(computed_features['head_exclam_marks'])
    x.append(computed_features['cont_exclam_marks'])
    x.append(computed_features['head_dashes'])
    x.append(computed_features['cont_dashes'])
    x.append(computed_features['head_apostrophes'])
    x.append(computed_features['cont_apostrophes'])
    x.append(computed_features['head_parentheses'])
    x.append(computed_features['cont_parentheses'])
    x.append(computed_features['head_other_puncts'])
    x.append(computed_features['cont_other_puncts'])
  elif head_cont_flag == 'H':
    x.append(computed_features['head_word_count'])
    x.append(computed_features['head_syllables_count'])
    x.append(computed_features['head_sentence_count'])
    x.append(computed_features['head_words_per_sentence'])
    x.append(computed_features['head_long_word_count'])
    x.append(computed_features['head_diff_word_count'])
    x.append(computed_features['head_TTR'])
    x.append(computed_features['head_allcaps_word_count'])
    x.append(computed_features['head_func_word'])
    x.append(computed_features['head_pronoun'])
    x.append(computed_features['head_pers_pronoun'])
    x.append(computed_features['head_fps_pronoun'])
    x.append(computed_features['head_fpp_pronoun'])
    x.append(computed_features['head_sec_pronoun'])
    x.append(computed_features['head_tps_pronoun'])
    x.append(computed_features['head_tpp_pronoun'])
    x.append(computed_features['head_impersonal_pronoun'])
    x.append(computed_features['head_article'])
    x.append(computed_features['head_prepositions'])
    x.append(computed_features['head_aux_verbs'])
    x.append(computed_features['head_common_adverbs'])
    x.append(computed_features['head_conjunctions'])
    x.append(computed_features['head_negations'])
    x.append(computed_features['head_common_verbs'])
    x.append(computed_features['head_common_adjectives'])
    x.append(computed_features['head_comparisons'])
    x.append(computed_features['head_interrogatives'])
    x.append(computed_features['head_concrete_figures'])
    x.append(computed_features['head_quantifiers'])
    x.append(computed_features['head_affect_process'])
    x.append(computed_features['head_pos_emotion'])
    x.append(computed_features['head_achievement'])
    x.append(computed_features['head_neg_emotion'])
    x.append(computed_features['head_anxiety'])
    x.append(computed_features['head_anger'])
    x.append(computed_features['head_sadness'])
    x.append(computed_features['head_cognitive_process'])
    x.append(computed_features['head_insight'])
    x.append(computed_features['head_causation'])
    x.append(computed_features['head_discrepancy'])
    x.append(computed_features['head_tentative'])
    x.append(computed_features['head_fillers'])
    x.append(computed_features['head_certainty'])
    x.append(computed_features['head_differentiation'])
    x.append(computed_features['head_informal_language'])
    x.append(computed_features['head_swear_words'])
    x.append(computed_features['head_internet_slang'])
    x.append(computed_features['head_sexual_words'])
    x.append(computed_features['head_nonfluencies'])
    x.append(computed_features['head_past_focus'])
    x.append(computed_features['head_present_focus'])
    x.append(computed_features['head_future_focus'])
    x.append(computed_features['head_punctuations'])
    x.append(computed_features['head_periods'])
    x.append(computed_features['head_commas'])
    x.append(computed_features['head_colons'])
    x.append(computed_features['head_semicolons'])
    x.append(computed_features['head_question_marks'])
    x.append(computed_features['head_exclam_marks'])
    x.append(computed_features['head_dashes'])
    x.append(computed_features['head_apostrophes'])
    x.append(computed_features['head_parentheses'])
    x.append(computed_features['head_other_puncts'])
  elif head_cont_flag == 'C':
    x.append(computed_features['cont_fkg'])
    x.append(computed_features['cont_fre'])
    x.append(computed_features['cont_cli'])
    x.append(computed_features['cont_ari'])
    x.append(computed_features['cont_dcrs'])
    x.append(computed_features['cont_gfi'])
    x.append(computed_features['cont_smog'])
    x.append(computed_features['cont_word_count'])
    x.append(computed_features['cont_syllables_count'])
    x.append(computed_features['cont_sentence_count'])
    x.append(computed_features['cont_words_per_sentence'])
    x.append(computed_features['cont_long_word_count'])
    x.append(computed_features['cont_diff_word_count'])
    x.append(computed_features['cont_TTR'])
    x.append(computed_features['cont_allcaps_word_count'])
    x.append(computed_features['cont_func_word'])
    x.append(computed_features['cont_pronoun'])
    x.append(computed_features['cont_pers_pronoun'])
    x.append(computed_features['cont_fps_pronoun'])
    x.append(computed_features['cont_fpp_pronoun'])
    x.append(computed_features['cont_sec_pronoun'])
    x.append(computed_features['cont_tps_pronoun'])
    x.append(computed_features['cont_tpp_pronoun'])
    x.append(computed_features['cont_impersonal_pronoun'])
    x.append(computed_features['cont_article'])
    x.append(computed_features['cont_prepositions'])
    x.append(computed_features['cont_aux_verbs'])
    x.append(computed_features['cont_common_adverbs'])
    x.append(computed_features['cont_conjunctions'])
    x.append(computed_features['cont_negations'])
    x.append(computed_features['cont_common_verbs'])
    x.append(computed_features['cont_common_adjectives'])
    x.append(computed_features['cont_comparisons'])
    x.append(computed_features['cont_interrogatives'])
    x.append(computed_features['cont_concrete_figures'])
    x.append(computed_features['cont_quantifiers'])
    x.append(computed_features['cont_affect_process'])
    x.append(computed_features['cont_pos_emotion'])
    x.append(computed_features['cont_achievement'])
    x.append(computed_features['cont_neg_emotion'])
    x.append(computed_features['cont_anxiety'])
    x.append(computed_features['cont_anger'])
    x.append(computed_features['cont_sadness'])
    x.append(computed_features['cont_cognitive_process'])
    x.append(computed_features['cont_insight'])
    x.append(computed_features['cont_causation'])
    x.append(computed_features['cont_discrepancy'])
    x.append(computed_features['cont_tentative'])
    x.append(computed_features['cont_fillers'])
    x.append(computed_features['cont_certainty'])
    x.append(computed_features['cont_differentiation'])
    x.append(computed_features['cont_informal_language'])
    x.append(computed_features['cont_swear_words'])
    x.append(computed_features['cont_internet_slang'])
    x.append(computed_features['cont_sexual_words'])
    x.append(computed_features['cont_nonfluencies'])
    x.append(computed_features['cont_past_focus'])
    x.append(computed_features['cont_present_focus'])
    x.append(computed_features['cont_future_focus'])
    x.append(computed_features['cont_punctuations'])
    x.append(computed_features['cont_periods'])
    x.append(computed_features['cont_commas'])
    x.append(computed_features['cont_colons'])
    x.append(computed_features['cont_semicolons'])
    x.append(computed_features['cont_question_marks'])
    x.append(computed_features['cont_exclam_marks'])
    x.append(computed_features['cont_dashes'])
    x.append(computed_features['cont_apostrophes'])
    x.append(computed_features['cont_parentheses'])
    x.append(computed_features['cont_other_puncts'])

  return [x] # Return as a 2-dimensional array

def scale_computed_features(computed_features,head_cont_flag):
  if head_cont_flag == 'B':
    x = transform_to_array(computed_features,'B')
    scaler = load(os.path.join(abs_path,'robust_scaler_both.bin'))
  elif head_cont_flag == 'H':
    x = transform_to_array(computed_features, 'H')
    scaler = load(os.path.join(abs_path,'robust_scaler_head.bin'))
  elif head_cont_flag == 'C':
    x = transform_to_array(computed_features, 'C')
    scaler = load(os.path.join(abs_path,'robust_scaler_cont.bin'))

  return scaler.transform(x)

def get_selected_features(computed_features,learning_model,head_cont_flag):
  if head_cont_flag == 'B':
    x = scale_computed_features(computed_features,'B')
  elif head_cont_flag == 'H':
    x = scale_computed_features(computed_features, 'H')
  elif head_cont_flag == 'C':
    x = scale_computed_features(computed_features, 'C')

  """ Map the computed features according to the selected features of each of the models. Basically, the arrays hard coded
  belowe were the actual displacement used during the training. I know that makes the readability of this code poor
  but it is the faster way of coding this instead of mapping it per feature """

  if learning_model == 'GNB' and head_cont_flag == 'B':
    X = x[:,[4, 7, 9, 13, 15, 19, 21, 22, 23, 26, 28, 30, 32, 34, 36, 48, 53, 65, 68, 74, 76, 78,
             79, 86, 92, 96, 106, 107, 110, 118, 120, 122, 124, 126, 128, 132]]
  elif learning_model == 'LR' and head_cont_flag == 'B':
    X = x[:,[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 25, 26, 31, 32,
             33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
             54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
             75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 96, 97, 98, 100,
             101, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 118, 119, 120, 121,
             122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]]
  elif learning_model == 'SVM' and head_cont_flag == 'B':
    X = x[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24,
             25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
             73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96,
             97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
             116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131]]
  elif learning_model == 'GNB' and head_cont_flag == 'H':
    X = x[:,[0, 1, 3, 4, 5, 6, 7, 8, 17, 23, 24, 29, 36, 50, 52]]
  elif learning_model == 'LR' and head_cont_flag == 'H':
    X = x[:,[0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23, 24, 25, 26, 27, 30, 32,
             33, 35, 36, 37, 38, 40, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
             59, 60, 62]]
  elif learning_model == 'SVM' and head_cont_flag == 'H':
    X = x[:,[0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23, 24, 25, 26, 27, 29,
             30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
             54, 55, 56, 57, 58, 59, 60, 62]]
  elif learning_model == 'GNB' and head_cont_flag == 'C':
    X = x[:,[4, 14, 16, 17, 18, 19, 20, 21, 27, 37, 39, 40, 41, 42, 46, 49, 51, 56, 58, 62, 63, 64,
             65, 66, 67, 69]]
  elif learning_model == 'LR' and head_cont_flag == 'C':
    X = x[:,[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19, 21, 22, 23, 24, 25, 26, 27, 28, 30,
             31, 32, 33, 35, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
             57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]]
  elif learning_model == 'SVM' and head_cont_flag == 'C':
    X = x[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 21, 22, 23, 24, 25, 26, 27, 28,
             30, 31, 32, 33, 35, 36, 37, 39, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 56, 57,
             59, 60, 61, 62, 63, 64, 65, 66, 67, 68]]

  return X

def predict_news(computed_features, head_cont_flag):
  prediction_and_probability = {}

  if head_cont_flag == 'B':
    """ Gaussian Naive Bayes """
    X_GNB = get_selected_features(computed_features,'GNB','B')
    WF_GNB_Final = pickle.load(open(os.path.join(abs_path, 'WF_GNB_Final.pkl'), 'rb'))

    proba_gnb = WF_GNB_Final.predict_proba(X_GNB)

    prediction_and_probability['gnb_leg_prob'] = '%.2f' % (proba_gnb[0][0] * 100)
    prediction_and_probability['gnb_fake_prob'] = '%.2f' % (proba_gnb[0][1] * 100)

    """ Logistic Regression """
    X_LR = get_selected_features(computed_features,'LR','B')
    WF_LR_Final = pickle.load(open(os.path.join(abs_path, 'WF_LR_Final.pkl'), 'rb'))

    proba_lr = WF_LR_Final.predict_proba(X_LR)

    prediction_and_probability['lr_leg_prob'] = '%.2f' % (proba_lr[0][0] * 100)
    prediction_and_probability['lr_fake_prob'] = '%.2f' % (proba_lr[0][1] * 100)

    """ Support Vector Machine """
    X_SVM = get_selected_features(computed_features,'SVM','B')
    WF_SVM_Final = pickle.load(open(os.path.join(abs_path, 'WF_SVM_Final.pkl'), 'rb'))

    proba_svm = WF_SVM_Final.predict_proba(X_SVM)

    prediction_and_probability['svm_leg_prob'] = '%.2f' % (proba_svm[0][0] * 100)
    prediction_and_probability['svm_fake_prob'] = '%.2f' % (proba_svm[0][1] * 100)

  elif head_cont_flag == 'H':
    """ Gaussian Naive Bayes """
    X_GNB = get_selected_features(computed_features,'GNB','H')
    WF_GNB_Final = pickle.load(open(os.path.join(abs_path, 'Head_GNB_Final.pkl'), 'rb'))

    proba_gnb = WF_GNB_Final.predict_proba(X_GNB)

    prediction_and_probability['gnb_leg_prob'] = '%.2f' % (proba_gnb[0][0] * 100)
    prediction_and_probability['gnb_fake_prob'] = '%.2f' % (proba_gnb[0][1] * 100)

    """ Logistic Regression """
    X_LR = get_selected_features(computed_features,'LR','H')
    WF_LR_Final = pickle.load(open(os.path.join(abs_path, 'Head_LR_Final.pkl'), 'rb'))

    proba_lr = WF_LR_Final.predict_proba(X_LR)

    prediction_and_probability['lr_leg_prob'] = '%.2f' % (proba_lr[0][0] * 100)
    prediction_and_probability['lr_fake_prob'] = '%.2f' % (proba_lr[0][1] * 100)

    """ Support Vector Machine """
    X_SVM = get_selected_features(computed_features,'SVM','H')
    WF_SVM_Final = pickle.load(open(os.path.join(abs_path, 'Head_SVM_Final.pkl'), 'rb'))

    proba_svm = WF_SVM_Final.predict_proba(X_SVM)

    prediction_and_probability['svm_leg_prob'] = '%.2f' % (proba_svm[0][0] * 100)
    prediction_and_probability['svm_fake_prob'] = '%.2f' % (proba_svm[0][1] * 100)

  elif head_cont_flag == 'C':
    """ Gaussian Naive Bayes """
    X_GNB = get_selected_features(computed_features,'GNB','C')
    WF_GNB_Final = pickle.load(open(os.path.join(abs_path, 'Cont_GNB_Final.pkl'), 'rb'))

    proba_gnb = WF_GNB_Final.predict_proba(X_GNB)

    prediction_and_probability['gnb_leg_prob'] = '%.2f' % (proba_gnb[0][0] * 100)
    prediction_and_probability['gnb_fake_prob'] = '%.2f' % (proba_gnb[0][1] * 100)

    """ Logistic Regression """
    X_LR = get_selected_features(computed_features,'LR','C')
    WF_LR_Final = pickle.load(open(os.path.join(abs_path, 'Cont_LR_Final.pkl'), 'rb'))

    proba_lr = WF_LR_Final.predict_proba(X_LR)

    prediction_and_probability['lr_leg_prob'] = '%.2f' % (proba_lr[0][0] * 100)
    prediction_and_probability['lr_fake_prob'] = '%.2f' % (proba_lr[0][1] * 100)

    """ Support Vector Machine """
    X_SVM = get_selected_features(computed_features,'SVM','C')
    WF_SVM_Final = pickle.load(open(os.path.join(abs_path, 'Cont_SVM_Final.pkl'), 'rb'))

    proba_svm = WF_SVM_Final.predict_proba(X_SVM)

    prediction_and_probability['svm_leg_prob'] = '%.2f' % (proba_svm[0][0] * 100)
    prediction_and_probability['svm_fake_prob'] = '%.2f' % (proba_svm[0][1] * 100)

  return prediction_and_probability

def compute_features(headline,content):
  computed_features = {}

  if headline != '':
    head_punct_scores = fe.get_punc_counter(headline)
    head_lwc_scores = fe.extract_features(headline.lower())

    computed_features['head_word_count'] = '%.2f'%(fe.word_count(headline))
    computed_features['head_syllables_count'] = '%.2f'%(fe.total_syllables(headline))
    computed_features['head_sentence_count'] = '%.2f'%(fe.sentence_count(headline))
    computed_features['head_words_per_sentence'] = '%.2f'%(fe.avg_words_per_sentence(headline))
    computed_features['head_long_word_count'] = '%.2f'%(fe.six_letter_word_count(headline))
    computed_features['head_diff_word_count'] = '%.2f'%(fe.three_syllable_word_count(headline))
    computed_features['head_TTR'] = '%.2f'%(fe.get_TTR(headline))
    computed_features['head_allcaps_word_count'] = '%.2f'%(fe.get_allcapswordcount(headline))

    computed_features['head_func_word'] = '%.2f'%(fe.get_function(head_lwc_scores))
    computed_features['head_pronoun'] = '%.2f'%(fe.get_pronoun(head_lwc_scores))
    computed_features['head_pers_pronoun'] = '%.2f'%(fe.get_personalpronoun(head_lwc_scores))
    computed_features['head_fps_pronoun'] = '%.2f'%(fe.get_FirstPersonSingular(head_lwc_scores))
    computed_features['head_fpp_pronoun'] = '%.2f'%(fe.get_FirstPersonPlural(head_lwc_scores))
    computed_features['head_sec_pronoun'] = '%.2f'%(fe.get_SecondPerson(head_lwc_scores))
    computed_features['head_tps_pronoun'] = '%.2f'%(fe.get_ThirdPersonSingular(head_lwc_scores))
    computed_features['head_tpp_pronoun'] = '%.2f'%(fe.get_ThirdPersonPlural(head_lwc_scores))
    computed_features['head_impersonal_pronoun'] = '%.2f'%(fe.get_ImpersonalPronoun(head_lwc_scores))
    computed_features['head_article'] = '%.2f'%(fe.get_article(head_lwc_scores))
    computed_features['head_prepositions'] = '%.2f'%(fe.get_Prepositions(head_lwc_scores))
    computed_features['head_aux_verbs'] = '%.2f'%(fe.get_AuxiliaryVerbs(head_lwc_scores))
    computed_features['head_common_adverbs'] = '%.2f'%(fe.get_CommonAdverbs(head_lwc_scores))
    computed_features['head_conjunctions'] = '%.2f'%(fe.get_Conjunctions(head_lwc_scores))
    computed_features['head_negations'] = '%.2f'%(fe.get_Negations(head_lwc_scores))
    computed_features['head_common_verbs'] = '%.2f'%(fe.get_CommonVerbs(head_lwc_scores))
    computed_features['head_common_adjectives'] = '%.2f'%(fe.get_CommonAdjectives(head_lwc_scores))
    computed_features['head_comparisons'] = '%.2f'%(fe.get_Comparisons(head_lwc_scores))
    computed_features['head_interrogatives'] = '%.2f'%(fe.get_Interrogatives(head_lwc_scores))
    computed_features['head_concrete_figures'] = '%.2f'%(fe.get_ConcreteFigures(head_lwc_scores))
    computed_features['head_quantifiers'] = '%.2f'%(fe.get_Quantifiers(head_lwc_scores))
    computed_features['head_affect_process'] = '%.2f'%(fe.get_AffectiveProcesses(head_lwc_scores))
    computed_features['head_pos_emotion'] = '%.2f'%(fe.get_PositiveEmotion(head_lwc_scores))
    computed_features['head_achievement'] = '%.2f'%(fe.get_Achievement(head_lwc_scores))
    computed_features['head_neg_emotion'] = '%.2f'%(fe.get_NegativeEmotion(head_lwc_scores))
    computed_features['head_anxiety'] = '%.2f'%(fe.get_Anxiety(head_lwc_scores))
    computed_features['head_anger'] = '%.2f'%(fe.get_Anger(head_lwc_scores))
    computed_features['head_sadness'] = '%.2f'%(fe.get_Sadness(head_lwc_scores))
    computed_features['head_cognitive_process'] = '%.2f'%(fe.get_CognitiveProcesses(head_lwc_scores))
    computed_features['head_insight'] = '%.2f'%(fe.get_Insight(head_lwc_scores))
    computed_features['head_causation'] = '%.2f'%(fe.get_Causation(head_lwc_scores))
    computed_features['head_discrepancy'] = '%.2f'%(fe.get_Discrepancy(head_lwc_scores))
    computed_features['head_tentative'] = '%.2f'%(fe.get_Tentative(head_lwc_scores))
    computed_features['head_fillers'] = '%.2f'%(fe.get_Fillers(head_lwc_scores))
    computed_features['head_certainty'] = '%.2f'%(fe.get_Certainty(head_lwc_scores))
    computed_features['head_differentiation'] = '%.2f'%(fe.get_Differentiation(head_lwc_scores))
    computed_features['head_informal_language'] = '%.2f'%(fe.get_InformalLanguage(head_lwc_scores))
    computed_features['head_swear_words'] = '%.2f'%(fe.get_SwearWords(head_lwc_scores))
    computed_features['head_internet_slang'] = '%.2f'%(fe.get_InternetSlang(head_lwc_scores))
    computed_features['head_sexual_words'] = '%.2f'%(fe.get_SexualWords(head_lwc_scores))
    computed_features['head_nonfluencies'] = '%.2f'%(fe.get_Nonfluencies(head_lwc_scores))
    computed_features['head_past_focus'] = '%.2f'%(fe.get_PastFocus(head_lwc_scores))
    computed_features['head_present_focus'] = '%.2f'%(fe.get_PresentFocus(head_lwc_scores))
    computed_features['head_future_focus'] = '%.2f'%(fe.get_FutureFocus(head_lwc_scores))

    computed_features['head_punctuations'] = '%.2f'%(fe.get_allpunc(head_punct_scores))
    computed_features['head_periods'] = '%.2f'%(fe.get_Periods(head_punct_scores))
    computed_features['head_commas'] = '%.2f'%(fe.get_Commas(head_punct_scores))
    computed_features['head_colons'] = '%.2f'%(fe.get_Colons(head_punct_scores))
    computed_features['head_semicolons'] = '%.2f'%(fe.get_Semicolons(head_punct_scores))
    computed_features['head_question_marks'] = '%.2f'%(fe.get_QuestionMarks(head_punct_scores))
    computed_features['head_exclam_marks'] = '%.2f'%(fe.get_ExclamationMarks(head_punct_scores))
    computed_features['head_dashes'] = '%.2f'%(fe.get_Dashes(head_punct_scores))
    computed_features['head_apostrophes'] = '%.2f'%(fe.get_Apostrophes(head_punct_scores))
    computed_features['head_parentheses'] = '%.2f'%(fe.get_Parentheses(head_punct_scores))
    computed_features['head_other_puncts'] = '%.2f'%(fe.get_OtherPunctuations(head_punct_scores))
  else:
    computed_features['head_word_count'] = '%.2f'%(0.00)
    computed_features['head_syllables_count'] = '%.2f'%(0.00)
    computed_features['head_sentence_count'] = '%.2f'%(0.00)
    computed_features['head_words_per_sentence'] = '%.2f'%(0.00)
    computed_features['head_long_word_count'] = '%.2f'%(0.00)
    computed_features['head_diff_word_count'] = '%.2f'%(0.00)
    computed_features['head_TTR'] = '%.2f'%(0.00)
    computed_features['head_allcaps_word_count'] = '%.2f'%(0.00)
    computed_features['head_func_word'] = '%.2f'%(0.00)
    computed_features['head_pronoun'] = '%.2f'%(0.00)
    computed_features['head_pers_pronoun'] = '%.2f'%(0.00)
    computed_features['head_fps_pronoun'] = '%.2f'%(0.00)
    computed_features['head_fpp_pronoun'] = '%.2f'%(0.00)
    computed_features['head_sec_pronoun'] = '%.2f'%(0.00)
    computed_features['head_tps_pronoun'] = '%.2f'%(0.00)
    computed_features['head_tpp_pronoun'] = '%.2f'%(0.00)
    computed_features['head_impersonal_pronoun'] = '%.2f'%(0.00)
    computed_features['head_article'] = '%.2f'%(0.00)
    computed_features['head_prepositions'] = '%.2f'%(0.00)
    computed_features['head_aux_verbs'] = '%.2f'%(0.00)
    computed_features['head_common_adverbs'] = '%.2f'%(0.00)
    computed_features['head_conjunctions'] = '%.2f'%(0.00)
    computed_features['head_negations'] = '%.2f'%(0.00)
    computed_features['head_common_verbs'] = '%.2f'%(0.00)
    computed_features['head_common_adjectives'] = '%.2f'%(0.00)
    computed_features['head_comparisons'] = '%.2f'%(0.00)
    computed_features['head_interrogatives'] = '%.2f'%(0.00)
    computed_features['head_concrete_figures'] = '%.2f'%(0.00)
    computed_features['head_quantifiers'] = '%.2f'%(0.00)
    computed_features['head_affect_process'] = '%.2f'%(0.00)
    computed_features['head_pos_emotion'] = '%.2f'%(0.00)
    computed_features['head_achievement'] = '%.2f'%(0.00)
    computed_features['head_neg_emotion'] = '%.2f'%(0.00)
    computed_features['head_anxiety'] = '%.2f'%(0.00)
    computed_features['head_anger'] = '%.2f'%(0.00)
    computed_features['head_sadness'] = '%.2f'%(0.00)
    computed_features['head_cognitive_process'] = '%.2f'%(0.00)
    computed_features['head_insight'] = '%.2f'%(0.00)
    computed_features['head_causation'] = '%.2f'%(0.00)
    computed_features['head_discrepancy'] = '%.2f'%(0.00)
    computed_features['head_tentative'] = '%.2f'%(0.00)
    computed_features['head_fillers'] = '%.2f'%(0.00)
    computed_features['head_certainty'] = '%.2f'%(0.00)
    computed_features['head_differentiation'] = '%.2f'%(0.00)
    computed_features['head_informal_language'] = '%.2f'%(0.00)
    computed_features['head_swear_words'] = '%.2f'%(0.00)
    computed_features['head_internet_slang'] = '%.2f'%(0.00)
    computed_features['head_sexual_words'] = '%.2f'%(0.00)
    computed_features['head_nonfluencies'] = '%.2f'%(0.00)
    computed_features['head_past_focus'] = '%.2f'%(0.00)
    computed_features['head_present_focus'] = '%.2f'%(0.00)
    computed_features['head_future_focus'] = '%.2f'%(0.00)
    computed_features['head_punctuations'] = '%.2f'%(0.00)
    computed_features['head_periods'] = '%.2f'%(0.00)
    computed_features['head_commas'] = '%.2f'%(0.00)
    computed_features['head_colons'] = '%.2f'%(0.00)
    computed_features['head_semicolons'] = '%.2f'%(0.00)
    computed_features['head_question_marks'] = '%.2f'%(0.00)
    computed_features['head_exclam_marks'] = '%.2f'%(0.00)
    computed_features['head_dashes'] = '%.2f'%(0.00)
    computed_features['head_apostrophes'] = '%.2f'%(0.00)
    computed_features['head_parentheses'] = '%.2f'%(0.00)
    computed_features['head_other_puncts'] = '%.2f'%(0.00)

  if content != '':
    cont_punct_scores = fe.get_punc_counter(content)
    cont_lwc_scores = fe.extract_features(content)

    computed_features['cont_fkg'] = '%.2f'%(fe.flesch_kincaid_grade(content))
    computed_features['cont_fre'] = '%.2f'%(fe.flesch_kincaid_ease(content))
    computed_features['cont_cli'] = '%.2f'%(fe.coleman_liau(content))
    computed_features['cont_ari'] = '%.2f'%(fe.ari(content))
    computed_features['cont_dcrs'] = '%.2f'%(fe.dcrs(content))
    computed_features['cont_gfi'] = '%.2f'%(fe.gunning_fog(content))
    computed_features['cont_smog'] = '%.2f'%(fe.smog(content))

    computed_features['cont_word_count'] = '%.2f'%(fe.word_count(content))
    computed_features['cont_syllables_count'] = '%.2f'%(fe.total_syllables(content))
    computed_features['cont_sentence_count'] = '%.2f'%(fe.sentence_count(content))
    computed_features['cont_words_per_sentence'] = '%.2f'%(fe.avg_words_per_sentence(content))
    computed_features['cont_long_word_count'] = '%.2f'%(fe.six_letter_word_count(content))
    computed_features['cont_diff_word_count'] = '%.2f'%(fe.three_syllable_word_count(content))
    computed_features['cont_TTR'] = '%.2f'%(fe.get_TTR(content))
    computed_features['cont_allcaps_word_count'] = '%.2f'%(fe.get_allcapswordcount(content))

    computed_features['cont_func_word'] = '%.2f'%(fe.get_function(cont_lwc_scores))
    computed_features['cont_pronoun'] = '%.2f'%(fe.get_pronoun(cont_lwc_scores))
    computed_features['cont_pers_pronoun'] = '%.2f'%(fe.get_personalpronoun(cont_lwc_scores))
    computed_features['cont_fps_pronoun'] = '%.2f'%(fe.get_FirstPersonSingular(cont_lwc_scores))
    computed_features['cont_fpp_pronoun'] = '%.2f'%(fe.get_FirstPersonPlural(cont_lwc_scores))
    computed_features['cont_sec_pronoun'] = '%.2f'%(fe.get_SecondPerson(cont_lwc_scores))
    computed_features['cont_tps_pronoun'] = '%.2f'%(fe.get_ThirdPersonSingular(cont_lwc_scores))
    computed_features['cont_tpp_pronoun'] = '%.2f'%(fe.get_ThirdPersonPlural(cont_lwc_scores))
    computed_features['cont_impersonal_pronoun'] = '%.2f'%(fe.get_ImpersonalPronoun(cont_lwc_scores))
    computed_features['cont_article'] = '%.2f'%(fe.get_article(cont_lwc_scores))
    computed_features['cont_prepositions'] = '%.2f'%(fe.get_Prepositions(cont_lwc_scores))
    computed_features['cont_aux_verbs'] = '%.2f'%(fe.get_AuxiliaryVerbs(cont_lwc_scores))
    computed_features['cont_common_adverbs'] = '%.2f'%(fe.get_CommonAdverbs(cont_lwc_scores))
    computed_features['cont_conjunctions'] = '%.2f'%(fe.get_Conjunctions(cont_lwc_scores))
    computed_features['cont_negations'] = '%.2f'%(fe.get_Negations(cont_lwc_scores))
    computed_features['cont_common_verbs'] = '%.2f'%(fe.get_CommonVerbs(cont_lwc_scores))
    computed_features['cont_common_adjectives'] = '%.2f'%(fe.get_CommonAdjectives(cont_lwc_scores))
    computed_features['cont_comparisons'] = '%.2f'%(fe.get_Comparisons(cont_lwc_scores))
    computed_features['cont_interrogatives'] = '%.2f'%(fe.get_Interrogatives(cont_lwc_scores))
    computed_features['cont_concrete_figures'] = '%.2f'%(fe.get_ConcreteFigures(cont_lwc_scores))
    computed_features['cont_quantifiers'] = '%.2f'%(fe.get_Quantifiers(cont_lwc_scores))
    computed_features['cont_affect_process'] = '%.2f'%(fe.get_AffectiveProcesses(cont_lwc_scores))
    computed_features['cont_pos_emotion'] = '%.2f'%(fe.get_PositiveEmotion(cont_lwc_scores))
    computed_features['cont_achievement'] = '%.2f'%(fe.get_Achievement(cont_lwc_scores))
    computed_features['cont_neg_emotion'] = '%.2f'%(fe.get_NegativeEmotion(cont_lwc_scores))
    computed_features['cont_anxiety'] = '%.2f'%(fe.get_Anxiety(cont_lwc_scores))
    computed_features['cont_anger'] = '%.2f'%(fe.get_Anger(cont_lwc_scores))
    computed_features['cont_sadness'] = '%.2f'%(fe.get_Sadness(cont_lwc_scores))
    computed_features['cont_cognitive_process'] = '%.2f'%(fe.get_CognitiveProcesses(cont_lwc_scores))
    computed_features['cont_insight'] = '%.2f'%(fe.get_Insight(cont_lwc_scores))
    computed_features['cont_causation'] = '%.2f'%(fe.get_Causation(cont_lwc_scores))
    computed_features['cont_discrepancy'] = '%.2f'%(fe.get_Discrepancy(cont_lwc_scores))
    computed_features['cont_tentative'] = '%.2f'%(fe.get_Tentative(cont_lwc_scores))
    computed_features['cont_fillers'] = '%.2f'%(fe.get_Fillers(cont_lwc_scores))
    computed_features['cont_certainty'] = '%.2f'%(fe.get_Certainty(cont_lwc_scores))
    computed_features['cont_differentiation'] = '%.2f'%(fe.get_Differentiation(cont_lwc_scores))
    computed_features['cont_informal_language'] = '%.2f'%(fe.get_InformalLanguage(cont_lwc_scores))
    computed_features['cont_swear_words'] = '%.2f'%(fe.get_SwearWords(cont_lwc_scores))
    computed_features['cont_internet_slang'] = '%.2f'%(fe.get_InternetSlang(cont_lwc_scores))
    computed_features['cont_sexual_words'] = '%.2f'%(fe.get_SexualWords(cont_lwc_scores))
    computed_features['cont_nonfluencies'] = '%.2f'%(fe.get_Nonfluencies(cont_lwc_scores))
    computed_features['cont_past_focus'] = '%.2f'%(fe.get_PastFocus(cont_lwc_scores))
    computed_features['cont_present_focus'] = '%.2f'%(fe.get_PresentFocus(cont_lwc_scores))
    computed_features['cont_future_focus'] = '%.2f'%(fe.get_FutureFocus(cont_lwc_scores))

    computed_features['cont_punctuations'] = '%.2f'%(fe.get_allpunc(cont_punct_scores))
    computed_features['cont_periods'] = '%.2f'%(fe.get_Periods(cont_punct_scores))
    computed_features['cont_commas'] = '%.2f'%(fe.get_Commas(cont_punct_scores))
    computed_features['cont_colons'] = '%.2f'%(fe.get_Colons(cont_punct_scores))
    computed_features['cont_semicolons'] = '%.2f'%(fe.get_Semicolons(cont_punct_scores))
    computed_features['cont_question_marks'] = '%.2f'%(fe.get_QuestionMarks(cont_punct_scores))
    computed_features['cont_exclam_marks'] = '%.2f'%(fe.get_ExclamationMarks(cont_punct_scores))
    computed_features['cont_dashes'] = '%.2f'%(fe.get_Dashes(cont_punct_scores))
    computed_features['cont_apostrophes'] = '%.2f'%(fe.get_Apostrophes(cont_punct_scores))
    computed_features['cont_parentheses'] = '%.2f'%(fe.get_Parentheses(cont_punct_scores))
    computed_features['cont_other_puncts'] = '%.2f'%(fe.get_OtherPunctuations(cont_punct_scores))
  else:
    computed_features['cont_fkg'] = '%.2f'%(0.00)
    computed_features['cont_fre'] = '%.2f'%(0.00)
    computed_features['cont_cli'] = '%.2f'%(0.00)
    computed_features['cont_ari'] = '%.2f'%(0.00)
    computed_features['cont_dcrs'] = '%.2f'%(0.00)
    computed_features['cont_gfi'] = '%.2f'%(0.00)
    computed_features['cont_smog'] = '%.2f'%(0.00)
    computed_features['cont_word_count'] = '%.2f'%(0.00)
    computed_features['cont_syllables_count'] = '%.2f'%(0.00)
    computed_features['cont_sentence_count'] = '%.2f'%(0.00)
    computed_features['cont_words_per_sentence'] = '%.2f'%(0.00)
    computed_features['cont_long_word_count'] = '%.2f'%(0.00)
    computed_features['cont_diff_word_count'] = '%.2f'%(0.00)
    computed_features['cont_TTR'] = '%.2f'%(0.00)
    computed_features['cont_allcaps_word_count'] = '%.2f'%(0.00)
    computed_features['cont_func_word'] = '%.2f'%(0.00)
    computed_features['cont_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_pers_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_fps_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_fpp_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_sec_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_tps_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_tpp_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_impersonal_pronoun'] = '%.2f'%(0.00)
    computed_features['cont_article'] = '%.2f'%(0.00)
    computed_features['cont_prepositions'] = '%.2f'%(0.00)
    computed_features['cont_aux_verbs'] = '%.2f'%(0.00)
    computed_features['cont_common_adverbs'] = '%.2f'%(0.00)
    computed_features['cont_conjunctions'] = '%.2f'%(0.00)
    computed_features['cont_negations'] = '%.2f'%(0.00)
    computed_features['cont_common_verbs'] = '%.2f'%(0.00)
    computed_features['cont_common_adjectives'] = '%.2f'%(0.00)
    computed_features['cont_comparisons'] = '%.2f'%(0.00)
    computed_features['cont_interrogatives'] = '%.2f'%(0.00)
    computed_features['cont_concrete_figures'] = '%.2f'%(0.00)
    computed_features['cont_quantifiers'] = '%.2f'%(0.00)
    computed_features['cont_affect_process'] = '%.2f'%(0.00)
    computed_features['cont_pos_emotion'] = '%.2f'%(0.00)
    computed_features['cont_achievement'] = '%.2f'%(0.00)
    computed_features['cont_neg_emotion'] = '%.2f'%(0.00)
    computed_features['cont_anxiety'] = '%.2f'%(0.00)
    computed_features['cont_anger'] = '%.2f'%(0.00)
    computed_features['cont_sadness'] = '%.2f'%(0.00)
    computed_features['cont_cognitive_process'] = '%.2f'%(0.00)
    computed_features['cont_insight'] = '%.2f'%(0.00)
    computed_features['cont_causation'] = '%.2f'%(0.00)
    computed_features['cont_discrepancy'] = '%.2f'%(0.00)
    computed_features['cont_tentative'] = '%.2f'%(0.00)
    computed_features['cont_fillers'] = '%.2f'%(0.00)
    computed_features['cont_certainty'] = '%.2f'%(0.00)
    computed_features['cont_differentiation'] = '%.2f'%(0.00)
    computed_features['cont_informal_language'] = '%.2f'%(0.00)
    computed_features['cont_swear_words'] = '%.2f'%(0.00)
    computed_features['cont_internet_slang'] = '%.2f'%(0.00)
    computed_features['cont_sexual_words'] = '%.2f'%(0.00)
    computed_features['cont_nonfluencies'] = '%.2f'%(0.00)
    computed_features['cont_past_focus'] = '%.2f'%(0.00)
    computed_features['cont_present_focus'] = '%.2f'%(0.00)
    computed_features['cont_future_focus'] = '%.2f'%(0.00)
    computed_features['cont_punctuations'] = '%.2f'%(0.00)
    computed_features['cont_periods'] = '%.2f'%(0.00)
    computed_features['cont_commas'] = '%.2f'%(0.00)
    computed_features['cont_colons'] = '%.2f'%(0.00)
    computed_features['cont_semicolons'] = '%.2f'%(0.00)
    computed_features['cont_question_marks'] = '%.2f'%(0.00)
    computed_features['cont_exclam_marks'] = '%.2f'%(0.00)
    computed_features['cont_dashes'] = '%.2f'%(0.00)
    computed_features['cont_apostrophes'] = '%.2f'%(0.00)
    computed_features['cont_parentheses'] = '%.2f'%(0.00)
    computed_features['cont_other_puncts'] = '%.2f'%(0.00)

  return computed_features

def process_link(article_url):
  head_and_cont = {}

  config = Config()
  config.headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.2 '
                                  '(KHTML, like Gecko) Chrome/15.0.874.121 Safari/535.2', }

  article = Article(article_url.strip(), config=config)

  try:
    article.download()
    article.parse()
    head_and_cont['newstitle'] = article.title
    head_and_cont['newsbody'] = article.text
    head_and_cont['newsurl']  = article.url
  except Exception as e:
    print(e)
    print("Article Download Failed :( ")

  return head_and_cont

def process_news(input1,input2,hoc_flag):
  metadata = {}

  if hoc_flag == 'B':
    clean_headline = clean_data(input1, '', '', '')
    clean_content = clean_data(input2, input1, '', 'C')
    metadata['headline'] = clean_headline
    metadata['content'] = clean_content
    metadata['computed_features'] = compute_features(clean_headline, clean_content)
    metadata['prediction_and_probability'] = predict_news(metadata['computed_features'],'B')
  elif hoc_flag == 'H':
    clean_headline = clean_data(input1, '', '', '')
    metadata['headline'] = clean_headline
    metadata['content'] = ''
    metadata['computed_features'] = compute_features(clean_headline, '')
    metadata['prediction_and_probability'] = predict_news(metadata['computed_features'],'H')
  elif hoc_flag == 'C':
    clean_content = clean_data(input1, '', '', '')
    metadata['headline'] = ''
    metadata['content'] = clean_content
    metadata['computed_features'] = compute_features('', clean_content)
    metadata['prediction_and_probability'] = predict_news(metadata['computed_features'],'C')
  elif hoc_flag == 'U':
    head_and_cont = process_link(input1)
    clean_headline = clean_data(head_and_cont['newstitle'], '','','')
    clean_content = clean_data(head_and_cont['newsbody'],head_and_cont['newstitle'],head_and_cont['newsurl'],'C')
    #clean_content = clean_data(head_and_cont['newsbody'], head_and_cont['newstitle'], '', 'C')
    metadata['headline'] = clean_headline
    metadata['content'] = clean_content
    metadata['computed_features'] = compute_features(clean_headline, clean_content)
    metadata['prediction_and_probability'] = predict_news(metadata['computed_features'], 'B')

  return metadata