from snorkel.labeling import labeling_function, LabelingFunction
from snorkel.labeling.lf.nlp import nlp_labeling_function
from SpamPreprocessors import *
import re

#labels
ABSTAIN = -1
HAM = 0
SPAM = 1

@labeling_function()
def check(x:str)->int:
    """
    Labeling function which classifies a piece of text as spam if it contains
    the word "check" in it.
    :param x: Text to evaluate
    :return: 1 if the word "check" is in the text else -1
    """

    return SPAM if 'check' in x.text.lower() else ABSTAIN

@labeling_function()
def check_out(x:str)->int:
    """
    Labeling function which classifies a piece of text as spam if it contains
    the phrase "check out" in it.
    :param x: Text to evaluate
    :return: 1 if the phrase "check out" is in the text else -1
    """
    return SPAM if "check out" in x.text.lower() else ABSTAIN

@labeling_function()
def regex_check_out(x:str)->int:
    """
    Labeling function which classifies a piece of text as spam if it contains
    the the words "check" and "out" with any set of characters in between.
    :param x: Text to evaluate
    :return: 1 if the condition is met else -1
    """
    return SPAM if re.search(r"check.*out", x.text, flags=re.I) else ABSTAIN

@labeling_function(pre=[textblob_sentiment])
def textblob_polarity(x:str, pol_thresh:float=0.9) -> int:
    """
    Identify if the text is not SPAM based on the subjectivity sentiment score
    :param x: Text to evaluate
    :param pol_thresh: Threshold for the polarity score to classify as HAM
    :return: 0 if the polarity of x is greater than the threshold
    """

    return HAM if x.polarity >= pol_thresh else ABSTAIN

@labeling_function(pre=[textblob_sentiment])
def textblob_subjectivity(x:str, subj_thresh:float=0.5) -> int:
    """
    Identify if the text is not SPAM based on the subjectivity sentiment score
    :param x: Text to evaluate
    :param subj_thresh: Threshold for the subjectivity score to classify as HAM
    :return: 0 if the subjectivity of x is greater than the threshold
    """

    return HAM if x.subjectivity >= subj_thresh else ABSTAIN

# Keyword LFs
def keyword_lookup(x:str, keywords:list, label:str) -> int:
    """
    Keyword template function that will return a label or ABSTAIN if the keyword
    is found in the text
    :param x: Text to evaluate
    :param keywords: Keywords that may or may not be present in the text
    :param label: Label given if a keyword is found in the text
    :return: Label or ABSTAIN if the keyword is not found
    """
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords:list, label:int=SPAM) -> LabelingFunction:
    """
    Function to create labeling functions based on different keywords
    :param keywords: List of keywords to evaluate
    :param label: Label used if the keyword is present
    :return: If a keyword is present, then return this label
    """

    return LabelingFunction(
        name=f'keyword_{keywords[0]}',
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label)
    )

"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments post links to other channels."""
keyword_link = make_keyword_lf(keywords=["http"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song"], label=HAM)

@labeling_function()
def short_comment(x:str, thresh:int=5)->int:
    """
    Short comments are often not SPAM (aka HAM).
    :param x: Text to evaluate
    :param thresh:Threshold where any text length below the threshold is HAM
    :return: 1 if text is less than threshold, otherwise return -1
    """
    return HAM if len(x.text.split())<=thresh else ABSTAIN

@nlp_labeling_function()
def has_person(x:str, thresh:int=20)->int:
    """
    Function to identify if the comment is short and includes a person which
    implies it is a HAM comment
    :param x: Text to evaluate
    :param thresh: Threshold to define a short comment
    :return: 1 if the comment length is less than the threshold and contains
    a person, else return -1
    """

    if len(x.doc) < thresh and any([ent.label_ == 'PERSON' for ent in x.doc.ents]):
        return HAM
    return ABSTAIN