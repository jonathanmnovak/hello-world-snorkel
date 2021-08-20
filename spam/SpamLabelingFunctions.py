from snorkel.labeling import labeling_function
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