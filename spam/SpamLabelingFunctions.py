from snorkel.labeling import labeling_function
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