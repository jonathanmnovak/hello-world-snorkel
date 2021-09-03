import re
from snorkel.preprocess import preprocessor
from snorkel.slicing import SlicingFunction, slicing_function
from SpamPreprocessors import *

@slicing_function()
def short_comment(x:str, thresh:int=5) -> bool:
    """
    Slicing function to identify Ham comments which are often short text
    :param x: Comment to evaluate
    :param thresh: Threshold text length to define short or not
    :return: Boolean indicating if the comment is a short-length or not
    """

    return len(x.text.split()) < thresh

def keyword_lookup(x:str, keywords:list) -> bool:
    """
    Keyword-based slicing functions
    :param x: Comment to evaluate
    :param keywords: Keywords to consider
    :return: Boolean if any of the keywords are found in the comment
    """
    return any(word in x.text.lower() for word in keywords)

def make_keyword_sf(keywords:list) -> SlicingFunction:
    """
    Function to create slicing function given a set of keywords
    :param keywords: Keywords unique to this slicing function
    :return: Slicing function with the keywords
    """
    return SlicingFunction(
        name=f"keyword_{keywords[0]}",
        f=keyword_lookup,
        resources=dict(keywords=keywords),
    )

"""Create please keyword slicing function"""
keyword_please = make_keyword_sf(keywords=["please", "plz"])


@slicing_function()
def regex_check_out(x:str)->bool:
    """
    Slicing function to check if the phrase 'check out' with words in between
    are in the comment
    :param x: Comment to evaluate
    :return: Boolean if the regex of 'check out' is in the comment
    """
    return bool(re.search(r"check.*out", x.text, flags=re.I))


@slicing_function()
def short_link(x:str)->bool:
    """Returns whether text matches common pattern for shortened ".ly" links.
    :param x: Comment to evaluate
    :return: Boolean if the regex of '.ly' is in the comment
    """
    return bool(re.search(r"\w+\.ly", x.text))

@slicing_function(pre=[textblob_sentiment])
def textblob_polarity(x:str, thresh:float=0.9) -> bool:
    """
    Slciing
    :param x: Comment to evaluate
    :return: Boolean if the polarity is above the given threshold
    """
    return x.polarity > thresh