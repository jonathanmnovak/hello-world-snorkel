from snorkel.preprocess import preprocessor
from textblob import TextBlob

@preprocessor(memoize=True)
def textblob_sentiment(x:str):
    """
    Uses the TextBlob package to apply a sentiment score to a given text.
    :param x: Text to evaluate
    :return: Text but with sentiment features (polarity and subjectivity)
    """

    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    return x