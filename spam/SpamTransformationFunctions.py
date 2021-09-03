"""Transformation functions for Spam data."""
import names
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
from snorkel.augmentation import transformation_function
from snorkel.augmentation.tf import LambdaTransformationFunction
from snorkel.map.core import LambdaMapper
from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field='text', doc_field='doc', memoize=True)
replacement_names = [names.get_full_name() for _ in range(50)]
nltk.download("wordnet")
pos_dict = {'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a'}


def _get_synonym(word: str, pos: str = None) -> str:
    """Get synonym for word given its part-of-speech (pos).

    :param word: Word to find a synonym for based on the pos
    :param pos: Part-of-speech
    :return: synonym
    """
    synsets = wn.synsets(word, pos=pos)

    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():
            return words[0].replace("_", " ")


def _replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement.

    :param spacy_doc: Spacy text
    :param idx: index of spacy_doc
    :param replacement: replacement text
    :return: Text with the word replaced
    """
    return " ".join(
        [spacy_doc[:idx].text, replacement, spacy_doc[1 + idx:].text])


@transformation_function(pre=[spacy])
def change_person(x: str) -> str:
    """Replace the name in a text with a different, randomly chosen name.

    :param x: Text to evaluate
    :return: Text with a different name
    """
    person_names = [ent.text for ent in x.doc.ents if ent.label_ == "PERSON"]

    if person_names:
        name_to_replace = np.random.choice(person_names)
        replacement_name = np.random.choice(replacement_names)
        x.text = x.text.replace(name_to_replace, replacement_name)
        return x


@transformation_function(pre=[spacy])
def swap_adjectives(x: str) -> str:
    """Swap two adjectives at random.

    :param x: Text to evaluate
    :return: Text with a different adjective
    """
    adjective_idxs = [i for i, token in enumerate(x.doc) if
                      token.pos_ == "ADJ"]
    # Check that there are at least two adjectives to swap
    if len(adjective_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(adjective_idxs, 2, replace=False))

        x.text = " ".join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1:idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2:].text,
            ]
        )
        return x


def _replace_pos_with_synonym(x: str, pos: str) -> str:
    """Replace a word (given a part-of-speech) with a random synonym.

    :param x: Text to evaluate
    :param pos: part-of-speech
    :return: Text with  replaced with synonym
    """
    idxs = [i for i, token in enumerate(x.doc) if token.pos_ == pos]

    if idxs:
        idx = np.random.choice(idxs)
        synonym = _get_synonym(x.doc[idx].text, pos=pos_dict[pos])
        if synonym:
            x.text = _replace_token(x.doc, idx, synonym)
            return x


@transformation_function(pre=[spacy])
def replace_noun_with_synonym(x: str) -> str:
    """Replace a noun in the text with a random synonym.

    :param x: Text to evaluate
    :return: Text with synonym replacing a randomly chosen noun
    """
    return _replace_pos_with_synonym(x, pos="NOUN")


@transformation_function(pre=[spacy])
def replace_verb_with_synonym(x: str) -> str:
    """Replace a verb in the text with a random synonym.

    :param x: Text to evaluate
    :return: Text with synonym replacing a randomly chosen verb
    """
    return _replace_pos_with_synonym(x, pos="VERB")


@transformation_function(pre=[spacy])
def replace_adj_with_synonym(x: str) -> str:
    """Replace an adjective in the text with a random synonym.

    :param x: Text to evaluate
    :return: Text with synonym replacing a randomly chosen adjective
    """
    return _replace_pos_with_synonym(x, pos="ADJ")
