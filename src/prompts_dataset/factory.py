import itertools
from typing import List

from nltk.corpus import cmudict


def permutation(words: List[str], n: int):
    """
    'a','b','c', n = 2
    ('a', 'b'), ('a', 'c'), ('b', 'a')
    ('b', 'c'), ('c', 'a'), ('c', 'b')
    all permutation of words
    each words will be present only once
    """
    return itertools.permutations(words, n)


def all_combinations(words: List[str], n: int):
    """
    'a','b','c', n = 2
    ('a','a'), ('a','b'), ('a','c'),
    ('b','a'), ('b','b'), ('b','c'),
    ('c','a'), ('c','b'), ('c','c')
    all possible combinations of words
    each words can be present n times
    """
    return itertools.product(words, repeat=n)


def combinations(words: List[str], n: int):
    """
    'a','b','c', n = 2
    ('a','b'), ('a','c'), ('b','c')
    each words can be present once
    there is no mirror repetition i.e. (a,b) and (b,a) are same
    """
    return itertools.combinations(words, n)


def multi_combinations(words: List[List[str]]):
    """
    [['a','b'],['c','d'],['e','f']]
    ('a', 'c', 'e'), ('a', 'c', 'f'), ('a', 'd', 'e')
    ('a', 'd', 'f'), ('b', 'c', 'e'), ('b', 'c', 'f')
    ('b', 'd', 'e'),('b', 'd', 'f')
    """
    return itertools.product(*words)


def unique_words(iterable, unique: List[List[int]]):
    """
    return only those combination which have unique words
    i.e. no word is repeated according to unique
    iter = [('a', 'c', 'e'), ('a', 'c', 'f'), ('a', 'd', 'a'), ('c','c','c')]
    unique = [[0,1,2]]
    pos 0 and 2 must be unique
    return [('a', 'c', 'e'), ('a', 'c', 'f')]
    """
    for words in iterable:
        words_to_control = [[words[i] for i in position] for position in unique]
        for to_control in words_to_control:
            if len(set(to_control)) != len(to_control):
                break
        else:
            yield words


def starts_with_vowel_sound(word, pronunciations=cmudict.dict()):
    for syllables in pronunciations.get(word, []):
        return syllables[0][-1].isdigit()


def get_article(input_str):
    """
    Returns 'a' or 'an' depending on the first word of the input string.
    """
    input_str = input_str.lower()
    return "an" if starts_with_vowel_sound(input_str) else "a"
