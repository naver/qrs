# qrs
# Copyright (c) 2022-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

from . import BooleanFeature
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

male_pronouns = ['he', 'him', 'himself']
female_pronouns = ['she', 'her', 'herself']

class GenderFeature(BooleanFeature):
    def __init__(self, gender):
        super(GenderFeature, self).__init__(self.get_gender_predicate(gender), gender)

    def get_gender_predicate(self, gender):
        assert gender in ['male', 'female']
        def predicate(s):
            s = nltk.word_tokenize(s)
            male_count = sum(s.count(p) for p in male_pronouns)
            female_count = sum(s.count(p) for p in female_pronouns)
            if gender == 'female':
                return female_count > male_count
            else:
                return male_count > female_count
        return predicate
