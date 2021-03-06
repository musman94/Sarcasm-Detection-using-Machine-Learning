'''
@rana
naive bayes features - boolean values
'''

import nltk
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from textblob.blob import TextBlob

lemmatizer = WordNetLemmatizer()

punctuations = set(string.punctuation)

happy = set([':-)', ':)', '(:', '=)', ':-d', ':d', '=d', 'xd',  'd=', 'd;', 'd:', 'dx', ';-)', ';d', ';)', ':-p', ':p', 'xp', 'x-p', '=p', '>:-)', 
        '>:)', 'o:-)', 'o:)', '0:-)', '0:)', 'b)',
            'bd', 'b-)', 'b-d', '( ͡° ͜ʖ ͡°)'])

sad = set([':-(', ':(', '):', '=(', '(´･ω･`)', '(`･ω･´)', '(･ω･)'])

cute = set([":3", "=3"])
                        
surprised = set([':o', ':-o'])

love = set([':*', ':-*'])

indifferent = set([':-/', ':/', '=/',
             ':-\ ', ':\ ', '=\ ', ':l', '=l', ':-|', ':|', '-_-', 
             '¯\_(ツ)_/¯'])

#  list of intensifiers that could be used in sarcastic text
intensifiers = set(['already', 'bleeping', 'bloody', 'dafuq', 'damn', 'dickens', 'downright', 'effing', 'ever',
                'everliving', 'everloving', 'flipping', 'freaking', 'fricking', 'frigging', 'fuck', 'fucking', 'hell',
                'hella', 'holy', 'motherfreaking', 'motherfucking', 'on earth', 'precious', 'heck', 'well'])

#  list of interjections that could be used in sarcastic text
interjections = set(['absolutely', 'ahh', 'aha', 'ahem', 'ahoy', 'agreed', 'alas', 'alright', 'alrighty', 'amen', 'anytime',
                 'argh', 'anyhow', 'as if', 'attaboy', 'attagirl', 'aww', 'bam', 'behold', 'bingo', 'blah', 'bless you',
                 'bravo', 'cheers', 'darn', 'dang', 'doh', 'duh', 'eh', 'gee', 'geepers', 'golly', 'goodness',
                 'goodness gracious', 'gosh', 'ha', 'hallelujah', 'hey', 'hmmm', 'huh', 'indeed', 'jeez', 'my gosh',
                 'no', 'now', 'nah', 'oops', 'ouch', 'phew', 'please', 'shoot', 'shucks', 'there', 'uggh', 'waa',
                 'what', 'woah', 'woops', 'wow', 'yay', 'yes', 'yikes'])

def replace(comment):
        new_comment = comment
        if type(comment) is str:
                for em in happy:
                        new_comment = new_comment.replace(em, "happy")
                for em in sad:
                        new_comment = new_comment.replace(em, "sad")
                for em in cute:
                        new_comment = new_comment.replace(em, "cute")
                for em in surprised:
                        new_comment = new_comment.replace(em, "surprised")
                for em in love:
                        new_comment = new_comment.replace(em, "love")
                for em in indifferent:
                        new_comment = new_comment.replace(em, "indifferent")
        return new_comment


def get_exclaimation_count(features, comment):
        if type(comment) is str:
                if comment.count('!') is 0:
                        features['exclaimation'] = False
                else:
                        features['exclaimation'] = True
        else:
                features['exclaimation'] = False

def get_capital_count(features, comment):
        count = 0
        if type(comment) is str:
                for word in comment:
                        if word.isupper():
                                count += 1
        if count is 0:
                features['capital'] = False
        else:
                features['capital'] = True

def get_intensifier_count(features, comment):
        count = 0
        if type(comment) is str:
                for word in comment:
                        if word not in punctuations:
                                count += 1
        if count is 0:
                features['intensifiers'] = False
        else:
                features['intensifiers'] = True
        

def get_injection_count(features, comment):
        count = 0
        if type(comment) is str:
                for word in comment:
                        if word not in punctuations:
                                count += 1
        if count is 0:
                features['interjections'] = False
        else:
                features['interjections'] = True

def get_emoji_count(features, comment):
        count = 0
        if type(comment) is str:
                for word in comment:
                        if word in happy or word in sad or word in cute or word in surprised or word in love or word in indifferent:
                                count += 1
        if count is 0:
                features['emoticons'] = False
        else:
                features['emoticons'] = True

def get_pos(features, comment):
        pos_dic = {'NN': 0, 'JJ': 0, 'VB': 0, 'RB' : 0}
        if type(comment) is str:
                tokens = nltk.word_tokenize(comment)
                tokens = [token.lower() for token in tokens]
                pos = nltk.pos_tag(tokens)
                for x in range(len(pos)):
                        tag = pos[x][1]
                        if tag[0:2] == 'NN':
                                pos_dic['NN'] += 1
                        elif tag[0:2] == 'JJ':
                                pos_dic['JJ'] += 1
                        elif tag[0:2] == 'VB':
                                pos_dic['VB'] += 1
                        elif tag[0:2] == 'RB':
                                pos_dic['RB'] += 1
        for key, value in pos_dic.items():
                if value is 0:
                        features['POS ' + key] = False
                else:
                        features['POS ' + key] = True

def get_subjectivity_score(feat, text):
        '''
        0 = very objective
        1 = very subjective
        '''
        if type(text) is str:
                try:
                        blob = TextBlob(text.strip())
                        score = blob.sentiment.subjectivity
                        if score is 0.5:
                                feat['subjectivity'] = False
                                feat['objectivity'] = False
                        elif score < 0.5:
                                feat['subjectivity'] = False
                                feat['objectivity'] = True
                        else:
                                feat['subjectivity'] = True
                                feat['objectivity'] = False
                except:
                        feat['subjectivity'] = False
                        feat['objectivity'] = False
        else:
                feat['subjectivity'] = False
                feat['objectivity'] = False

def get_polarity_score(feat, text):
        '''
        -1 = very negative
        1 = very positive
        '''
        if type(text) is str:
                try:
                        blob = TextBlob(text.strip())
                        score = blob.sentiment.polarity
                        if score is 0:
                                feat['positivity'] = False
                                feat['negativity'] = False
                        elif score < 0:
                                feat['positivity'] = False
                                feat['negativity'] = True
                        else:
                                feat['positivity'] = True
                                feat['negativity'] = False
                except:
                        feat['positivity'] = False
                        feat['negativity'] = False
        else:
                feat['positivity'] = False
                feat['negativity'] = False

def get_average_contrast(feat, text):
        '''
        0 = no contrast
        1 = high contrast
        '''
        negCount = 0
        posCount = 0
        negTotal = 0.0
        posTotal = 0.0
        polarityTemp = 0.0
        polarityDif = 0.0
        if type(text) is str:
                try:
                        blob = TextBlob(text.strip())
                        for sentence in blob.sentences:
                                polarityTemp = sentence.sentiment.polarity
                                if polarityTemp < 0:
                                        negTotal += polarityTemp
                                        negCount += 1
                                elif polarityTemp > 0:
                                        posTotal += polarityTemp
                                        posCount += 1
                        if negCount > 0:
                                if posCount > 0:
                                        polarityDif = ((posTotal / posCount) - (negTotal / negCount)) / 2
                        if polarityDif < 0.5:
                                feat['average_contrast'] = False
                        else:
                                feat['average_contrast'] = True
                except:
                        feat['average_contrast'] = False
        else:
                feat['average_contrast'] = False

def get_extreme_contrast(feat, text):
        '''
        0 = no contrast
        1 = high contrast
        '''
        minPolarity = 0.0
        maxPolarity = 0.0
        polarityTemp = 0.0
        if type(text) is str:
                try:
                        blob = TextBlob(text.strip())
                        for sentence in blob.sentences:
                                polarityTemp = sentence.sentiment.polarity
                                if polarityTemp > maxPolarity:
                                        maxPolarity = polarityTemp
                                elif polarityTemp < minPolarity:
                                        minPolarity = polarityTemp
                        score = (maxPolarity - minPolarity)/2
                        if score < 0.5:
                                feat['extreme_contrast'] = False
                        else:
                                feat['extreme_contrast'] = True
                except:
                        feat['extreme_contrast'] = False
        else:
                feat['extreme_contrast'] = False

def get_half_contrast(feat, text):
        '''
        0 = no difference in polarity
        1 = high difference in polarity
        '''
        first_half_polarity = 0.0
        second_half_polarity = 0.0
        if type(text) is str:
                tokens = nltk.word_tokenize(text, language = 'english', preserve_line = False)
                if len(tokens) == 1:
                        feat['half_contrast'] = False
                else:
                        first_half = tokens[0: int(len(tokens)/2)]
                        second_half = tokens[int(len(tokens)/2):]
                        
                        try:
                                blob = TextBlob("".join([" "+i if i not in string.punctuation else i for i in first_half]).strip())
                                first_half_polarity = blob.sentiment.polarity
                        except:
                                first_half_polarity = 0.0
                        
                        try:
                                blob = TextBlob("".join([" "+i if i not in string.punctuation else i for i in second_half]).strip())
                                second_half_polarity = blob.sentiment.polarity
                        except:
                                second_half_polarity = 0.0
                        score = np.abs(first_half_polarity - second_half_polarity) / 2
                        if score < 0.5:
                                feat['half_contrast'] = False
                        else:
                                feat['half_contrast'] = True
        else:
                feat['half_contrast'] = False

def get_third_contrast(feat, text):
        '''
        0 = no difference in polarity
        1 = high difference in polarity
        '''
        first_half_polarity = 0.0
        second_half_polarity = 0.0
        third_half_polarity = 0.0
        if type(text) is str:
                tokens = nltk.word_tokenize(text, language = 'english', preserve_line = False)
                if len(tokens) < 2:
                        feat['third_contrast_12'] = False
                        feat['third_contrast_13'] = False
                        feat['third_contrast_23'] = False
                elif len(tokens) == 2:
                        try:
                                blob = TextBlob(tokens[0])
                                first_half_polarity = blob.sentiment.polarity
                        except:
                                first_half_polarity = 0.0
                        try:
                                blob = TextBlob(tokens[1])
                                second_half_polarity = blob.sentiment.polarity
                        except:
                                second_half_polarity = 0.0
                        feat['third_contrast_13'] = False
                        feat['third_contrast_23'] = False
                        score = np.abs(first_half_polarity - second_half_polarity) / 2
                        if score < 0.5:
                                feat['third_contrast_12'] = False
                        else:
                                feat['third_contrast_12'] = True
                else:
                        first_half = tokens[0: int(len(tokens)/3)]
                        second_half = tokens[int(len(tokens)/3): 2*int(len(tokens)/3)]
                        third_half = tokens[2*int(len(tokens)/3):]
                        try:
                                blob = TextBlob(first_half)
                                first_half_polarity = blob.sentiment.polarity
                        except:
                                first_half_polarity = 0.0
                        try:
                                blob = TextBlob(second_half)
                                second_half_polarity = blob.sentiment.polarity
                        except:
                                second_half_polarity = 0.0
                        try:
                                blob = TextBlob(third_half)
                                third_half_polarity = blob.sentiment.polarity
                        except:
                                third_half_polarity = 0.0
                        score1 = np.abs(first_half_polarity - second_half_polarity) / 2
                        score2 = np.abs(first_half_polarity - third_half_polarity) / 2
                        score3 = np.abs(second_half_polarity - third_half_polarity) / 2
                        if score1 < 0.5:
                                feat['third_contrast_12'] = False
                        else:
                                feat['third_contrast_12'] = True
                        if score2 < 0.5:
                                feat['third_contrast_13'] = False
                        else:
                                feat['third_contrast_13'] = True
                        if score3 < 0.5:
                                feat['third_contrast_23'] = False
                        else:
                                feat['third_contrast_23'] = True
        else:
                feat['third_contrast_12'] = False
                feat['third_contrast_13'] = False
                feat['third_contrast_23'] = False


def get_features(features, comment):
        get_exclaimation_count(features, comment)
        get_capital_count(features, comment)
        get_intensifier_count(features, comment)
        get_injection_count(features, comment)
        get_emoji_count(features, comment)
        comment = replace(comment)
        get_pos(features, comment)
        get_subjectivity_score(features, comment)
        get_polarity_score(features, comment)
        get_average_contrast(features, comment)
        get_extreme_contrast(features, comment)
        get_half_contrast(features, comment)
        get_third_contrast(features, comment)

        

