from gensim import corpora, models
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string

class topic(object):
	def __init__(self, numTopics = 100, alpha = 1, model = None, dictionary = None):
		self.numTopics = numTopics
		self.alpha = alpha
		self.stop = set(stopwords.words('english'))
		self.punctuations = set(string.punctuation)
		self.porter = nltk.PorterStemmer()
		if model != None and dictionary != None:
			self.model = models.ldamodel.LdaModel.load(model)
			self.dictionary = corpora.Dictionary.load(dictionary)
    
	def generate(self, comments):
		tokens = [nltk.word_tokenize(comment) for comment in comments]
		tokens = [[self.porter.stem(t.lower()) for t in token if t.lower() not in self.stop and t not in self.punctuations] for token in tokens]

		self.dictionary = corpora.Dictionary(tokens)
		corpus = [self.dictionary.doc2bow(token) for token in tokens]
		self.model = models.ldamodel.LdaModel(corpus, id2word = self.dictionary, num_topics = self.numTopics, alpha = self.alpha)
		
		self.model.save("topics.model")
		self.dictionary.save("dictionary.model")

	def get_topics(self, comment):
		tokens = nltk.word_tokenize(comment)
		tokens = [self.porter.stem(t.lower()) for t in tokens if t.lower() not in self.stop and t not in self.punctuations]
		corpus_comment = self.dictionary.doc2bow(tokens)
		return self.model[corpus_comment]
		