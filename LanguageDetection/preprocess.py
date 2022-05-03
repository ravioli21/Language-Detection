import re, string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class NLP_Preprocess():
        def __init__(self):
                self.cv = CountVectorizer()
                pass
        
        def removeSymbolsAndNumbers(self, text): # removes punctuations and numbers
                text = re.sub(r'[{}]'.format(string.punctuation), '', text)
                text = re.sub(r'\d+', '', text)
                text = re.sub(r'[@]', '', text)
                return text.lower()
                
        def removeEnglishLetters(self, text): # removes english letters from languages not containing english alphabets
                text = re.sub(r'[a-zA-Z]+', '', text)
                return text.lower()

        def vectorize(self, corpus):
                X = self.cv.fit_transform(corpus).toarray()
                return X
        
        def preprocess(self, X, y=None):
                corpus = []
                for i in range(len(X)):
                        text = self.removeSymbolsAndNumbers(X[i])
                        if(y is not None):
                                if(y[i] in ['Russian','Malyalam','Hindi','Kannada','Tamil','Arabic']):
                                        text = self.removeEnglishLetters(text)
                        corpus.append(text)
                        
                X = self.vectorize(corpus)
                return X