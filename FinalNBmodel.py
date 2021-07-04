#Building the Naive Bayes model
import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # calculate prior for each class
        self.priors =  np.zeros(n_classes, dtype=np.float64)
        for c in self.classes:
            self.priors[c] = np.count_nonzero(y==c)/n_samples
         
        #claculating likelihood
        self.likelihood = np.zeros((n_features, n_classes),dtype=np.float64)
        for c in self.classes:
            denominator = np.count_nonzero(y==c)
            for f in range(0,n_features):
                numerator = np.sum(X[i][f] for i in range(0,n_samples) if y[i]==c)                  
                self.likelihood[f][c] = (numerator+ 1)/(denominator + n_features)
            
        
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for c in self.classes:
            prior = np.log(self.priors[c])
            posterior = 0.0
            for idx,n in enumerate(x):
                posterior += n*np.log(self.likelihood[idx][c])
            
            posterior = prior + posterior
            posteriors.append(posterior)
            
        # return class with highest posterior probability
        return self.classes[np.argmax(posteriors)]
#--------------------------------------------------------------------------------------------

#Implementing the model

import pandas as pd

#importing the dataset
dataset = pd.read_csv("a1_d3.txt", delimiter = '\t', quoting = 3, header = None)

#Cleaning the text
import re
stop_words = pd.read_csv("stopwords.txt")
corpus = [] # will have the cleaned reviews

for i in range(0,1000):
 review = re.sub('[^a-zA-Z]', ' ', dataset[0][i]) # text pre-processing
 review = review.lower()
 review = review.split() # spilits the string to a list
 review = [word for word in review if word not in stop_words]
 review = ' '.join(review) # join the list into a string saperated by a space
 corpus.append(review)

#Building vocabulary
word_list = [] # list of all words from corpus(vocabulary)
for review in corpus:
    words_in_review = review.split()
    word_list.extend(words_in_review)# adding the list of words to the vocabulary
word_list = sorted(list(set(word_list)))

#Building the Bag of Words model
bag = np.zeros((len(corpus),len(word_list))) # creating the sparce matrix
review_index = 0
for review in corpus:
    for review_word in review.split():
        for i,vocab_word in enumerate(word_list):
            if vocab_word == review_word:
                bag[review_index][i] +=1
    review_index+=1
bag = pd.DataFrame(bag)

#data for the model
X = bag.iloc[:, :].values
y = dataset.iloc[:, -1].values

#5 fold splitting along with accuracy and fscore measurements
accuracy_scores = []
f1_scores = []

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

for train_index, test_index in kf.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    classifier = NaiveBayes()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
# calculating the accuracy and F-score for each split   
    cm = pd.crosstab(y_test, y_pred)
    cm= cm.to_numpy()
    accuracy = np.sum(np.diag(cm))/ np.sum(cm)
    recall = cm[0][0] / (cm[0][0]+cm[1][0])
    precision = cm[0][0] / (cm[0][0]+cm[0][1])
    fscore = 2*recall*precision/(recall+precision)
    accuracy_scores.append(accuracy)
    f1_scores.append(fscore)

#calculating mean and std deviation of accuracy and F-score    
accuracy_mean = sum(accuracy_scores) / len(accuracy_scores) 
variance = sum([((x - accuracy_mean) ** 2) for x in accuracy_scores]) / len(accuracy_scores) 
accuracy_std = variance ** 0.5

fscore_mean = sum(f1_scores) / len(f1_scores) 
variance = sum([((x - fscore_mean) ** 2) for x in f1_scores]) / len(f1_scores) 
fscore_std = variance ** 0.5

print("Accuracy: %0.3f +/- %0.3f" % (accuracy_mean, accuracy_std))
print("F-score: %0.3f +/- %0.3f" % (fscore_mean, fscore_std))           