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
            