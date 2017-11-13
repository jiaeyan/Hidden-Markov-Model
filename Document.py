class Document():
  '''This is a document class for HMM instance, with feature list and label list as atrributes.'''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def modify_f(self):
      '''Implement DIY method to modigfy feature set.'''
        self.features = map(str.lower, self.features)
