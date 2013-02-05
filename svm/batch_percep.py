import sys
from multitron import MultitronParameters

class Problem:
   def __init__(self):
      self.data = []
      self.params = None

   def AddInstance(self, labels, features):
      self.data.append((map(int,labels), features))

   def Train(self, numiters = 10, model_fname = "percep_model"):
      labels = set()
      for ls, fs in self.data:
         labels.update(ls)
      nlabels = max(labels) + 1
      self.params = MultitronParameters(nlabels)
      for iter in xrange(numiters):
         print >> sys.stderr,"iteration",iter
         for labels, fs in self.data:
            pred = params.predict_best_class(features)
            params.tick()
            if pred not in labels:
               scores = params.get_scores(features)
               goods = ((scores[l], l) for l in labels)
               higest_score, best_label = max(goods)
               params.add(fs, pred, -1.0)
               params.add(fs, best_label, 1.0)

   def Save(self, model_fname):
      self.params.finalize()
      self.params.dump(file(model_fname, "w"))

   def TrainFromFile(self, fname, numiters = 10, model_fname = "percep_model"):
      labelset = set()
      for line in file(fname):
         labels, features = line.strip().split(None, 1)
         labels = [int(l) for l in labels.split(",")]
         labelset.update(labels)
      nlabels = max(labelset) + 1

      params = MultitronParameters(nlabels)
      for iter in xrange(numiters):
         print >> sys.stderr,"iteration",iter
         for line in file(fname):
            labels, fs = line.strip().split(None, 1)
            fs = fs.split()
            labels = [int(l) for l in labels.split(",")]
            pred = params.predict_best_class(fs)
            params.tick()
            if pred not in labels:
               scores = params.get_scores(features)
               goods = ((scores[l], l) for l in labels)
               higest_score, best_label = max(goods)
               params.add(fs, pred, -1.0)
               params.add(fs, best_label, 1.0)
      params.finalize()
      params.dump(file(model_fname, "w"))

if __name__ == '__main__':
   p = Problem()
   p.TrainFromFile(sys.argv[1], 10, sys.argv[2])

