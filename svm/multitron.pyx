import sys
import math

from stdlib cimport *
cdef class MulticlassParamData:
    def __cinit__(self, int nclasses):
        cdef int i
        self.lastUpd = <int *>malloc(nclasses*sizeof(int))
        self.acc     = <double *>malloc(nclasses*sizeof(double))
        self.w       = <double *>malloc(nclasses*sizeof(double))
        for i in range(nclasses):
            self.lastUpd[i]=0
            self.acc[i]=0
            self.w[i]=0

    def __dealloc__(self):
        free(self.lastUpd)
        free(self.acc)
        free(self.w)

cdef class MultitronParameters:
    
    def __cinit__(self, nclasses):
        self.nclasses = nclasses
        self.scores = <double *>malloc(nclasses*sizeof(double))

    cpdef set_labels(self, labels):
        assert len(labels) == self.nclasses
        self.labels = list(labels)
        self.label_to_i = dict([(label, i) for (i, label) in enumerate(labels)])

    cpdef getW(self, clas): 
        d={}
        cdef MulticlassParamData p
        for f,p in self.W.iteritems():
            d[f] = p.w[clas]
        return d

    def __init__(self, nclasses):
        self.nclasses = nclasses
        self.now = 0
        self.W = {}

    cdef _tick(self):
        self.now=self.now+1

    def tick(self): self._tick()

    cpdef scalar_multiply(self, double scalar):
        """
        note: DOES NOT support averaging
        """
        cdef MulticlassParamData p
        cdef int c
        for p in self.W.values():
            for c in xrange(self.nclasses):
                p.w[c]*=scalar

    cpdef add(self, list features, int clas, double amount):
        cdef MulticlassParamData p
        assert clas <= self.nclasses
        for f in features:
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p
            print 'upd'
            p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
            p.w[clas]+=amount
            p.lastUpd[clas]=self.now
            print 'done' 
        
    cpdef add_r(self, list features, int clas, double amount):
        """
        like "add", but with real values features: 
            each feature is a pair (f,v), where v is the value.
        """
        cdef MulticlassParamData p
        cdef double v
        cdef str f
        for f,v in features:
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p

            p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
            p.w[clas]+=amount*v
            p.lastUpd[clas]=self.now

    cpdef set(self, list features, int clas, double amount):
        """
        like "add", but replaces instead of adding
        """
        cdef MulticlassParamData p
        cdef double v
        cdef str f
        for f in features:
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p

            p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
            p.w[clas]+=amount
            p.lastUpd[clas]=self.now

    cpdef add_params(self, MultitronParameters other, double factor):
        """
        like "add", but with data from another MultitronParameters object.
        they must both share the number of classes
        add each value * factor
        """
        cdef MulticlassParamData p
        cdef MulticlassParamData op
        cdef double v
        cdef str f
        cdef int clas
        assert(self.nclasses==other.nclasses),"incompatible number of classes in add_params"
        for f,op in other.W.items():
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p

            for clas in xrange(self.nclasses):
                if op.w[clas]<0.0000001: continue
                p.acc[clas]+=(self.now-p.lastUpd[clas])*p.w[clas]
                #print p.w[clas], op.w[clas]
                p.w[clas]+=(op.w[clas]*factor)
                p.lastUpd[clas]=self.now

    cpdef do_pa_update(self, list feats, int gold_cls, double C=1.0):
        cdef double go_scr
        cdef double gu_scr
        cdef double loss
        cdef double norm
        cdef double tau
        cdef int prediction
        cdef dict scores
        self._tick()
        prediction = self._predict_best_class(feats)
        if prediction==gold_cls: return prediction
        scores = self.get_scores(feats)
        go_scr = scores[gold_cls]
        gu_scr = scores[prediction]

        loss = gu_scr - go_scr + 1
        norm = len(feats)+len(feats)
        tau = loss / norm
        if tau>C: tau=C
        self.add(feats,prediction,-tau)
        self.add(feats,gold_cls,+tau)
        return prediction

    cpdef pa_update(self, object gu_feats, object go_feats, int gu_cls, int go_cls,double C=1.0):
        cdef double go_scr
        cdef double gu_scr
        cdef double loss
        cdef double norm
        cdef double tau
        go_scr = self.get_scores(go_feats)[go_cls]
        gu_scr = self.get_scores(gu_feats)[gu_cls]
        loss = gu_scr - go_scr + 1
        norm = len(go_feats)+len(gu_feats)
        tau = loss / norm
        if tau>C: tau=C
        self.add(gu_feats,gu_cls,-tau)
        self.add(go_feats,go_cls,+tau)

    cpdef get_scores(self, features):
        cdef MulticlassParamData p
        cdef int i
        cdef double w
        for i in xrange(self.nclasses):
            self.scores[i]=0
        for f in features:
            try:
               p = self.W[f]
               for c in xrange(self.nclasses):
                   self.scores[c] += p.w[c]
            except KeyError: pass
        cdef double tot = 0
        res={}
        for i in xrange(self.nclasses):
            res[i] = self.scores[i]
        return res

    cpdef get_scores_r(self, features):
        """
        like get_scores but with real values features
            each feature is a pair (f,v), where v is the value.
        """
        cdef MulticlassParamData p
        cdef int i
        cdef double w
        cdef double v
        for i in xrange(self.nclasses):
            self.scores[i]=0
        for f,v in features:
            try:
                p = self.W[f]
                for c in xrange(self.nclasses):
                    self.scores[c] += p.w[c]*v
            except KeyError: pass
        cdef double tot = 0
        res={}
        for i in xrange(self.nclasses):
            res[i] = self.scores[i]
        return res

    def update(self, correct_class, features):
        """
        does a prediction, and a parameter update.
        return: the predicted class before the update.
        """
        self._tick()
        prediction = self._predict_best_class(features)
        if prediction != correct_class:
            self._update(correct_class, prediction, features)
        return prediction

    cpdef predict_best_class_r(self, list features):
        scores = self.get_scores_r(features)
        scores = [(s,c) for c,s in scores.iteritems()]
        s = max(scores)[1]
        return max(scores)[1]

    cpdef predict_best_class(self, list features):
        return self._predict_best_class(features)

    def update_r(self, correct_class, features):
        self._tick()
        prediction = self.predict_best_class_r(features)
        if prediction != correct_class:
            self._update_r(correct_class, prediction, features)
        return prediction

    cdef int _predict_best_class(self, list features):
        cdef int i
        cdef MulticlassParamData p
        for i in range(self.nclasses):
            self.scores[i]=0
        for f in features:
            #print "lookup", f
            try:
                p = self.W[f]
                for c in xrange(self.nclasses):
                    self.scores[c] += p.w[c]
            except KeyError: 
                #print "feature",f,"not found"
                pass
        # return best_i
        cdef int best_i = 0
        cdef double best = self.scores[0]
        for i in xrange(1,self.nclasses):
            if best < self.scores[i]:
                best_i = i
                best = self.scores[i]
        return best_i

    cdef _update(self, int goodClass, int badClass, list features):
        cdef MulticlassParamData p
        for f in features:
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p

            p.acc[badClass]+=(self.now-p.lastUpd[badClass])*p.w[badClass]
            p.acc[goodClass]+=(self.now-p.lastUpd[goodClass])*p.w[goodClass]
            p.w[badClass]-=1.0
            p.w[goodClass]+=1.0
            p.lastUpd[badClass]=self.now
            p.lastUpd[goodClass]=self.now

    cdef _update_r(self, int goodClass, int badClass, list features):
        cdef MulticlassParamData p
        for f,v in features:
            try:
                p = self.W[f]
            except KeyError:
                p = MulticlassParamData(self.nclasses)
                self.W[f] = p

            p.acc[badClass]+=(self.now-p.lastUpd[badClass])*p.w[badClass]
            p.acc[goodClass]+=(self.now-p.lastUpd[goodClass])*p.w[goodClass]
            p.w[badClass]-=v
            p.w[goodClass]+=v
            p.lastUpd[badClass]=self.now
            p.lastUpd[goodClass]=self.now

    def finalize(self):
        cdef MulticlassParamData p
        # average
        for f in self.W.keys():
            for c in xrange(self.nclasses):
                p = self.W[f]
                p.acc[c]+=(self.now-p.lastUpd[c])*p.w[c]
                p.w[c] = p.acc[c] / self.now

    def dump(self, out=sys.stdout):
        cdef MulticlassParamData p
        # Write LibSVM compatible format
        out.write(u'solver_type L1R_LR\n')
        out.write(u'nr_class %d\n' % self.nclasses)
        out.write(u'label %s\n' % ' '.join([str(self.labels[i]) for i in range(self.nclasses)]))
        out.write(u'nr_feature %d\n' % max(self.W.keys()))
        out.write(u'bias -1\n')
        out.write(u'w\n')
        max_key = max(self.W.keys())
        for f in range(1, max_key):
            if f not in self.W:
                out.write((u'0 ' * self.nclasses) + u'\n')
                continue
            p = self.W[f]
            for c in xrange(self.nclasses):
                out.write(u" %s" % p.w[c])
            out.write(u"\n")
        out.close()

    def load(self, in_=sys.stdin):
        cdef MulticlassParamData p
        n_labels = -1
        label_map = None
        header, data = in_.read().split('w\n')
        for line in header.split('\n'):
            if line.startswith('label'):
                label_names = line.strip().split()
                # Remove the word "label"
                label_names.pop(0)
        self.set_labels([int(ln) for ln in label_names])
        for i, line in enumerate(data.strip().split('\n')):
            pieces = line.strip().split()
            assert len(pieces) == len(label_names)
            p = MulticlassParamData(len(label_names))
            for j, w in enumerate(pieces):
                p.w[j] = float(w)
            self.W[i + 1] = p  

#    def dump_fin(self,out=sys.stdout):
#        cdef MulticlassParamData p
#        # write the average
#        for f in self.W.keys():
#            out.write("%s" % f)
#            for c in xrange(self.nclasses):
#               p = self.W[f]
#               out.write(" %s " % ((p.acc[c]+((self.now-p.lastUpd[c])*p.w[c])) / self.now))
#            out.write("\n")

