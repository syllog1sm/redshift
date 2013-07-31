# cython: profile=True
"""
Handle parser features
"""
from libc.stdlib cimport malloc, free, calloc
from libc.stdint cimport uint64_t
import index.hashes

from ext.murmurhash cimport *


cdef int free_predicate(Template* pred) except -1:
    free(pred.raws)
    free(pred.args)
    free(pred)
 

cdef class Extractor:
    def __cinit__(self, templates, match_templates):
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        self.mask_value = index.hashes.encode_word('<MASKED>')
        templates = tuple(sorted(set([tuple(sorted(f)) for f in templates])))
        self.nr_template = len(templates)
        self.templates = <Template**>malloc(self.nr_template * sizeof(Template*))
        # Sort each feature, and sort and unique the set of them
        cdef Template* pred
        for id_, args in enumerate(templates):
            pred = <Template*>malloc(sizeof(Template))
            pred.id = id_
            pred.n = len(args)
            pred.raws = <uint64_t*>malloc((len(args) + 1) * sizeof(uint64_t))
            pred.args = <int*>malloc(len(args) * sizeof(int))
            for i, element in enumerate(sorted(args)):
                pred.args[i] = element
            self.templates[id_] = pred

        self.nr_match = len(match_templates)
        self.match_preds = <MatchPred**>malloc(self.nr_match * sizeof(MatchPred*))
        cdef MatchPred* match_pred
        for id_, (idx1, idx2) in enumerate(match_templates):
            match_pred = <MatchPred*>malloc(sizeof(MatchPred))
            match_pred.id = id_ + self.nr_template
            match_pred.idx1 = idx1
            match_pred.idx2 = idx2
            self.match_preds[id_] = match_pred

    def __dealloc__(self):
        #free(self.context)
        for i in range(self.nr_template):
            free_predicate(self.templates[i])
        free(self.templates)
        for i in range(self.nr_match):
            free(self.match_preds[i])
        free(self.match_preds)

    cdef int extract(self, uint64_t* features, size_t* context) except -1:
        cdef:
            size_t i, j, size
            uint64_t value
            bint seen_non_zero, seen_masked
            Template* pred
        cdef size_t f = 0
        # Extra trick:
        # Always include this feature to give classifier priors over the classes
        features[0] = 1
        f += 1
        for i in range(self.nr_template):
            pred = self.templates[i]
            seen_non_zero = False
            seen_masked = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                # Extra trick: provide a way to exclude features that depend on
                # rare vocabulary items
                if value == self.mask_value:
                    seen_masked = True
                    break
                if value != 0:
                    seen_non_zero = True
                pred.raws[j] = value
            if seen_non_zero and not seen_masked:
                pred.raws[pred.n] = pred.id
                size = (pred.n + 1) * sizeof(uint64_t)
                features[f] = MurmurHash64A(pred.raws, size, i)
                f += 1
        cdef MatchPred* match_pred
        cdef size_t match_id
        for match_id in range(self.nr_match):
            match_pred = self.match_preds[match_id]
            value = context[match_pred.idx1]
            if value != 0 and value == context[match_pred.idx2]:
                match_pred.raws[0] = value
                match_pred.raws[1] = match_pred.id
                features[f] = MurmurHash64A(match_pred.raws, 2 * sizeof(size_t),
                                            match_pred.id)
                f += 1
                match_pred.raws[0] = 0
                features[f] = MurmurHash64A(match_pred.raws, 2 * sizeof(size_t),
                                            match_pred.id)
                f += 1
        features[f] = 0
