"""Test the dynamic oracle for sentence boundary detection cases"""


from redshift import Sentence, Token
from redshift.tester import State
from redshift.sentence import get_labels

class Tester(object):
    def __init__(self):
        tokens = [
            Token(0, 'I', 'PRP', 1, 'subj', False),
            Token(1, 'like', 'VBP', -1, 'ROOT', False),
            Token(2, 'cookies', 'NNS', 1, 'dobj', False),
            Token(0, 'Bob', 'PRP', 4, 'subj', False),
            Token(1, 'likes', 'VBZ', -1, 'ROOT', False),
            Token(2, 'cake', 'NN', 4, 'dobj', False)
        ]

        self.sent = Sentence(0, tokens)
        assert self.sent.tokens[0].word == 'I'
        assert self.sent.tokens[5].word == 'cake'

    def test_s_l_s(self):
        s = State(self.sent, *get_labels([self.sent]))
        moves = [s.S, s.L + 1, s.S]
        for move in moves:
            assert s.is_valid(move)
            s.transition(move)
        assert s.is_valid(s.D) == False

    def test_s_l_s_r_b(self):
        s = State(self.sent, *get_labels([self.sent]))
        moves = [(s.S, s.s_cost), (s.L, s.l_cost), (s.S, s.s_cost), (s.R, s.r_cost)]
        for move, cost in moves:
            assert s.is_valid(move), move
            assert cost() == 0, move
            s.transition(move)
        assert s.is_valid(s.B)
        assert s.b_cost() == 0, s.b_cost()
        assert s.d_cost() == 1, s.d_cost()
        assert s.s_cost() == 1, s.s_cost()
