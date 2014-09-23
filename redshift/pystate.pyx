from .ae_transitions cimport Transition, transition, fill_valid, fill_costs
from .ae_transitions cimport get_nr_moves, fill_moves, move_name
from ._state cimport init_state

import index.hashes


cdef class PyState:
    def __cinit__(self, unicode string, list transitions=None, list gold=None):
        self.mem = Pool()
        self.sent = Input.from_untagged(string.encode('utf8'))
        self.state = init_state(self.sent.c_sent, self.mem)
        self.left_labels = ['SUBJ', 'ROOT', 'OTHER']
        self.encoded_left = [index.hashes.encode_label(label) for label in self.left_labels]
        self.right_labels = ['OBJ', 'OTHER']
        self.encoded_right = [index.hashes.encode_label(label) for label in self.right_labels]
        self.nr_moves = get_nr_moves(self.encoded_left, self.encoded_right, [], False)
        self.moves = <Transition*>self.mem.alloc(self.nr_moves, sizeof(Transition))
        fill_moves(self.encoded_left, self.encoded_right, [], False, self.moves)
        self.moves_by_name = {}
        for i in range(self.nr_moves):
            self.moves_by_name[move_name(&self.moves[i])] = i
        if transitions is not None:
            for transition in transitions:
                self.transition(transition)
        self.gold = <Token*>self.mem.alloc(self.sent.length, sizeof(Token))
        if gold is not None:
            for head, child, label in gold:
                self.gold[child].head = head
                self.gold[child].label = index.hashes.encode_label(label)
                
    def transition(self, unicode move_name):
        assert self.is_valid(move_name)
        transition(&self.moves[self.moves_by_name[move_name]], self.state)

    def is_valid(self, unicode move_name):
        fill_valid(self.state, self.moves, self.nr_moves)
        return self.moves[self.moves_by_name[move_name]].is_valid
 
    def is_gold(self, unicode move_name): 
        fill_costs(self.state, self.moves, self.nr_moves, self.gold)
        return self.moves[self.moves_by_name[move_name]].cost == 0

    property top:
        def __get__(self):
            return self.state.top
