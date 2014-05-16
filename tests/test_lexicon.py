import index.lexicon

def test_vocab():
    addr = index.lexicon.lookup('Hello')
    assert index.lexicon.get_str(addr) == 'Hello'
