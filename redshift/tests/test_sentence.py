from redshift.sentence import Input
import index.vocab

def test_vocab():
    index.vocab.load_vocab()
    addr = index.vocab.lookup('Hello')
    assert index.vocab.get_str(addr) == 'Hello'
    

def test_tokens():
    blob = Input.from_tokens(['Hello', 'world'])
    assert blob.length == 2
    assert blob.words == ['Hello', 'world']
