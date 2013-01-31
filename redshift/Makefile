REPO=/Users/matt/matthew.honnibal/redshift

parser: setup_parser.py features.so io_parse.so _state.so parser.pyx 
	python setup_parser.py build_ext --inplace

features.so: setup_features.py features.pyx features.pxd
	python setup_features.py build_ext --inplace

io_parse.so: setup_io.py io_parse.pyx io_parse.pxd
	python setup_io.py build_ext --inplace

_state.so: setup_state.py _state.pyx _state.pxd
	python setup_state.py build_ext --inplace

index/hashes.so: index/hashes.pyx index/hashes.pxd
	make -C index/

svm/cy_svm.so: svm/cy_svm.pyx svm/cy_svm.pxd
	make -C svm/

clean:
	rm -rf *~ features.cpp parser.cpp _state.cpp io_parse.cpp *.so *.o *.html build

