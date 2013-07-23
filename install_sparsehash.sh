#!/usr/bin/env bash

if [-n $VIRTUALENV] 
  then 
    dest=$VIRTUALENV
  else
    dest=`pwd`
fi
cd /tmp
wget https://sparsehash.googlecode.com/files/sparsehash-2.0.2.tar.gz
tar -xzf sparsehash-2.0.2.tar.gz
cd sparsehash-2.0.2
./configure --prefix=$dest
make
make install
