#!/usr/bin/env bash

if [ $VIRTUAL_ENV ] 
  then 
    dest="$VIRTUAL_ENV"
  else
    dest=`pwd`/ext
fi
cd /tmp
wget https://sparsehash.googlecode.com/files/sparsehash-2.0.2.tar.gz
tar -xzf sparsehash-2.0.2.tar.gz
cd sparsehash-2.0.2
./configure --prefix=$dest
make
make install
