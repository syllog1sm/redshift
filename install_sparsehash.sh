#!/usr/bin/env bash

if [ $VIRTUALENV ] 
  then 
    dest="$VIRTUALENV"
  else
    dest=`pwd`/ext
fi
echo $dest
exit
cd /tmp
wget https://sparsehash.googlecode.com/files/sparsehash-2.0.2.tar.gz
tar -xzf sparsehash-2.0.2.tar.gz
cd sparsehash-2.0.2
./configure --prefix=$dest
make
make install
