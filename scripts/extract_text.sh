#!/usr/bin/env bash

cut -f2,4 | tr '\t' '/' | tr '\n' ' ' | sed 's/  /@/g' | tr '@' '\n'
