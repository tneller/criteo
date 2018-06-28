#!/bin/sh
cd java-eclipse-workspace/criteo/
java MakeOneHotFreqCat < ../../train_small.txt > ../../train_small_one_hot.txt
java MakeOneHotFreqCat < ../../train_smaller.txt > ../../train_smaller_one_hot.txt
cd ../../
