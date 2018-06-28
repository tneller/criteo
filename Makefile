SHELL = /bin/sh

srcdir = .

all: install

install: datafiles

datafiles: train.txt test.txt train_half.txt test_half.txt train_small.txt train_small_one_hot.txt train_smaller.txt train_smaller_one_hot.txt

dac.tar.gz:
	wget https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz

dac.tar: dac.tar.gz
	gunzip dac.tar.gz

criteo_data: dac.tar
	tar -xvf dac.tar

train.txt: criteo_data

test.txt: criteo_data

train_half.txt: criteo_data
	head -22920309 train.txt > train_half.txt

test_half.txt: criteo_data
	tail -22920308 train.txt > test_half.txt

train_small.txt: criteo_data
	head -1000000 train.txt > train_small.txt

train_smaller.txt: train.txt
	head -100000 train.txt > train_smaller.txt

one_hot: train_small.txt train_smaller.txt
	sh make_one_hots.sh	

train_small_one_hot.txt: train_small.txt one_hot

train_smaller_one_hot.txt: train_smaller.txt one_hot



