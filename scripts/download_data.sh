#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/fitzgerald

mkdir -p $data/fitzgerald/raw

# wget https://gutenberg.org/cache/epub/64317/pg64317.txt
# mv pg64317.txt $data/fitzgerald/raw/gatsby.txt

# wget https://gutenberg.org/cache/epub/805/pg805.txt 
# mv pg805.txt $data/fitzgerald/raw/paradise.txt

# cat \
#   $data/fitzgerald/raw/gatsby.txt \
#   $data/fitzgerald/raw/paradise.txt \
#   > $data/fitzgerald/raw/all.txt

# # preprocess slightly

# cat $data/fitzgerald/raw/all.txt | python $base/scripts/preprocess_raw.py > $data/fitzgerald/raw/all.cleaned.txt

# tokenize, fix vocabulary upper bound

# cat $data/fitzgerald/raw/all.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 7000 --tokenize --lang "en" --sent-tokenize --language "english" > \
#     $data/fitzgerald/raw/all.preprocessed.txt

# split into train, valid and test


total=$(wc -l < $data/fitzgerald/raw/all.preprocessed.txt)

train_lines=$(( total * 80 / 100 ))
valid_lines=$(( total * 10 / 100 ))
test_lines=$(( total - train_lines - valid_lines ))

head -n $train_lines $data/fitzgerald/raw/all.preprocessed.txt > $data/fitzgerald/train.txt

tail -n +$(( train_lines + 1 )) $data/fitzgerald/raw/all.preprocessed.txt | \
head -n $valid_lines > $data/fitzgerald/valid.txt

tail -n $test_lines $data/fitzgerald/raw/all.preprocessed.txt > $data/fitzgerald/test.txt
