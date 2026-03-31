# MT Exercise 2: Pytorch RNN Language Models

# Requirements 
*(from initial description, still applies)*

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`


# Part 1: Training the Model

## Data Preparation

I have chosen two novels by F. Scott Fitzgerald, “The Great Gastby” and “This Side of Paradise” as my database.
I adapted the provided script to fit my choice of data. 

The data download and preprocessing I did can be replicated by calling:

    ./scripts/download_data.sh

What this does:
- creates a directory named `fitzgerald` for the data
    - `raw` stores the source and preprocessed files
- automatically downloads the two novels from Project Gutenberg using `wget` 
- concatenating the two novels
- preproceessing based on the provided code
    - `vocab_size` = 7000 (increased this from 5000, because in the first run with 5000 this produced a lot of <unk> tokens)
- splitting into train, valid and test sets
    - created code to automatically adapt thresholds: 
        - 80% of lines -> train.txt
        - 10% of lines -> valid.txt
        - 10% of lines -> test.txt
- final segments:
    output from `wc -l data/fitzgerald/[tv]*`:
    ```text
    679 data/fitzgerald/test.txt
    5422 data/fitzgerald/train.txt
    677 data/fitzgerald/valid.txt
    6778 total
    ```

## Training

Before training I adapted the `main.py` from `tools/pytorch-examples/word_language_model` to match my hardware and increase efficiency. This sets the accelerator to using "mps" if applicable. Using GPU instead of CPU reduced the runtime from around 350 seconds to 140 seconds for a training run with 40 epochs.  
To use this the argument `--accel` must be added to the function call. 
A copy of `main.py` is added to the `scripts` depository as `main_modified.py`.

Then, I trained the model on my fitzgerald data using the settings of dropout = 0.5 and epochs = 40.

## Generation

To generate the text from the trained model I adapted the provided script with the relevant model name. This can still be called with:

    ./scripts/generate.sh

The pdf document from my submission discusses my findings from the sample generation.

# Part 2: Dropout Experiments

## Logging

I modified the `main.py` to store the logging information in a tabular format. 

Explanation: 
- Input: called with `--log-file "desired_output_file.tsv"`  
- Ouput: Each run produces a log file: `logs/dropout_<value>.tsv`
- Log Format:
    ```text
    split   epoch   batch   loss   ppl
    train   1       100     ...    ...
    valid   1       end     ...    ...
    test    final   end     ...    ...
    ``` 
    - Contains:
        - training perplexity (per log interval)
        - validation perplexity (per epoch)
        -  test perplexity (final)
    
In order to iterate over the different dropout values I also adapted `train.sh`.
- The desired dropout values can now be indicated in the variable `dropouts=("0.0" "0.5")`.
- The script loops over the values and passes them to the train function. 
- The code also builds the matching model name, with which it stores the model and log file.

## Analysis

I created the script `scripts/dropout_analysis.py` for processing the log files into tables and graphs. This can then be used for comparison of the model performances and analysis. 

The script takes one or more TSV log files as input and creates:
- tables for training perplexity by epoch
- tables for validation perplexity by epoch
- a table with final test perplexity
- a line plot for training perplexity
- a line plot for validation perplexity

### Package Installations

I made use of pandas and matplotlib. These must first be installed in the venv before running the script. 
I used:
`pip3 install matplotlib`
`pip3 install pandas`

### Input format
The script takes one or more TSV log files as input. The expected column format in each log file is:

split   epoch   batch   loss   ppl

This is in line with my previous code adaptations that creates the log files. 

The path for the desired output directory can also be passed as an input argument with the flag `--outdir`. Without it this defaults to `results/dropout_analysis/`.

### Note about the Table creation
The logs contain multiple training rows per epoch. The last train row of each epoch is used as the epoch-level training perplexity.

### Run

The script can be run as follows: `python3 scripts/dropout_analysis.py logs/log_dp_*.tsv`

### Output

The script writes results to the desired output directory and produces:
- tables: 
    - content: 
        - for training perplexity by epoch
        - for validation perplexity by epoch
        - with final test perplexity
    - format
        - `.tsv` tables, suitable for further processing
        -  `.md` tables, more readable
- visualisations:
    - training_perplexity.png
    - validation_perplexity.png