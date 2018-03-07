# SequenceLabeling
A TensorFlow (>=0.12.1) implementation of BRNN + CRF + Char-level embedding model for general sequence labeling, e.g., POS tagging, Named Entity Recognition, and Slot Tagging.

# Help
python SequenceLabelingBRNNCRF.py -h

# Training
python SequenceLabelingBRNNCRF.py -T train -d data_dir -t training_filename -v dev_filename -p pretrained_word_embedding_path -C LSTM -e 300 -H 300 -k 0.5 -l 0.001 -b 20 -i 30 -c 1 --use_chars -E 100 --hidden_size_char 100 --decay 0.90 --fix_word_embedding --annotation_scheme CoNLL --brnn_type vanilla [--yield_data]
yield_data here means that only a mini-batch data is yield from a dataset iterator, which is more memory efficient for large scale data.

# Evaluation
python SequenceLabelingBRNNCRF.py -T eval --model_dir model_path --eval_filepath evaluation_filepath --annotation_scheme CoNLL

# Prediction
python SequenceLabelingBRNNCRF.py -T eval --model_dir model_path --test_filepath test_filepath
In test_filepath, each line is a sequence of tokens. 

# Interactive prediction
python SequenceLabelingBRNNCRF.py -T online --model_dir model_path
