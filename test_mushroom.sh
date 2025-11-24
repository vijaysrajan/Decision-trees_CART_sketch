./venv/bin/python run_binary_classification.py \
      ./tests/resources/binary_classification_data.csv target \
      --lg_k 16 \
      --max_depth 5 \
      --criterion gini \
      --tree_builder intersection \
      --verbose 1 \
      --sample_size 10000 2>&1 > o3.json
