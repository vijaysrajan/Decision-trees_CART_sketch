#  Step 1: Create 2-Column Sketches from Raw CSV
./venv/bin/python tools/create_2col_sketches.py tests/resources/agaricus-lepiota.csv class --lg_k 19
#  Step 2: Convert 2-Column to 3-Column Format

./venv/bin/python tools/simple_convert_to_3col.py \
      agaricus_lepiota_sketches/agaricus_lepiota_positive_2col_sketches_lg_k_19.csv \
      agaricus_lepiota_sketches/agaricus_lepiota_negative_2col_sketches_lg_k_19.csv \
      agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
      agaricus_lepiota_sketches/

#  Step 3: Train from 3-Column Sketches

./venv/bin/python tools/train_from_3col_sketches.py \
      agaricus_lepiota_sketches/agaricus_lepiota_3col_sketches.csv \
      agaricus_lepiota_sketches/agaricus_lepiota_feature_mapping.json \
      agaricus_lepiota_sketches/mushroom_training_config.yaml

