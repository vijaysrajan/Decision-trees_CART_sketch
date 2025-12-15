 # Basic usage - show all rules
  ./venv/bin/python tools/extract_tree_rules.py DU_output/test_model_lg_k_18.json/3col_sketches_model_lg_k_18.json

  # Save SQL validation queries
  ./venv/bin/python tools/extract_tree_rules.py DU_output/test_model_lg_k_18.json/3col_sketches_model_lg_k_18.json \
      --save_sql DU_output/tree_validation_queries.sql \
      --save_rules DU_output/tree_rules.json \
      --table DU_raw --target_column target

  # Show only top 10 rules  
  ./venv/bin/python tools/extract_tree_rules.py tree.json --max_display 10

  2. tools/quick_tree_test.py - Quick Rule Testing

  Usage:
  # Get top 3 rules for quick testing
  ./venv/bin/python tools/quick_tree_test.py DU_output/test_model_lg_k_18.json/3col_sketches_model_lg_k_18.json --num_rules 3

  # Different table/column names
  ./venv/bin/python tools/quick_tree_test.py tree.json --table my_table --target my_target_col

