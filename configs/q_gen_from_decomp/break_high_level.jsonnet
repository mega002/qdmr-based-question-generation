local config = import "base.jsonnet";

config {
  num_epochs:: 15,
  "train_data_path": "data/break_high_level/train.csv",
  "validation_data_path": "data/break_high_level/dev.csv",
}
