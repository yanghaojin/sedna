#!/bin/bash

set +x

KAGGLE_DATASETS_PATH=../kaggle

sedna-update-to-kaggle() {
  set -x
  rm $KAGGLE_DATASETS_PATH/sedna.zip
  zip -q -r $KAGGLE_DATASETS_PATH/sedna.zip ./lib ./examples
  kaggle datasets version -p $KAGGLE_DATASETS_PATH -m "$(date)"
  set +x
}


# update_sedna_to_kaggle_from_github

