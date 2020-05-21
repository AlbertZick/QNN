# data set hierarchy
../DataSet/
├ F10
│   ├ long
│   └ short
└ Sony
    ├ long
    └ short

# How to train model ?
1. Command:
python qnn [option]

option:
   * rtrain  [Model_File]
   * test    [Model_File]
   * train   [sub-option]
   * -tr     [input train file]
   * -v      [input evaluate file]
   * -t      [test file]

sub-option:
   * -f [init file]


model automatically uses train_list.txt, val_list.txt and test_list.txt to specify train data, 
evaluate data and test data respectively if these files is not given in input arguments

