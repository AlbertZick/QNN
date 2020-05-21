# data set hierarchy
```python
#  ../DataSet/
#  ├ F10
#  │   ├ long
#  │   └ short
#  └ Sony
#      ├ long
#      └ short
```
# How to train model ?
##Command
python &nbsp; &nbsp; qnn.py &nbsp; &nbsp; [option]

option:
   * rtrain &nbsp; &nbsp; &nbsp; [Model_File]
   * test &nbsp; &nbsp; &nbsp; [Model_File]
   * train &nbsp; &nbsp; &nbsp; [sub-option]
   * -tr &nbsp; &nbsp; &nbsp; &nbsp; [input train file]
   * -v &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [input evaluate file]
   * -t &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [test file]

sub-option:
   * -i &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [init file]

Code automatically uses train_list.txt, val_list.txt and test_list.txt to specify train data, 
evaluate data and test data respectively if these files is not given in input arguments

