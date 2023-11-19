
# Shell script to run the experiments for the paper for all the test with DenseNet161 model
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2020/test_split-Seg.csv'
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2017/test_train_split01-Seg.csv'
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2017/test_train_split02-Seg.csv'
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2017/test_train_split03-Seg.csv'
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2017/test_train_split04-Seg.csv'
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2017/test_train_split05-Seg.csv'
python selective-kernel-fusion-resnet101.py -model DenseNet161 -splitPath 'Data-Splits/LivDet-Iris-2017/test_train_split06-Seg.csv'


