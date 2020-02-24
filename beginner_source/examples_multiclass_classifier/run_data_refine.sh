if [ ! -d ".data" ]; then
    mkdir .data
fi

if [ ! -d ".data/aclass" ]; then
    mkdir .data/aclass
fi

python file_analyzer.py '/home/helloahn/Downloads/AClassification.train/AClassification.train.txt'