# program to combine training data files

filenames = ['TrainingData/iX_6_100000_.csv',
             'TrainingData/iX_6_100001_.csv',
             'TrainingData/iX_6_100002_.csv',
             'TrainingData/iX_7_100000_.csv',
             'TrainingData/iX_7_100001_.csv',
             'TrainingData/iX_7_100002_.csv',
             'TrainingData/iX_10_100000_.csv']


with open('TrainingData/iX1', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

filenames = ['TrainingData/iY_6_100000_.csv',
             'TrainingData/iY_6_100001_.csv',
             'TrainingData/iY_6_100002_.csv',
             'TrainingData/iY_7_100000_.csv',
             'TrainingData/iY_7_100001_.csv',
             'TrainingData/iY_7_100002_.csv',
             'TrainingData/iY_10_100000_.csv']


with open('TrainingData/iY1', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)


