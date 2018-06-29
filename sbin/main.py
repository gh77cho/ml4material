import os
import signal
import sys
import getopt
sys.path.append('.')
from common import load_field_definition
from common import load_split_data
from common import pearson
from train import ML
import math
import json

def print_common_msg():
    print >> sys.stderr, 'Enter `%s --help` if you want to kwow more options' % sys.argv[0]
    sys.exit(2)

def usage(name):
    print 'Usage: %s [OPTIONS]' % name
    print ''
    print 'OPTIONS:'
    print '  -h, --help                                   print this message'
    print '  -C, --command=COMMAND                        Command'
    print '  -D, --filed-definiton-json-file=FILEPATH     Field Definition Json File'
    print '  -i, --input-csv-file=FILEPATH                Input CSV File'
    print '      --data-directory=DIRPATH                 Output Data Directory Path'
    print '  -T, --target=TARGET_FEATURE                  Target File Path'
    print '  -F, --feature=FEATURE_DEFINIITON             Feature List (delimeter : "," Interaction Operator : "*")'
    print '  -o, --output-file=FILEPATH                   Output File Path'
    print '  -m, --model-json-file=FILEPATH               Input Model Definition Json File'
    print '  -M, --model-file=FILEPATH                    Input Trained Model File'
    print '  -L, --with-loss-file                         Write Loss to file named "CONF.NAME.loss"'
    print '      --fine-tune-json-file=FILEPATH           Fine Tuning. Weight will be initialized by thie model file'
    print ''
    print 'COMMANDS:'
    print '  1. split-data                                Split Input CSV Data'
    print '     Mandatory Options:'
    print '     -D or --filed-definiton-json-file'
    print '     -i or --input-csv-file'
    print '           --data-directory'
    print ''
    print '  2. norm-data                                 Make Normalized Input Data'
    print '     Mandatory Options:'
    print '     -D or --filed-definiton-json-file'
    print '           --data-directory'
    print ''
    print '  3. tarin-data                                Make Train Data'
    print '     Mandatory Options:'
    print '           --data-directory'
    print '     -T or --target'
    print '     -F or --feature'
    print '     -o or --output-file'
    print ''
    print '  4. train                                     Run Training'
    print '     Mandatory Options:'
    print '     -i or --input-csv-file'
    print '     -m or --model-json-file'
    print ''
    print '  5. estimate                                  Estimate Target Value'
    print '     Mandatory Options:'
    print '     -m or --model-json-file'
    print '     -M or --model-file'
    print '     -i or --input-csv-file'
    print ''
    print 'MODELS:'
    print ''
    sys.exit(2)

def make_directory_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif os.path.isfile(path):
        print >> sys.stderr, '"%s" is file. Use another directory path' % path
        print_common_msg()

def check_command_options():
    CMDS = [ 'split-data', 'norm-data', 'train-data', 'train', 'estimate' ]
    if 'command' not in globals():
        print >> sys.stderr, '"-C or --command" option reuquired'
        print_common_msg()

    if command not in CMDS:
        print >> sys.stderr, 'command "%s" not recognized ' % command
        print_common_msg()

    if command == 'split-data':
        if 'FieldDefinitionJsonFile' not in globals():
            print >> sys.stderr, '"-D or --filed-definiton-json-file" option required'
            print_common_msg()
        if 'InputCSVFile' not in globals():
            print >> sys.stderr, '"-i or --input-csv-file" option required'
            print_common_msg()
        if 'DataDirectory' not in globals():
            print >> sys.stderr, '"--data-directory" option required'
            print_common_msg()
    elif command == 'norm-data':
        if 'FieldDefinitionJsonFile' not in globals():
            print >> sys.stderr, '"-D or --filed-definiton-json-file" option required'
            print_common_msg()
        if 'DataDirectory' not in globals():
            print >> sys.stderr, '"--data-directory" option required'
            print_common_msg()
    elif command == 'train-data':
        if 'DataDirectory' not in globals():
            print >> sys.stderr, '"--data-directory" option required'
            print_common_msg()
        if 'Target' not in globals():
            print >> sys.stderr, '"-T or --target" option required'
            print_common_msg()
        if 'FeatureList' not in globals():
            print >> sys.stderr, '"-F or --feature" option required'
            print_common_msg()
        if 'OutputFile' not in globals():
            print >> sys.stderr, '"-o or --output-file" option required'
            print_common_msg()
    elif command == 'train':
        if 'InputCSVFile' not in globals():
            print >> sys.stderr, '"-i or --input-csv-file" option required'
            print_common_msg()
        if 'ModelJsonFile' not in globals():
            print >> sys.stderr, '"-m or --model-json-file" option required'
            print_common_msg()
    elif command == 'estimate':
        if 'ModelJsonFile' not in globals():
            print >> sys.stderr, '"-m or --model-json-file" option required'
            print_common_msg()
        if 'ModelFile' not in globals():
            print >> sys.stderr, '"-M or --model-file" option required'
            print_common_msg()
        if 'InputCSVFile' not in globals():
            print >> sys.stderr, '"-i or --input-csv-file" option required'
            print_common_msg()

    if WithLossFile:
        if command != 'train':
            print >> sys.stderr, '"-L or --with-loss-file" can be used with command "train"'
            print_common_msg()

    if FineTuneJsonFile:
        if command != 'train':
            print >> sys.stderr, '"--fine-tune-json-file" can be used with command "train"'
            print_common_msg()

def split_data():
    FieldDefJson = load_field_definition(FieldDefinitionJsonFile)
    make_directory_if_not_exist(DataDirectory)

    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, 'Split Data'
    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, '\t- Field Definition File : %s' % FieldDefinitionJsonFile
    print >> sys.stderr, '\t- Input CSV File        : %s' % InputCSVFile
    print >> sys.stderr, '\t- Output Data Directory : %s' % DataDirectory

    iFields = FieldDefJson['Input']
    tFields = FieldDefJson['Target']
    n_iFields = len(iFields)
    n_tFields = len(tFields)

    fo = []
    f = open(InputCSVFile)
    lineno = 0
    for line in f:
        lineno += 1
        line = line.strip('\r\n')
        cols = line.split(',')

        if len(cols) != n_iFields + n_tFields:
            print >> sys.stderr, 'error: invalid field count of "%s" (line no=%d)' % (InputCSVFile, lineno)
            sys.exit(1)

        if lineno == 1:
            i = 0
            n_i = n_iFields
            n_t = n_tFields
            for field in cols:
                if n_i > 0:
                    if field != iFields[i]:
                        print >> sys.stderr, 'error: invalid input CSV file header (Input)' 
                        print >> sys.stderr, '"%s"' % line
                        sys.exit(0)
                    n_i -= 1
                    i += 1
                    fo.append(open('%s/i.%s' % (DataDirectory, field), 'w'))
                    continue
                elif n_i == 0:
                    i = 0
                    n_i = -1

                if n_t > 0:
                    if field != tFields[i]:
                        print >> sys.stderr, 'error: invalid input CSV file header (Target)'
                        print >> sys.stderr, '"%s"' % line
                        sys.exit(0)
                    n_t -= 1
                    i += 1
                    fo.append(open('%s/t.%s' % (DataDirectory, field), 'w'))

            continue

        for i, field in enumerate(cols):
            print >> fo[i], field
                
    f.close()


    for f in fo:
        f.close()

def norm_data():
    FieldDefJson = load_field_definition(FieldDefinitionJsonFile)

    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, 'Make Normalized Data'
    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, '\t- Field Definition File : %s' % FieldDefinitionJsonFile
    print >> sys.stderr, '\t- Output Data Directory : %s' % DataDirectory

    iFields = FieldDefJson['Input']

    STAT = {}
    for field in iFields:
        inputFile = '%s/i.%s' % (DataDirectory, field)
        print >> sys.stderr, '\t- Normalize Input Data : %s' % inputFile
        f = open(inputFile)
        lineno = 0
        n = 0
        sum1 = 0.
        sum2 = 0.
        rawData = []
        for line in f:
            lineno += 1
            line = line.strip()
            if len(line) == 0:
                rawData.append(None)
                continue
            n += 1

            v = float(line)
            rawData.append(v)
            sum1 += v
            sum2 += v*v
        f.close()

        avg = float(sum1) / float(n)

        sum2 = 0.
        for v in rawData:
            if v:
                sum2 += (v-avg)*(v-avg)
        std = math.sqrt(sum2/(n-1))

        STAT[field] = {}
        STAT[field]['mean'] = avg
        STAT[field]['stddev'] = std

        if std < 1e-16:
            print >> sys.stderr, '\t\t- Standard-Deviation is Zero. This feature cannot be normalized by "z-score"'
            continue

        print >> sys.stderr, '\t\t- Statistic[LineNo, ValidLineNo, mean, StandardDev] : [ %d, %d, %f, %f]' % (lineno, n, avg, std)

        outputFile = '%s/i.%s.zmean' % (DataDirectory, field)
        print >> sys.stderr, '\t\t- ZeroMean Normalize : %s' % outputFile
        f = open(outputFile, 'w')
        for v in rawData:
            if v == None:
                print >> f, ''
            else:
                print >> f, '%f' % (v-avg)
        f.close()

        outputFile = '%s/i.%s.zscore' % (DataDirectory, field)
        print >> sys.stderr, '\t\t- Z-Score Normalize  : %s' % outputFile
        f = open(outputFile, 'w')
        for v in rawData:
            if v == None:
                print >> f, ''
            else:
                print >> f, '%f' % ((v-avg)/std)
        f.close()

    statJson = json.dumps(STAT)
    statFile = '%s/stat.json' % DataDirectory
    f = open(statFile, 'w')
    print >> f, statJson
    f.close()

def get_feature_name(name):
    return '.'.join(name.split('/')[-1].split('.')[1:])

def print_correlation(target, featureList, corr):
    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, 'Pearson correlation coefficient'
    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, '\tTarget : %s' % target
    for i, feature in enumerate(featureList):
        print >> sys.stderr, '\tFeature Corr. "%s" : %f' % (feature, corr[i] if corr[i] else 0)


def train_data():
    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, 'Make Train Data'
    print >> sys.stderr, '----------------------------------------------------------------------'
    print >> sys.stderr, '\t- Load Target Feature : %s' % Target
    TargetFile = '%s/t.%s' % (DataDirectory, Target)
    if not os.path.isfile(TargetFile):
        print >> sys.stderr, 'error: invalid Target Feature Name - "%s"' % Target
        sys.exit(1)
    target = load_split_data(TargetFile)
    singleFeatureList = {}
    trainFeatureNameList = []
    trainFeatureList = []

    for feature in FeatureList:
        print >> sys.stderr, '\t- Load Feature : %s' % feature
        files = feature.split('*')
        isPolyFeature = False
        if len(files) > 1:
            isPolyFeature = True

        if not isPolyFeature:
            featureFile = '%s/i.%s' % (DataDirectory, feature)
            if not os.path.isfile(featureFile):
                print >> sys.stderr, 'error: invalid Feature Name - "%s"' % feature
                sys.exit(1)
            
            loadedFeature = load_split_data(featureFile)
            trainFeatureNameList.append(feature)
            trainFeatureList.append(loadedFeature)
            singleFeatureList[feature] = loadedFeature
        else:
            for f in files:
                featureFile = '%s/i.%s' % (DataDirectory, f)
                if not os.path.isfile(featureFile):
                    print >> sys.stderr, 'error: invalid Feature Name - "%s"' % f
                    sys.exit(1)
                if f not in singleFeatureList:
                    singleFeatureList[f] = load_split_data(featureFile)

                polyFeatureName = ''
                polyFeature = []
                for v in singleFeatureList[files[0]]:
                    polyFeatureName = files[0]
                    polyFeature.append(v)

                for f in files[1:]:
                    polyFeatureName += '-'+f
                    if f not in singleFeatureList:
                        singleFeatureList[f] = load_split_data(featureFile)
                    for i, v in enumerate(singleFeatureList[f]):
                        polyFeature[i] *= v

            trainFeatureNameList.append(polyFeatureName)
            trainFeatureList.append(polyFeature)


    fo = open(OutputFile, 'w')
    for i, head in enumerate(trainFeatureNameList):
        if i > 0:
            fo.write(',')
        fo.write(head)
    fo.write(',%s\n' % Target)
    for i, t in enumerate(target):
        if t == None:
            continue

        for fno, feature in enumerate(trainFeatureList):
            if fno > 0:
                fo.write(',')
            fo.write('%f' % feature[i])
        fo.write(',%f\n' % t)

    fo.close()
            
    corr = []
    for feature in trainFeatureList:
        corr.append(pearson(target, feature))

    print_correlation(Target, trainFeatureNameList, corr)

# Main 

try:
    optlist, args = getopt.getopt(sys.argv[1:], 'hC:D:i:T:F:o:m:M:L', [ 'help', 'command=', 'filed-definiton-json-file=', 'input-csv-file=', 'data-directory=', 'target=', 'feature=', 'output-file=', 'model-json-file=', 'model-file=', 'with-loss-file', 'fine-tune-json-file=' ])
except getopt.GetoptError, errmsg:
    print >> sys.stderr, errmsg
    print_common_msg()


WithLossFile = False
FineTuneJsonFile = None
for opt, value in optlist:
    if opt == '-h' or opt == '--help':
        usage(sys.argv[0])
        sys.exit()
    elif opt == '-C' or opt == '--command':
        command = value
    elif opt == '-D' or opt == '--filed-definiton-json-file':
        FieldDefinitionJsonFile = value
    elif opt == '-i' or opt == '--input-csv-file':
        InputCSVFile = value
    elif opt == '--data-directory':
        DataDirectory = value.strip('/')
    elif opt == '-T' or opt == '--target':
        Target = value
    elif opt == '-F' or opt == '--feature':
        FeatureList = value.split(',')
    elif opt == '-o' or opt == '--output-file':
        OutputFile = value
    elif opt == '-m' or opt == '--model-json-file':
        ModelJsonFile = value
    elif opt == '-M' or opt == '--model-file':
        ModelFile = value
    elif opt == '-L' or opt == '--with-loss-file':
        WithLossFile = True
    elif opt == '--fine-tune-json-file':
        FineTuneJsonFile = value

check_command_options()

if command == 'split-data':
    split_data()
elif command == 'norm-data':
    norm_data()
elif command == 'train-data':
    train_data()
elif command == 'train':
    train = ML(InputCSVFile, ModelJsonFile, FineTuneJsonFile, WithLossFile)
    train.learning()
elif command == 'estimate':
    train = ML(InputCSVFile, ModelJsonFile)
    train.estimate(ModelFile)
