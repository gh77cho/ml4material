import sys
import tensorflow as tf
import linecache
import json
import random
from model import Model
from model import LinearRegression
import cPickle

class ML:
    MODELS = [ 'linear-regression' ]
    input_file = None
    target_name = None
    feature_name = None
    conf = None
    fine_tune_param = None
    loss_file = None

    def get_feature_name(self):
        header = linecache.getline(self.input_file, 1).strip().split(',')
        return header[-1], header[:-1]

    def get_model_conf(self, JsonFile):
        f = open(JsonFile)
        conf = json.loads(f.read())
        f.close()
        return conf

    def get_fine_tune_param(self, JsonFile):
        f = open(JsonFile)
        model = json.loads(f.read())
        f.close()

        for feature in self.feature_name:
            if feature not in model['Input']:
                print >> sys.stderr, 'error: fail to run fine tune. feature "%s" is undefined' % feature
                sys.exit(1)
        if len(self.conf['hidden-layer'])+1 != len(model['Weight']):
            print >> sys.stderr, 'error: fail to run fine tune. Network is different' % feature
            sys.exit(1)

        f_idx = {}
        for i, feature in enumerate(model['Input']):
            f_idx[feature] = i

        self.fine_tune_param = []
        W = []
        B = []
        I = []
        for feature in self.feature_name:
            i = f_idx[feature]
            I.append(model['Weight'][0][i])
        W.append(I)

        for weight in model['Weight'][1:]:
            W.append(weight)
        for bias in model['Bias']:
            B.append(bias[0])

        self.fine_tune_param.append(W)
        self.fine_tune_param.append(B)

    def get_data(self):
        train_data = []
        train_Y = []
        test_data = []
        test_Y = []

        lines = []
        f = open(self.input_file)
        lineno = 0
        for line in f:
            lineno += 1
            if lineno == 1:
                continue
            line = line.strip()
            lines.append(line)
        f.close()

        data = []
        Y = []
        random.shuffle(lines)
        for line in lines:
            cols = line.split(',')
            Y.append(float(cols[-1]))
            data.append([float(v) for v in cols[:-1]])


        train_n = int((1.-self.conf['test-ratio']) * len(data))
        train_data = data[:train_n]
        train_Y = Y[:train_n]
        test_data = data[train_n:]
        test_Y = Y[train_n:]
        print >> sys.stderr, '\t- record count : %d' % (lineno-1)
        print >> sys.stderr, '\t- record count of train : %d' % len(train_data)
        print >> sys.stderr, '\t- record count of test : %d (%.2f%%)' % (len(test_data), 100.*self.conf['test-ratio'])

        return train_data, train_Y, test_data, test_Y

    def __init__(self, InputFile=None, ModelJsonFile=None, FineTuneJsonFile=None, WithLossFile=False):
        print >> sys.stderr, '----------------------------------------------------------------------'
        print >> sys.stderr, 'Train'
        print >> sys.stderr, '----------------------------------------------------------------------'
        self.input_file = InputFile
        self.target_name, self.feature_name = self.get_feature_name()
        self.conf = self.get_model_conf(ModelJsonFile)
        if FineTuneJsonFile:
            self.get_fine_tune_param(FineTuneJsonFile)

        print >> sys.stderr, '\t- Input File : %s' % InputFile
        print >> sys.stderr, '\t- Model Definition File : %s' % ModelJsonFile
        if self.fine_tune_param:
            print >> sys.stderr, '\t- Fine Tune Model Json File : %s' % FineTuneJsonFile
        print >> sys.stderr, '\t- Target Feature : %s' % self.target_name
        print >> sys.stderr, '\t- Input Features : %s' % self.feature_name

        self.train_data, self.train_Y, self.test_data, self.test_Y = self.get_data()
        if WithLossFile:
            self.loss_file = '%s.loss' % self.conf['name']
        
    def learning(self):
        if self.conf['model'] not in self.MODELS:
            print >> sys.stderr, 'error: invalid model - "%s"' % self.conf['model']
            sys.exit(1)

        if self.conf['model'] == 'linear-regression':
            model_lr = LinearRegression(self.conf, len(self.feature_name), param=self.fine_tune_param, loss_file=self.loss_file)
            model_lr.learning(self.train_data, self.train_Y, self.test_data, self.test_Y)
            model_lr.print_model(self.feature_name)

    def estimate(self, ModelFile):
        if self.conf['model'] not in self.MODELS:
            print >> sys.stderr, 'error: invalid model - "%s"' % self.conf['model']
            sys.exit(1)

        param = cPickle.load(open(ModelFile))
        sample_data = linecache.getline(self.input_file, 1).strip().split(',')

        if self.conf['model'] == 'linear-regression':
            model_lr = LinearRegression(self.conf, len(sample_data), param=param)
            model_lr.estimate(self.input_file)


