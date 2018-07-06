from __future__ import (absolute_import, division, unicode_literals)
import signal
import sys
import tensorflow as tf
import numpy as np
import math
import json
import cPickle


class Model():
    def signal_handler(self, signal, frame):
        self.UserStop = True

    def initializer_w (self, hid, i, o):
        InitFunc = self.conf['initialize-function'] if 'initialize-function' in self.conf else 'xavier'
        name = 'weight_%d'%hid
        if InitFunc == 'random':
            return tf.get_variable(name, [i, o], initializer=tf.random_uniform_initializer([i,o],-1.,1,))
        elif InitFunc == 'normal':
            return tf.get_variable(name, [i, o], initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        return tf.get_variable(name, [i, o], initializer=self.xavier_init(i, o))

    def initializer_b (self, hid, o):
        InitFunc = self.conf['initialize-function'] if 'initialize-function' in self.conf else 'xavier'
        name = 'bias_%d'%hid
        if InitFunc == 'random':
            return tf.get_variable(name, [o], initializer=tf.random_uniform_initializer([o],-1.,1,))
        elif InitFunc == 'normal':
            return tf.get_variable(name, [o], initializer=tf.constant_initializer(0.0))
        return tf.get_variable(name, [o], initializer=tf.constant_initializer(0.0))

    def xavier_init (self, n_inputs, n_outputs, uniform=True):
        if uniform:
            init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
            return tf.random_uniform_initializer(-init_range, init_range)
        else:
            stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
            return tf.truncated_normal_initializer(stddev=stddev)

    def __init__(self, model, conf, feature_size, param=None, loss_file=None):
        self.UserStop = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.siginterrupt(signal.SIGINT, False)

        print '\t- model : %s' % model
        self.model = model
        self.conf = conf
        self.params = []
        self.data = tf.placeholder(tf.float32, shape=[None, feature_size], name='data')
        self.Y = tf.placeholder(tf.float32, shape=[None], name='Y')

        self.w = []
        self.b = []

        InitFunc = self.conf['initialize-function'] if 'initialize-function' in self.conf else 'xavier'

        print '\t- hiddel layer : %s' % conf['hidden-layer']
        if not param:
            i_n = feature_size
            for n in conf['hidden-layer']:
                n = feature_size if n <= 0 else n
                lid = len(self.w)
                o_n = n
                self.w.append(self.initializer_w(lid, i_n, o_n))
                self.b.append(self.initializer_b(lid, o_n))
                i_n = o_n
            self.w.append(self.initializer_w(len(self.w), i_n, 1))
            self.b.append(self.initializer_b(len(self.w), 1))

        else:
            l = 0
            for n in conf['hidden-layer']:
                self.w.append(tf.Variable(param[0][l]))
                l+=1
            self.w.append(tf.Variable(param[0][l]))
            l=0
            for n in conf['hidden-layer']:
                self.b.append(tf.Variable(param[1][l]))
                l+=1
            self.b.append(tf.Variable(param[1][l]))

        self.params.append(self.w)
        self.params.append(self.b)
        
        self.d_updates = None
        self.loss_file = loss_file

    def learning(self, train_data, train_Y, test_data, test_Y):
        self.train_data = train_data
        self.train_Y = train_Y
        self.test_data = test_data
        self.test_Y = test_Y

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.initialize_all_variables())
        rmse_best_val = 1000000.

        train_size = len(train_data)
        batch_size = 30

        stop_n = 0
        prev_loss = 0.

        if self.loss_file:
            f_loss = open(self.loss_file, 'w')

        for epoch in range(100):
            print >> sys.stderr, 'learning epoch - %d' % epoch
            
            for d_epoch in range(2000):
                index = 0
                f = 0
                t = batch_size
                sum_loss = 0.
                n = 0
                while True:
                    n += 1
                    if index > train_size:
                        break
                    if index + batch_size <= train_size + 1:
                        f = index
                        t = index+batch_size
                    else:
                        f = index
                        t = index+train_size-index
                    index += batch_size

                    _, loss = self.sess.run([self.d_updates, self.loss],
                            feed_dict={self.data: train_data[f:t], self.Y: train_Y[f:t]})
#                            feed_dict={self.data: np.asarray(train_data[f:t]), self.Y: train_Y[f:t]})
                    sum_loss += loss

                    if False:
                        print 'X: ', train_data
                        print 'Y: ', train_Y
                        print 'W: ', self.sess.run(self.w[0])
                        print 'W: ', self.sess.run(self.w[1])
                        print 'B: ', self.sess.run(self.b[0])
                        print 'B: ', self.sess.run(self.b[1])
                        print 'y: ', self.sess.run(self.y[-1], feed_dict={self.data: np.asarray(train_data), self.Y: np.asarray(train_Y)})
                        print 'D: ', self.sess.run(self.diff, feed_dict={self.data: np.asarray(train_data), self.Y: np.asarray(train_Y)})
                        sys.exit(0)

                avg_loss = float(sum_loss) / float(n)
                if math.fabs(avg_loss-prev_loss) < 0.0000001:
                    stop_n += 1
                prev_loss = avg_loss

                if self.conf['test-ratio'] == 0.:
                    print >> sys.stderr, 'train loss : %f' % avg_loss
                    if self.loss_file:
                        print >> f_loss, '%f' % avg_loss
                else:
                    test_loss = self.sess.run(self.loss, feed_dict={self.data: np.asarray(test_data), self.Y: np.asarray(test_Y)})
                    print >> sys.stderr, 'train loss : %f\ttest loss : %f' % (avg_loss, test_loss)
                    if self.loss_file:
                        print >> f_loss, '%f,%f' % (avg_loss, test_loss)

                if self.UserStop:
                    break
            if self.UserStop:
                break


            if stop_n >= 10:
                break

        if self.loss_file:
            f_loss.close()

        print 'EPOCH: %d - %d' % (epoch, d_epoch)

        all_data = self.train_data + self.test_data
        all_Y = self.train_Y + self.test_Y
        pred = self.sess.run(self.y[-1], feed_dict={self.data: np.asarray(all_data)})
        model_file = self.conf['name'] + '.estimate.csv'
        data = []
        if self.conf['hidden-layer'] == 0:
            for i, y in enumerate(pred[0]):
                Y = all_Y[i]
                data.append([Y, y])
        else:
            for i, y in enumerate(pred):
                Y = all_Y[i]
                data.append([Y, y])
        f = open(model_file, 'w')
        print >> f, 'id,Y,eY'
        for i, y in enumerate(sorted(data)):
            print >> f, '%d,%s,%f,%f' % (i,str(all_data[i]), y[0], y[1])
        f.close()

    def format(self, v):
        return v
#        return '  '.join([str(float(x)) for x in np.asarray(v)])

    def print_model(self, feature):
        print >> sys.stderr, '----------------------------------------------------------------------'
        print >> sys.stderr, 'Model : %s' % self.model
        print >> sys.stderr, '----------------------------------------------------------------------'
        param = self.sess.run(self.params)
        W = param[0]
        B = param[1]

        model = {}
        model['Input'] = []
        model['Weight'] = []
        model['Bias'] = []

        for i, w in enumerate(W):
            L = []
            b = []
            for j, v in enumerate(w):
                if i == 0:
                    print 'Input-Layer\t%s\t%s' % (feature[j], self.format(v))
                    model['Input'].append(feature[j])
                else:
                    print 'Hidden-Layer-%d\tnode-%d\t%s' % (i, j+1, self.format(v))

                flist = [ float(o) for o in v ]
                L.append(flist)

            if i == 0:
                print 'Input-Layer\t%s\t%s' % ('bias', self.format(B[i]))
            else:
                print 'Hidden-Layer-%d\t%s\t%s' % (i, 'bias', self.format(B[i]))

            blist = [ float(o) for o in B[i] ]
            b.append(blist)

            model['Weight'].append(L)
            model['Bias'].append(b)
                
        cPickle.dump(param, open('%s.model'%self.conf['name'], 'w'))

        with open('%s.model.json'%self.conf['name'], 'w') as f:
            json.dump(model, f, indent=4)

    def estimate(self, InputFile):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print >> sys.stderr, '----------------------------------------------------------------------'
        print >> sys.stderr, 'Estimate Target Value'
        print >> sys.stderr, '----------------------------------------------------------------------'

        data = []
        f = open(InputFile)
        for line in f:
            line = line.strip()
            data.append([float(v) for v in line.split(',')])
        f.close()
        eY = sess.run(self.y[-1], feed_dict={self.data: data})

#        for i, f in enumerate(data):
#            print 'Features=%s\tEstimate=%f'%(f, eY[i])
#            print '%f'%eY[i]
        with open('%s.prediction.csv'%self.conf['name'], 'a') as f:
            print >> f, '[BEGIN]'
            for y in eY:
                print >> f, y[0]
                print y[0]

        sess.close()


class LinearRegression(Model):
    def __init__(self, conf, feature_size, param=None, loss_file=None):
        Model.__init__(self, 'linear-regression', conf, feature_size, param, loss_file)

        self.y = []
        for i in range(len(self.w[:-1])):
            if conf['activation-function'] == 'tanh':
                if len(self.y) == 0:
                    self.y.append(tf.nn.tanh(tf.matmul(self.data, self.w[i]) + self.b[i]))
                else:
                    self.y.append(tf.nn.tanh(tf.matmul(self.y[-1], self.w[i]) + self.b[i]))
            elif conf['activation-function'] == 'sigmoid':
                if len(self.y) == 0:
                    self.y.append(tf.sigmoid(tf.matmul(self.data, self.w[i]) + self.b[i]))
                else:
                    self.y.append(tf.sigmoid(tf.matmul(self.y[-1], self.w[i]) + self.b[i]))
            elif conf['activation-function'] == 'relu':
                if len(self.y) == 0:
                    self.y.append(tf.nn.relu(tf.matmul(self.data, self.w[i]) + self.b[i]))
                else:
                    self.y.append(tf.nn.relu(tf.matmul(self.y[-1], self.w[i]) + self.b[i]))
            elif conf['activation-function'] == 'linear':
                if len(self.y) == 0:
                    self.y.append(tf.matmul(self.data, self.w[i]) + self.b[i])
                else:
                    self.y.append(tf.matmul(self.y[-1], self.w[i]) + self.b[i])


        if len(self.y) == 0:
            self.y.append(tf.matmul(self.data, self.w[0]) + self.b[0])
        else:
            self.y.append(tf.matmul(self.y[-1],self.w[-1]) + self.b[-1])


        self.loss = tf.reduce_mean(tf.square(self.Y - tf.reshape(self.y[-1],[-1]))) + 0.000*(tf.nn.l2_loss(self.w[0]))
#        self.loss = tf.reduce_mean(tf.square(self.Y - self.y[-1])) + 0.000*(tf.nn.l2_loss(self.w[0]))
        self.diff = self.Y - tf.reshape(self.y[-1],[-1])
        self.optimizer = tf.train.AdamOptimizer(conf['learning-rate'])
        self.d_updates = self.optimizer.minimize(self.loss, var_list=self.params)
