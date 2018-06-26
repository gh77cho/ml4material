#!/usr/bin/python

import sys
import json
import re
import string
import math

def load_field_definition(jsonfile):
    f = open(jsonfile)
    FieldDefinition = json.loads(f.read())
    f.close()

    if 'Input' not in FieldDefinition:
        print >> sys.stderr, 'error: [FIELD DEFINITION FILE] "Input" is required'
        sys.exit(2)
    if 'Target' not in FieldDefinition:
        print >> sys.stderr, 'error: [FIELD DEFINITION FILE] "Target" is required'
        sys.exit(2)
    if len(FieldDefinition['Input']) <= 0:
        print >> sys.stderr, 'error: [FIELD DEFINITION FILE] "Input" list is required'
        sys.exit(2)
    if len(FieldDefinition['Target']) <= 0:
        print >> sys.stderr, 'error: [FIELD DEFINITION FILE] "Target" list is required'
        sys.exit(2)

    for f in FieldDefinition['Input']:
        if re.search(' x', f):
            print >> sys.stderr, 'error: [FIELD DEFINITION FILE] invalid "Input" filed name "%s". Remove space' % f
            sys.exit(2)
        if any(char in set(string.punctuation.replace('_', '')) for char in f):
            print >> sys.stderr, 'error: [FIELD DEFINITION FILE] invalid "Input" filed name "%s". Remove special character.' % f
            sys.exit(2)

    for f in FieldDefinition['Target']:
        if re.search(' x', f):
            print >> sys.stderr, 'error: [FIELD DEFINITION FILE] invalid "Target" filed name "%s". Remove space' % f
            sys.exit(2)
        if any(char in set(string.punctuation.replace('_', '')) for char in f):
            print >> sys.stderr, 'error: [FIELD DEFINITION FILE] invalid "Target" filed name "%s". Remove special character.' % f
            sys.exit(2)

    return FieldDefinition

def load_split_data(splitFile):
    featureList = []
    f = open(splitFile)
    lineno = 0
    n = 0
    for line in f:
        lineno += 1
        line = line.strip()
        if len(line) == 0:
            featureList.append(None)
        else:
            n += 1
            featureList.append(float(line))
    f.close()

    print >> sys.stderr, '\t- Load Split File : %s (%d / %d = %2.f%%)' % (splitFile, n, lineno, 100.*n/lineno)

    return featureList

def get_stat(X):
    v_sum = 0.
    v_sum2 = 0.
    n = 0
    for v in X:
        if v == None:
            continue
        v_sum += v
        v_sum2 += v*v
        n += 1
    avg = float(v_sum) / float(n)
    std = math.sqrt(float(v_sum2) / float(n) - avg*avg)
    return avg, std

def pearson (X, Y):
    avg_X, std_X = get_stat(X)
    avg_Y, std_Y = get_stat(Y)

    if avg_X == 0.:
        return None

    sum_co = 0.
    sum_x = 0.
    sum_y = 0.
    for i, x in enumerate(X):
        y = Y[i]
        if x == None or y == None:
            continue
        dx = x-avg_X
        dy = y-avg_Y

        sum_co += dx*dy
        sum_x += dx*dx
        sum_y += dy*dy

    if sum_x == 0. or sum_y == 0.:
        return None

    return float(sum_co) / math.sqrt(sum_x) / math.sqrt(sum_y)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s FieldDefinitionJsonFile\n' % sys.argv[0])
        sys.exit(1)
    jsonfile = sys.argv[1]
    FieldDefinition = load_field_definition(jsonfile)

    print FieldDefinition
