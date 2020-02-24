#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TAGS: ['eng', 'nn', 'nbu', 'con', 'lat', 'num', 'nq', 'jap', 'nc', 'asc', 'hanja']
# CLASS: 0~19
import argparse
import random

def check_word_num_unique(file_path):
  wordTagCnt = {}
  class_cnt = [
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0]
  max = 0
  maxWord = 0
  with open(file_path, 'r') as fp:
    try:
      line = fp.readline()
      cnt = 1
      while line:
        line = line.strip()
        splited = line.split('	')
        classes = splited[1].split(' ')

        for clsNum in classes:
          class_cnt[int(clsNum)] += 1

        raw_data = splited[2].split(' ')
        if len(raw_data) > max:
          max = len(raw_data)

        for wordtag in raw_data:
          word = wordtag.split('/')[0]
          tag = wordtag.split('/')[1]
          if int(word) > maxWord:
            maxWord = int(word)
          if word in wordTagCnt:
            if tag != wordTagCnt[word]:
              print("[tag:%s] is already existed at [word:%s]. [new tag:%s] won't be inserted!!!" % (wordTagCnt[word], word, tag))
          else:
            wordTagCnt[word] = tag

        line = fp.readline()
        cnt += 1

    finally:
      fp.close()
      print("MAX Len: %d" % max)
      print("MAX Word: %d" % maxWord)
      print("Vocab Size: %d" % len(wordTagCnt))
      print(class_cnt)


def devide_file(file_path):
  trainWithWordArr = []
  trainWithTagArr = []
  testWithWordArr = []
  testWithTagArr = []
  with open(file_path, 'r') as fp:
    try:
      line = fp.readline()
      cnt = 1
      while line:
        line = line.strip()
        splited = line.split('	')

        docid = splited[0]
        classes = splited[1]
        raw_data = splited[2].split(' ')

        withWord = withTag = "\""
        for cs in classes.split(' '):
          withWord += cs + ' '
          withTag += cs + ' '

        withWord = withWord.strip()
        withTag = withTag.strip()

        withWord += "\",\""
        withTag += "\",\""

        for wordtag in raw_data:
          word = wordtag.split('/')[0]
          withWord += word + ' '
          tag = wordtag.split('/')[1]
          withTag += tag + ' '
        
        if cnt % 10 == 0:
          testWithWordArr.append(withWord.strip())
          testWithTagArr.append(withTag.strip())
        else:
          trainWithWordArr.append(withWord.strip())
          trainWithTagArr.append(withTag.strip())

        line = fp.readline()
        cnt += 1

    finally:
      fp.close()

  SEED = 321
  train_zip = list(zip(trainWithWordArr, trainWithTagArr))
  random.shuffle(train_zip)
  trainWithWordArr, trainWithTagArr = zip(*train_zip)
  
  with open(".data/aclass/train_word.csv", 'w') as fp:
    try:
      for line in trainWithWordArr:
        fp.write(line)
        fp.write('\"\n')
    finally:
      fp.close()

  with open(".data/aclass/train_tag.csv", 'w') as fp:
    try:
      for line in trainWithTagArr:
        fp.write(line)
        fp.write('\"\n')
    finally:
      fp.close()

  with open(".data/aclass/test_word.csv", 'w') as fp:
    try:
      for line in testWithWordArr:
        fp.write(line)
        fp.write('\"\n')
    finally:
      fp.close()

  with open(".data/aclass/test_tag.csv", 'w') as fp:
    try:
      for line in testWithTagArr:
        fp.write(line)
        fp.write('\"\n')
    finally:
      fp.close()

if __name__=='__main__':

  parser = argparse.ArgumentParser(
        description='Train a text classification model on text classification datasets.')
  parser.add_argument('data_path', help='path for train data')
  args = parser.parse_args()

  data_path = args.data_path
  check_word_num_unique(data_path)
  devide_file(data_path)