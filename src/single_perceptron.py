# -*- coding: utf-8 -*-
import csv
from builtins import print

import numpy as np

# ----------------------------------------------
#  定数 Pythonは定数をサポートしないので、大文字で書く
# ----------------------------------------------

# ラベルを含む教師データのカラム数
TEACHER_FACTOR = 5
# 教師データがラベルを格納しているインデックス
INDEX_LABEL = 0
# 学習率(learning coefficient)の設定
LC = 0.01
# 学習の試行回数制限
ITERATION_LIMIT = 1000
# ラベル用定数(Irisの場合:1=setosa,-1=virginica)
LABEL_TRUE = 1
LABEL_FALSE = -1

# ----------------------------------------------
# 教師データ読み込み、最適化
# ----------------------------------------------

# csvファイルのカラムindex_data目までを読み込む
# |がく片長|がく片幅|花びら長|花びら幅|種|
# |Sepal Length|Sepal Width|Petal Length|Petal Width|Species|
fp = open('iris.csv', 'rt')
dataReader = csv.reader(fp)
teacher = np.empty((0, TEACHER_FACTOR), np.float)

# 教師データを配列に格納
# 配列の先頭はラベルとする。数値と内容の関係は以下
#  1: setosa, -1: virginica
# [label, x1, x2, x3, ...]
for row in dataReader:
    label = (LABEL_TRUE if row[TEACHER_FACTOR - 1] == 'setosa' else LABEL_FALSE)
    tmp_row = [[label] + row[0:TEACHER_FACTOR - 1]]
    teacher = np.append(teacher, np.array(tmp_row), axis=0)

# 教師データの型を浮動小数点型へ変更
teacher = teacher.astype(np.float64)

# 教師データの順序をシャッフルする
np.random.shuffle(teacher)

# ----------------------------------------------
# 変数等の初期化
# ----------------------------------------------

# 判別器の初期化。0埋めにした
weight = np.zeros([TEACHER_FACTOR - 1])

# 更新回数の初期化
times_update = 0

# 留年フラグ
# 教師データについて、現在の判別器で誤った判断をした場合はTrueを立てる
need_learning = True

# ----------------------------------------------
# 学習フェイズ
# ----------------------------------------------

# 教師データ全てについてループし、教師データ全てについて、
# 判別器が正しい判定、即ちラベルと同じ符号を返すまで判別器の更新を繰り返す。
# ただし、線形分離不可能な教師データを入力とするケースも有るため、
# 1000回試行して全ての教師データで学習が成功しなかった場合はループを抜ける。
iterator = 0
print('into learning loop.')
while need_learning:
    need_learning = False
    iterator += 1
    print('iterator: ' + str(iterator))
    # 教師データについてループ
    for row in teacher:

        # ループ毎に初期化するもの
        # 更新した判別器
        new_weight = []
        # 更新中の判別器の配列インデックス
        updating_index = 0

        # 教師データ配列の先頭はラベルを格納している
        label = row[INDEX_LABEL]
        # 誤差の算出に使う部分をスライスで切り出し
        x = np.array(row[INDEX_LABEL + 1:TEACHER_FACTOR])

        # 誤差関数 max(lwx)
        # 教師データについて、現在の判別器と内積を求め、誤差を算出.
        # ラベルと同じ符号の誤差が返らない場合は判別失敗、判別器の更新処理へ
        err = label * weight.dot(x)
        print('e: ' + str(err) + '\tlabel: ' + str(label) + '\tx: ' + str(x) + '\tw: ' + str(weight))

        if err <= 0:
            # 判別器の更新処理

            # 判別器が判別に失敗したので、失敗フラグを立てる
            need_learning = True
            times_update += 1

            # 学習
            for w in weight:
                new_weight.append(w + LC * label * x[updating_index])
                updating_index += 1

            print('\tweight: ' + str(weight) + ' -> ' + str(new_weight))
            weight = np.array(new_weight)

    # 留年フラグによるエスケープか、ループ回数超過によるエスケープかを切り分けたいのでループ内にいるうちにアレコレする
    if not need_learning:
        print('learning complete in ' + str(iterator) + ' loop(s) with ' + str(
            times_update) + ' times update. (learning coefficient: ' +str(LC)+ ')')

    if iterator >= ITERATION_LIMIT:
        print('exceed learning loop limit(1000). escape!')
        break

# ----------------------------------------------
# 質問フェイズ
# ----------------------------------------------

# 入力待ちにする
while True:
    print('=============================')
    print('\"I know iris. Ask me !\"')
    print('tell me sepal length:  ')
    sepal_length = input('>>>  ')
    print('tell me sepal width:  ')
    sepal_width = input('>>>  ')
    print('tell me petal length:  ')
    petal_length = input('>>>  ')
    print('tell me petal width:  ')
    petal_width = input('>>>  ')

    print('sepal_length: ' + str(sepal_length))
    print('sepal_width: ' + str(sepal_width))
    print('petal_length: ' + str(petal_length))
    print('petal_width: ' + str(petal_width))

    # 入力したデータについて配列へ変換、判別器を使用して誤差を求める
    guess = np.array(([sepal_length, sepal_width, petal_length, petal_width]), np.float64)
    err_guess = weight.dot(guess)

    # 誤差の符号によって推定する
    # Pythonの三項演算子は慣れるまでキモい
    answer = 'It is \"setosa\", no doubt!' if err_guess > 0 else 'It is \"virginica\", no doubt!'
    print(answer)
