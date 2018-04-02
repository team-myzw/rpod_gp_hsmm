# encoding: utf8
#from __future__ import unicode_literals
import GaussianProcessMiltiDim
import random
import math
import matplotlib.pyplot as plt
import time
import numpy
import os
import gc
import copy
import sympy as sp
from progressbar import ProgressBar

"""
Cythonのコンパイルできないときは，

  E:\Python27_64\Lib\distutils\msvc9compiler.py

のget_build_version()の

  majorVersion = int(s[:-2]) - 6

を使いたいコンパラのバージョンに書き換える．
VC2012の場合は

 majorVersion = 11
"""
class GPSegmentation():
    MAX_LEN = 20
    MIN_LEN = 3
    AVE_LEN = 13
    SKIP_LEN = 1

    CORD_TRA = 0
    CORD_LAND = 1
    CORD_MOV = 2
    NUM_CORD = 3

    def __init__(self, dim, nclass, cordinates):
        self.dim = dim
        self.numclass = nclass
        self.segmlen = 3
        self.gps = [GaussianProcessMiltiDim.GPMD(dim)
                    for i in range(self.numclass)]
        self.cordinates = cordinates
        self.segm_in_class = [[] for i in range(self.numclass)]
        self.segmclass = {}
        self.segmlandmark = {}
        self.segments = []
        self.landmarks = []
        self.landmark_lists = []
        self.trans_prob = numpy.ones((nclass, nclass))
        self.trans_prob_bos = numpy.ones(nclass)
        self.trans_prob_eos = numpy.ones(nclass)
        self.is_initialized = False
        self.land_choice = []

    def first_land(self, segm, lands):
        n = len(lands)
        length_list = [0.] * n
        for i in range(n):
            len_sum = 0.0
            for s in segm:
                length = numpy.sqrt(numpy.power(s[0] - lands[i][0],2)+numpy.power(s[1] - lands[i][1],2)+numpy.power(s[2] - lands[i][2],2))
                len_sum += length
            length_list[i] = len_sum
        num = numpy.argmin(length_list)            

        return lands[num]

    def load_data(self, traj_filenames, land_filenames, classfile=None):
        self.data = []
        self.segments = []
        self.is_initialized = False
#       参照点の座標位置の登録
        for land_names in land_filenames:
            land_list = []
            for land in land_names:
                land_list.append(numpy.loadtxt(land))
            self.land_choice.append(range(len(land_list)))
            self.landmark_lists.append(land_list)

#       初期分節位置の決定
        for k in range(len(traj_filenames)):
            y = numpy.loadtxt(traj_filenames[k])
            segm = []
            self.data.append(y)


            i = 0
            while i < len(y):
                length = random.randint(self.MIN_LEN, self.MAX_LEN)

                if i + length + 1 >= len(y):
                    length = len(y)-i

                segm.append(y[i: i+length + 1])

                i += length

            self.segments.append(segm)

            # ランダムに割り振る
            for i, s in enumerate(segm):
                c = random.randint(0, self.numclass -1)
                self.segmclass[id(s)] = c
#               初期参照点座標の決定。分節に最も近い点を最初は割り振る
                self.segmlandmark[id(s)] = self.first_land(s,self.landmark_lists[k])


# 遷移確率更新
        self.calc_trans_prob()

    def normlize_time(self, num_step, max_time):
        step = float(max_time)/(num_step+1)
        time_stamp = []

        for n in range(num_step):
            time_stamp.append((n + 1) * step)

        return time_stamp

    def normalize_samples(self, d, nsamples):
        if len(d) == 1:
            return numpy.ones(nsamples) * d[0]
        else:
            return numpy.interp(range(nsamples),
                                numpy.linspace(0, nsamples - 1, len(d)), d)

    def load_model(self, basename):
        # GP読み込み
        for c in range(self.numclass):
            filename = os.path.join(basename, "class%03d.npy" % c)
            self.segm_in_class[c] = [s for s in numpy.load(filename)]

            landmarks = numpy.load(os.path.join(basename,
                                                "landmarks%03d.npy" % c))

            for s, l in zip(self.segm_in_class[c], landmarks):
                self.segmlandmark[id(s)] = l

            self.update_gp(c)

        # 遷移確率更新
        self.trans_prob = numpy.load(os.path.join(basename, "trans.npy"))
        self.trans_prob_bos = numpy.load(os.path.join(basename,
                                                      "trans_bos.npy"))
        self.trans_prob_eos = numpy.load(os.path.join(basename,
                                                      "trans_eos.npy"))

        
    def update_gp(self, c):
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            s = self.cordinate_transform(s, self.segmlandmark[id(s)],
                                         self.cordinates[c])

            datay += [y for y in s]
            datax += range(len(s))

        self.gps[c].learn(numpy.array(datax), datay)

    def sample_class(self, landmark, segm):
        prob = []

        for c, gp in enumerate(self.gps):
            slen = len(segm)
            plen = 1.0
            if len(segm) > 2:
                plen = (self.AVE_LEN**slen * math.exp(-slen) /
                        math.factorial(self.AVE_LEN))

                cord = self.cordinates[c]
                s = self.cordinate_transform(segm, landmark, cord)
                p = gp.calc_lik(range(len(s)), s, self.MAX_LEN)
                prob.append((math.exp(p) * plen))
            else:
                prob.append(0)


        accm_prob = [0]*self.numclass
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i, prob[i]

    def calc_output_prob(self, c, segm, landmark):
        gp = self.gps[c]
        slen = len(segm)
        plen = 1.0
        if len(segm) > 2:
            plen = (self.AVE_LEN**slen * math.exp(-slen) /
                    math.factorial(self.AVE_LEN))
            cord = self.cordinates[c]
            s = self.cordinate_transform(segm, landmark, cord)
            p = gp.calc_lik(range(len(s)), s, self.MAX_LEN)
            return sp.exp(p) * plen
        else:
            return 0


    def save_model(self, basename):
        for n, segm in enumerate(self.segments):
            classes = []
            clen = []
            for s in segm:
                c = self.segmclass[id(s)]
                classes += [c for i in range(len(s))]
                clen.append([c, len(s)])
            numpy.savetxt(basename+"segm%03d.txt" % n, classes, fmt="%d")
            numpy.savetxt(basename+"slen%03d.txt" % n, numpy.array(clen), fmt="%d")
        plt.figure()
        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.subplot(self.dim, self.numclass, c + d * self.numclass + 1)
                for data in self.segm_in_class[c]:
                    trans_data = self.cordinate_transform(data,
                                                          self.segmlandmark[id(data)],
                                                          self.cordinates[c])

                    if self.dim == 1:
                        plt.plot(range(len(trans_data)), trans_data, "o-")
                    else:
                        plt.plot(range(len(trans_data)),
                                 trans_data[:, d], "o-")
                    plt.ylim(-1.1, 1.1)

        plt.savefig(basename+"class.png")

        # テキストでも保存
        numpy.save(basename + "trans.npy", self.trans_prob)
        numpy.save(basename + "trans_bos.npy", self.trans_prob_bos)
        numpy.save(basename + "trans_eos.npy", self.trans_prob_eos)
        numpy.savetxt(basename + "lik.txt", [self.calc_lik()])

        for c in range(self.numclass):
            numpy.save(basename+"class%03d.npy" % c,
                       self.segm_in_class[c])
            numpy.save(basename+"landmarks%03d.npy" % c,
                       [self.segmlandmark[id(s)]
                        for s in self.segm_in_class[c]])

        numpy.save(basename + "cordinates.npy", self.cordinates)

        
    def cordinate_transform(self, s, land_pos, cord):
        if cord == self.CORD_TRA:
#       手先基準の軌道に変換
#       座標の基準は可変にしたい
#       手先座標系は、軌道の中心を基準とする。
#       z軸(高さ)は変換を行わない
#       返還前の軌道はWorld座標系
            ss = numpy.array(s).T
            offset = numpy.zeros(len(ss))
            for i in range(len(ss)):
                offset[i] = ss[i].sum()
            offset = offset / float(len(s))                
            if len(offset) ==4:
                offset[3] = 0.0
                offset[2] = s[0][2]
            ss = s - offset

            t = -math.atan2(ss[0][1],ss[0][0])
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            v = []

            for i, sss in enumerate(ss):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                h = sss[3]
                v.append([x, y, z, h])
            ss = numpy.array(v)

            if ss[1][1] < 0:
                for i in range(len(ss)):
                    ss[i][1] *= -1

        elif cord == self.CORD_LAND:
#       物体基準の軌道に変換
#       座標の基準は可変にしたい
#       参照点位置は事前に与える
            ss = s - land_pos
#           座標変換の為の回転量を計算
            t = -math.atan2(ss[0][1], ss[0][0])
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            v = []
            for i, sss in enumerate(ss):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                h = sss[3]
                v.append([x, y, z, h])

            t = -math.atan2(ss[0][2], numpy.sqrt(ss[0][0]**2 + ss[0][1]**2))
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            vv = []
            for i, sss in enumerate(v):
                x = R[0][0] * sss[0] + R[0][1] * sss[2]
                z = R[1][0] * sss[0] + R[1][1] * sss[2]
                y = sss[1]
                h = sss[3]
                vv.append([x, y, z, h])
            ss = numpy.array(vv)

        elif cord == self.CORD_MOV:

            offset = copy.copy(s[0])
            
            ss = s - offset

            t = -math.atan2(ss[-1][1] - ss[0][1], ss[-1][0] - ss[0][0])
            R = numpy.array([[numpy.cos(t), -numpy.sin(t)],
                             [numpy.sin(t), numpy.cos(t)]])
            v = []

            for i, sss in enumerate(ss):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                h = sss[3]
                v.append([x, y, z, h])
            ss = numpy.array(v)



        return ss
        
    def forward_filtering(self, d, ii):
        T = len(d)
        a = numpy.zeros((len(d), self.MAX_LEN, self.numclass),dtype="object")  # 前向き確率
        ll = numpy.zeros((T, self.MAX_LEN, self.numclass))
        p_bar = ProgressBar(max_value=T)
        for t in range(T):
            p_bar.update(t+1)
            for k in range(self.MIN_LEN, self.MAX_LEN, self.SKIP_LEN):
                if t-k < 0:
                    break

                for c in range(self.numclass):
                    out_prob = sp.numbers.Float(0.0)
                    lll = 0
                    for iii in range(len(self.landmark_lists[ii])):
                        calc_prob = self.calc_output_prob(c, d[t-k:t+1], self.landmark_lists[ii][iii])
                        if calc_prob >= out_prob:
                            out_prob = calc_prob
                            lll = iii
                        ll[t,k,c] = lll
                    # 遷移確率
                    tt = t-k-1
                    if tt >= 0:
                        for kk in range(self.MAX_LEN):
                            for cc in range(self.numclass):
                                a[t, k, c] += sp.numbers.Float((a[tt, kk, cc] *
                                               self.trans_prob[cc, c]))
                        a[t, k, c] *= out_prob
                    else:
                        # 最初の単語
                        a[t, k, c] = sp.numbers.Float(out_prob * self.trans_prob_bos[c])
                    try:
                        a[t, k, c] = a[t, k, c].evalf()
                    except:
                       pass 
                    if t == T - 1:
                        # 最後の単語
                        a[t, k, c] *= self.trans_prob_eos[c]

                    if math.isnan(a[t, k, c]):
                        print "nanananananan"
                        print t, k, tt, kk
                        if tt >= 0:
                            for kk in range(self.MAX_LEN):
                                print(cc, tt, kk, a[tt, kk],
                                      self.trans_prob[cc, c])
                            print "a", a[tt, kk], out_prob
                        else:
                            # 最初の単語
                            print "a", a[t, k], out_prob
                        raw_input()

        for t in range(T):
            for k in range(self.MIN_LEN, self.MAX_LEN, self.SKIP_LEN):
                if t-k < 0:
                    break

        return a, ll

    def sample_idx(self, prob):
        accm_prob = [0, ] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]
        accm_prob = numpy.array(accm_prob)
        accm_prob = accm_prob / accm_prob[-1] * 100.0
        rnd = numpy.random.uniform(0.0, accm_prob[-1])
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i

    def backward_sampling(self, a, d, ll):
        T = a.shape[0]
        t = T-1

        segm = []
        segm_class = []
        land = []
        print ""
        while True:
            idx = self.sample_idx(a[t].reshape(self.MAX_LEN * self.numclass))
            l = ll[t].reshape(self.MAX_LEN * self.numclass)[idx]
            k = int(idx/(self.numclass))
            c = int(idx % self.numclass)

            s = d[t-k:t+1]

            # パラメータ更新

            segm.insert(0, s)
            segm_class.insert(0, c)
            land.insert(0, l)
            t = t-k-1

            if t <= 0:
                break
#        raw_input()
        print "class",
        print segm_class
        return segm, segm_class, land

    def calc_trans_prob(self):
        self.trans_prob = numpy.zeros((self.numclass, self.numclass))
        self.trans_prob += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments):
            for i in range(1, len(segm)):
                try:
                    cc = self.segmclass[id(segm[i-1])]
                    c = self.segmclass[id(segm[i])]
                except KeyError, e:
                    # gibss samplingで除かれているものは無視
                    break
                self.trans_prob[cc, c] += 1.0

        # 正規化
        self.trans_prob = (self.trans_prob /
                           self.trans_prob.sum(1).reshape(self.numclass, 1))

    def remove_ndarray(self, lst, elem):
        l = len(elem)
        for i, e in enumerate(lst):
            if len(e) != l:
                continue
            if id(e) == id(elem):
                lst.pop(i)
                return
        raise ValueError("ndarray is not found!!")

    def learn(self):
        if not self.is_initialized:
            # GPの学習
            for i in range(len(self.segments)):
                for s in self.segments[i]:
                    c = self.segmclass[id(s)]
                    self.segm_in_class[c].append(s)

            # 各クラス毎に学習
            for c in range(self.numclass):
                self.update_gp(c)

            self.is_initialized = True

        self.update(True)

    def recog(self):
        self.update(False)

    def update(self, learning_phase=True):
        cls = [0] * self.numclass
        for i in range(len(self.segments)):
            d = self.data[i]
            segm = self.segments[i]
            for s in segm:
                c = self.segmclass[id(s)]
                self.segmclass.pop(id(s))
                self.segmlandmark.pop(id(s))
                if learning_phase:
                    self.remove_ndarray(self.segm_in_class[c], s)

            if learning_phase:
                # GP更新
                start = time.clock()
                print "update1"
                for c in range(self.numclass):
                    self.update_gp(c)

                # 遷移確率更新
                print time.clock()-start, "sec"
                self.calc_trans_prob()


            start = time.clock()
            print "forward...",
            a, ll = self.forward_filtering(d, i)

            print "backward...",
            segm, segm_class, land = self.backward_sampling(a, d, ll)
            print time.clock()-start, "sec"



            self.segments[i] = segm

            for s, c, ll in zip(segm, segm_class, land):
                self.segmclass[id(s)] = c
                self.segmlandmark[id(s)] = self.landmark_lists[i][int(ll)]
                cls[c] += 1
                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[c].append(s)


            print " [",
            for s in self.segm_in_class:
                print len(s),
            print "]"

            print(i, u"/", len(self.segments))

            if learning_phase:
                # GP更新
                start = time.clock()
                for c in range(self.numclass):
                    self.update_gp(c)

                # 遷移確率更新
                self.calc_trans_prob()
                print cls
        return

    def calc_lik(self):
        lik = 0
        for segm in self.segments:
            for s in segm:
                c = self.segmclass[id(s)]
                lik += self.gps[c].calc_lik(range(len(s)), s, self.MAX_LEN)

        return lik

def learn(savedir, dim, nclass, cordinates, N):
    I =N
    gpsegm = GPSegmentation(dim, nclass, cordinates)
#   軌道を読み込む
    tra_files = ["test/joint_data{0:d}.txt".format(i)
                 for i in range(I)]

    land_files = []
    for i in range(I):
        file_name = []
        j = 0
        while 1:
            try:
#               参照点を記したがファイルを読み込む
                f = open("test/land{0:d}_{1:d}.txt".format(i,j))
                f.close()
                file_name.append("test/land{0:d}_{1:d}.txt".format(i,j))
            except:
                break
            j += 1
        land_files.append(file_name)
    gpsegm.load_data(tra_files, land_files)

    for it in range(10):
        print "-----", it, "-----"
        gpsegm.learn()
        print "lik =", gpsegm.calc_lik()

    print("now saving")
    gpsegm.save_model(savedir)

    return gpsegm.calc_lik()

def recog(modeldir, savedir, dim, nclass, cordinates,N):
    I =N

    gpsegm = GPSegmentation(dim, nclass, cordinates)

    gpsegm.load_model(modeldir)

    tra_files = ["test/joint_data{0:d}.txt".format(i)
                 for i in range(I)]

    land_files = []
    for i in range(I):
        file_name = []
        j = 0
        while 1:
            try:
                f = open("test/land{0:d}_{1:d}.txt".format(i,j))
                f.close()
                file_name.append("test/land{0:d}_{1:d}.txt".format(i,j))
            except:
                break
            j += 1
        land_files.append(file_name)

    gpsegm.load_data(tra_files, land_files)

    for it in range(10):
        print "-----", it, "-----"
        gpsegm.learn()
        print "lik =", gpsegm.calc_lik()
    gpsegm.save_model(savedir)

    print "lik", gpsegm.calc_lik()
def make_dir(name):
    try:
        os.mkdir(name)
    except:
        pass

def main():

    N = 55
    name = "503/"
    cor = [1,1,1,0,0]
    make_dir(name)
#    try:
    learn(name, 4, 5, cor, N)
#    except:
#        os.rmdir(name)
    gc.collect()
    return
  

if __name__ == '__main__':
    main()
