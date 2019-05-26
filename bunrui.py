import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import MeCab, random

fname_a="a/uwasa.txt"
fname_b="b/nin.txt"

train_size=10

#必要な関数
def make_wakati(text):
    """
    入力された文章を分かち書きする関数
    param: text = 入力する文字列
    return: 分かち書きした単語を要素に持つリスト
    """
    mecab = MeCab.Tagger("-Owakati")
    wakati_str = mecab.parse(text)
    wakati_li = wakati_str.split()
    return wakati_li

#############データの準備############
print("データの読み込み")
dic={}#単語と数字の対応
with open(fname_a, 'r') as f:
    atext=f.read()
    #print(atext)
    ali = make_wakati(atext)
for i in ali:
    if i not in dic:
        dic[i] = len(dic)
atrain=[]
for i in ali:
    atrain.append(dic[i])

with open(fname_b, 'r') as f:
    bli = make_wakati(f.read())
for i in bli:
    if i not in dic:
        dic[i] = len(dic)
btrain=[]
for i in bli:
    btrain.append(dic[i])

alen=len(atrain)
blen=len(btrain)
##########モデルの記述#################
class Bunruichain(Chain):
    def __init__(self):
        super(Bunruichain, self).__init__(
            l1=L.Linear(train_size,50),
            l2=L.Linear(50,25),
            l3=L.Linear(25,2)
            )

    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)

    def fwd(self,x):
        h1=F.sigmoid(self.l1(x))
        h2=F.sigmoid(self.l2(h1))
        h3=self.l3(h2)
        return h3

####モデルと最適化アルゴリズムの設定#####
model=Bunruichain()
optimizer=optimizers.Adam()
optimizer.setup(model)

#####学習部分############################
atrain = np.array(atrain).astype(np.float32)
btrain = np.array(btrain).astype(np.float32)
epoch_count=500
for j in range(epoch_count):
    #aについて
    r = random.choice([0, alen - train_size])
    x=[]
    for i in range(5):
        x.append(atrain[r:r+train_size])
        r = random.choice([0, alen - train_size])
    x=Variable(np.array(x).astype(np.float32))
    y=Variable(np.array([[1,0],[1,0],[1,0],[1,0],[1,0]]).astype(np.float32))#aはゼロ
    model.cleargrads()   #勾配初期化
    loss=model(x,y)     #誤差計算
    loss.backward()     #勾配計算
    optimizer.update()  #パラメータ更新
    r = random.choice([0, blen - train_size])
    x=[]
    for i in range(5):
        x.append(atrain[r:r+train_size])
        r = random.choice([0, alen - train_size])
    x=Variable(np.array(x).float32)
    y=Variable(np.array([[0,1],[0,1],[0,1],[0,1],[0,1]]).astype(np.float32))#bは１
    model.cleargrads()   #勾配初期化
    loss=model(x,y)     #誤差計算
    loss.backward()     #勾配計算
    optimizer.update()  #パラメータ更新
    if j%10 ==0:
        print(j,"/",epoch_count,"回終了")

#####結果の出力##########################
xt=Variable(make_wakati(恥の多い生涯を送って来ました))
yt=model.fwd(xt)
ans=yt.data
print(ans)

#####学習済みデータの保存#####
serializers.save_npz('finished_bunrui.npz', model)
