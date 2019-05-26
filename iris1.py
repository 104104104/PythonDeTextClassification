import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

#############データの準備################
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data.astype(np.float32)
y=iris.target
N=y.size
y2=np.zeros(3*N).reshape(N,3).astype(np.float32)
for i in range(N):
    y2[i,y[i]]=1.0  #y2が入力ベクトル

index=np.arange(N)
xtrain=x[index[index%2!=0],:]#教師データ
ytrain=y2[index[index%2!=0],:]
xtest=x[index[index%2==0],:]#テストデータ
yans=y[index[index%2==0]]#答え合わせデータ


##########モデルの記述#################
class Irischain(Chain):
    def __init__(self):
        super(Irischain, self).__init__(
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
            )

    def __call__(self, x, y):
        return F.mean_squared_error(self.fwd(x), y)

    def fwd(self,x):
        h1=F.sigmoid(self.l1(x))
        h2=self.l2(h1)
        return h2

####モデルと最適化アルゴリズムの設定#####
model=Irischain()
optimizer=optimizers.Adam()
optimizer.setup(model)

#####学習部分############################
for i in range(5000):
    x=Variable(xtrain)
    print(x.shape)
    y=Variable(ytrain)
    print(y.shape)
    model.cleargrads()   #勾配初期化
    loss=model(x,y)     #誤差計算
    loss.backward()     #勾配計算
    optimizer.update()  #パラメータ更新
    if i%100==0:
        print(i)

#####結果の出力##########################
xt=Variable(xtest)
yt=model.fwd(xt)
ans=yt.data
nrow, ncol=ans.shape
ok=0
for i in range(nrow):
    cls=np.argmax(ans[i,:])
    if cls==yans[i]:
        ok+=1
print(ok, '/', nrow, '=', (ok*1.0)/nrow)
