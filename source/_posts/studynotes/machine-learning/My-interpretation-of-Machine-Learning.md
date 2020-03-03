---
title: Machine Learning
toc: true
date: 2020-03-01 08:51:55
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# My Interpretation of Machine Learning

통계학을 분류하는 방법으로는 여러 가지가 있지만, 다음과 같이 통계학을 분류할 수도 있다.

- **Frequentist Statistics**
- **Bayesian Statistics**

각 통계학에서 machine learning의 방법론은 약간씩 차이가 있는 듯 하다. 하지만, 기본적으로 각 통계학의 궁극적인 목표 중 하나는 우리가 모르는, sample space distribution을 최대한 추정하는 것이다.

대충, 추정하는 방법은 먼저, 우리가 수집한 데이터셋 $D$의 likelihood를 최대화 하는 분포가 진짜 sample space를 가장 유사하게 추정하는 방법이라고 가정한다.

다시 한번 반복하자면, 통계학의 목표중 하나는 우리가 모르는, sample space의 분포를 추정하는 것이다.



### Machine Learning in Frequentist Statistics

Frequentist statistics에서는 많은 데이터 샘플을 뽑는 시행을 한 후, 데이터를 이용해서 모분포를 추정하는 것이라는 목표가 있다. 이것을 하기 위해 데이터셋을 바탕으로 확률 분포가 대충 어떻게 생겼을지 모델링하게 되며, 이때, 확률 분포를 모델링하는데 쓰이는 것이 바로 **머신러닝**이다.

데이터셋 $D$가 있다고 하자. 이 데이터셋에 있는 각 샘플들 $d_i$들은 어쨌든 모 분포(sample space distribution)에서 나올 확률이 어느 정도 높으니까 샘플링되어 우리 손에 들어왔을 것이다. Frequentist statistics에서는 바로 이것에 주목한다. 우리가 모은 데이터셋은 모 분포를 반영해서, 확률이 높은 애들이 많이 뽑히고 낮은 애들이 적게 뽑힐 것이다. 따라서, 우리가 모은 데이터셋이 뽑혀왔을 확률, 즉, likelihood를 최대화 하는 분포를 찾는다면, 이 분포가 모분포와 매우 유사할 것이라고 가정한다.

그 전에, 조건이 있다.

1. 각 데이터 샘플 $d_i$는 반드시 모든 샘플과 동일한 분포에서 샘플링되어야 한다. 즉, 서울에서 온도측정하고 북극가서 온도 측정하면 안 된다. -> identical
2. 각 데이터 샘플 $d_i$는 모두 독립적인 시행으로 인한 샘플링이어야 한다. 즉, 첫 번째 동전던지기가 세 번째 동전던지기에 영향을 주지 않는다. 이와 같이, 이 조건을 자동으로 만족시키는 샘플링도 있다. -> independent

위 두 가지 조건을 합쳐서, "각 데이터 샘플 $d_i$는 **iid**(Identical independent distribution) 하에서 샘플링 되어야 한다"고 말한다.

iid를 만족시킴으로써, 우리는 하나의 모분포를 추정하기만 하면 된다. 즉, 한 분포의 파라미터 $\theta$를 추정하기만 하면 된다.

iid를 만족시키는 시행으로 얻어진 데이터셋의 joint distribution, 즉, 데이터셋을 얻었을 확률을 다음과 같이 계산할 수 있다.

$$
P(D|\theta) = P(d_1, d_2,...,d_n|\theta) = P(d_1|\theta)P(d_2|\theta)\cdots P(d_n|\theta)​
$$


#### Maximum Likelihood Estimation

앞서 말했듯이, frequentist statistics에서는 모분포(sample space distribution)을 추정하기 위해서, likelihood를 모델링하고 likelihood를 최대화 하는 확률분포를 계산한다. 이 확률분포가 모분포의 추정이 된다. Likelihood라고 함은, 우리의 데이터 셋이 샘플링되어왔을 확률이라고 보면 된다.

$$
\text{Likelihood Dist.} = P(d_1|\theta)P(d_2|\theta)\cdots P(d_n|\theta) = P(D|\theta)
$$
여기서 $\theta$는 모분포의 파라미터이다.

이 likelihood를 최대화하는 분포를 찾는다면, 즉, 모분포의 파라미터 $\theta$의 추정값 $\hat{\theta}$를 찾는다면, 이 분포가 모분포와 유사할 것이다 라는 것이다.

$$
\hat{\theta} = argmax_{\theta} P(D|\theta)​
$$
이와 같이 likelihood를 최대화 시키는 분포 파라미터 $\hat{\theta}$를 찾고, 그것을 파라미터로 하는 분포는 모분포와 가깝다는 것이 frequentist statistics에서 모분포를 추정하는 대표적인 방법이다. 이것을 MLE(Maximum likelihood estimation)라고 부른다.

하지만, 어떻게 최대화 시킬까? 무언가를 최대화 최소화시키는데 가장 먼저 떠오르는건 미분을 통해 극점을 찾는 것이다. 하지만, likelihood가 주어지지 않아서 미분또한 할 수 없다. 따라서 우리는 먼저 likelihood를 모델링해야한다. 이것은 잠시 후에 설명한다.



#### How to Maximize Likelihood

Likelihood를 최대화하는 분포를 구한다면, 그것이 모분포와 비슷해질 것이라는 것은 알겠다. 그렇다면, 어떻게 최대화시키는지 알아야 할 것이다. 

그에 앞서서 동전을 100번 던지는 시행을 예로 들자. 우리는 동전을 100번 던졌을 때, 앞면이 몇번 나올까에 대한 확률 분포를 추정하고 싶다. 이때, **각 시행은 동전 1번 던지는게 아니라 100번 던지는게 1번의 시행이다.** 우리는 시행 1회에 대한 확률을 먼저 모델링해야 한다. 이것은 binomial distribution으로 모델링할 수 있을 것이다.

$$
\text{i-th experiment} = P(d_i|\theta) = \begin{pmatrix} 100 \\ n_i \end{pmatrix} \theta^{n_i} (1 - \theta)^{100-n_i}​
$$
$i$번째의 시행에서는 100번 동전을 던지고 $n_i$회의 앞면이 나왔다. 그리고 동전이 앞면이 나올 확률은 $\theta$이다.

모든 시행은 iid조건을 만족한다면, identical한 distribution에서 샘플링된 데이터이므로 모든 시행에서 $\theta$는 같다. 이 시행을 1000회 해서 1000개의 데이터를 모았다고 가정한다. 그럼 likelihood는 이들을 곱한 것이다.

$$
P(D|\theta) = P(d_1|\theta)P(d_2|\theta)\cdots P(d_1000|\theta)
$$
이때, likelihood는 파라미터 $\theta$에 대한 함수가 된다. Likelihood는 이처럼 분포 파라미터의 함수가 된다. 그런데, 동전던지기는 우리가 미리 잘 알고있다시피 베르누이 시행이고, 이들을 100번 던졌을때 앞면이 몇번 나올까에 대한 것은 binomial distribution을 따른다. 따라서 **시행 1회를 binomial distribution으로 모델링할 수 있었지만, 일반적으로 데이터 샘플링을 모델링할때는 무슨 distribution으로  모델링을 해야 할지 알 수 없다.**  이때 등장하는게 바로 Machine Learning이다. Machine Learning은 이 "시행"을 모델링하는데 사용한다. 더 나아가 likelihood를 모델링하는데 사용한다!

Likelihood를 모델링했다면, 이 likelihood는 우리가 추정하고자 하는 parameter인 $\theta$에 대해 미분이 가능해진다(parameter인 $\theta$를 구한다는 것은 likelihood를 추정하는 것이다. $\theta$는 likelihood를 나타내는 파라미터이기 때문이다.). 미분이 가능하다면, $\theta$에 따른 likelihood의 극점을 찾을 수 있다는 것이다. 많은 분들이 아시다시피 다음 조건을 만족할때, 극점이라고 부른다.

$$
\frac{dP(D|\theta)}{d\theta} = 0​
$$
하지만, 이 식은 극점을 가르처주지만, 그 점이 극대인지, 극소인지 가르쳐주지는 않는다. 이 것을 해결하기 위한 것이 **gradient(기울기)**이다.



#### Negative Log Likelihood(NLL) & Gradient Descent

흔히, 우리는 gradient descent를 machine learning 알고리즘의 최적화 방법론으로 알고 있다. 그런데 이것이 왜 machine learning 알고리즘을 최적화 할 수 있는 것인지 알아보려고 한다.

likelihood를 모델링했고, 극대점을 찾아야 한다는 것도 알았다. 그런데, 단순히 미분값이 0인 $\hat{\theta}$를 찾는 것만으로는 극대인지, 극소인지 알 수 없다. 이때 사용하는 것이 gradient인데, 방법은 다음과 같다.

1. 일단 $\theta$를 임의로 초기화한다.

2. 현재 $\theta$값에 대해서 likelihood를 미분해본다.

3. 미분해서 나온 값**(gradient라고 부른다)**의 부호가 (+)이라는 의미는 $\theta$를 증가시키면 likelihood가 증가한다는 의미이다.

   ![image-20200303084924874](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200303084924874.png)

4. 반대로, gradient가 (-)부호라는 것은 $\theta$를 감소시켜야 likelihood가 증가한다는 의미이다.

   ![image-20200303084905099](../../../../../../Notes/note-images/Principle_of_Machine_Learning/image-20200303084905099-1583192979660.png)

따라서 likelihood의 완전한 maximum 지점을 찾을 수는 없더라도 현재 위치에서 어느 방향으로 가야 likelihood를 증가시키는지 알 수 있다. gradient를 이용해서 likelihood 산을 오른다는 느낌으로, gradient ascent 라는 용어가 있을 수 있겠다. 하지만, 문제가 있다. Likelihood는 확률을 데이터 샘플 수만큼 곱한 것이다. 즉, 매우매우 0에 가까운 값으로, 일반적으로 데이터 샘플 수는 1만개, 10만개가 넘어가는 경우도 많다. 이들을 다 곱하면 컴퓨터에게는 그냥 0이다. 따라서 미분을 하기도 전에 이미 likelihood는 표현조차 불가능하다.

이것을 해결하기 위해 likelihood에 로그를 씌워서 log likelihood를 만든다. 확률값은 0과 1 사이값이므로, log를 씌우면 상당히 절댓값이 큰 (-)값이 나올 가능성이 높다. 즉, 너무너무 작아서 컴퓨터로 표시되지 않는 문제를 해결할 수 있다. 또한, log를 씌운다고 해서 씌우기 전에 대소관계가 씌운 후에 바뀐다거나 하지 않는다. (log함수는 monotonically 증가하는 함수이다.) 다른 의미로, log likelihood를 최대화하는 $\hat{\theta}$와 likelihood를 최대화하는 $\hat{\theta}$가 같다.

좋은 예시로, $y=-(x-1)^2+1$를 최대화 하는 x값이나, $\text{log}[y] = \text{log}[-(x-1)^2+1]$를 최대화하는 x값은 똑같이 1이다.

그런데, likelihood는 확률이므로, 0과 1사이 값이다. 따라서 log를 씌우면 무조건 0보다 작거나 같다. 이 모양이 조금 이상하니까, -1를 곱해서 negative log likelihood를 도입한다. 즉 다음과 같다.

$$
\text{NLL} = -\text{log}[P(D|\theta)]​
$$
그런데, 정보 이론을 공부해보신 분이라면 어디서 많이 본 모양일 것이다. 정보 이론에서 entroy와 cross entropy라는 개념이 있다. 다음과 같다.

$$
\text{Entropy} = \mathbb{E}_p[-\text{log}~p], \\

\text{Cross Entropy} = \mathbb{E}_p[-\text{log}~q]​
$$
**정보 이론에서 엔트로피란, 불확실성의 높고 낮음을 나타낸다.** (다른 의미로 정보량이 적고 많음을 의미한다) 즉, 확률 분포 p가 uniform distribution과 같이 뭐가 샘플링될지 전혀 알 수 없을수록, 엔트로피는 증가한다. 반대로, 어느 지점에서 확률이 매우 높은(분산이 매우 작은 normal distribution을 떠올리자) 분포는 우리가 어느 정도 뭐가 나올지 알고, 여러개 샘플링해보면 데이터의 다양성이 떨어진다. 따라서 엔트로피가 감소한다.

**Cross entropy란, 어느 분포 p에 대해서 다른 분포 q의 기댓값이다.** 무슨 의미냐면, p분포와 q분포가 많이 다르게 생기면 생길수록 cross entropy가 증가한다. 반면, p분포와 q분포가 비슷하게 생길수록 cross entropy가 감소한다. 이는 두 분포간의 거리를 계산한다는 KL-divergence의 개념과도 거의 유사하며, 사실상 다 이어져 있는 개념이다. KL-divergence는 실제로 entropy와 cross entropy의 합이다.

왜 이 개념을 말했냐면, negative log likelihood에서, 모분포를 $\mathbb{P}$라고 했을 때, 우리 손에 들어온 모든 데이터 샘플이 나왔을 확률을 모두 같다고 가정해보자. 왜 모두 같다고 해도 되냐면, 데이터 샘플에는 중복되어 샘플링 된 샘플이 많을 것이다. 동전 던지기를 10번 해서 앞면이 7번 나왔다면, 그 10번 시행에 모두 같은 비중치를 뒀지만, 다 더해보면 앞면의 비중치는 0.7, 뒷면의 비중치는 0.3으로, 많이 샘플링된 얘들은 높은 비중치를 가지게 된다. 따라서, 모든 데이터 샘플들에 비중치를 같다고 둬도, 중복된 샘플들 덕분에 실제로 비중치는 데이터 확률 분포를 반영하게 되는 것이다.

그렇게 되면 각 비중치 $\alpha$를 통해 다음과 같이 표현될 수 있다.

$$
\text{NLL} = \Sigma_i-\alpha\text{log}P(d_i|\theta) = [\text{예시}]: \Sigma_i[-0.1*\text{log}\theta] = -0.7*\text{log}\theta - 0.3*\text{log}(1-\theta)
$$
각 비중치 위 모양은 cross entropy와 정확히 일치한다. 참고로 $\alpha$를 곱해준다고 해서 likelihood를 최대화시키는 $\hat{\theta}$값은 변하지 않는다. $-(x-1)^2$를 최대화 시키는 x나, $-0.1*(x-1)^2$을 최대화 시키는 x는 모두 1이다. 같은 이유이다.

따라서, negative log likelihood는 cross entropy라고도 부른다.

종합해보면 다음 문장들은 모두 같은 의미이다.

- Likelihood를 최대화 시키는 파라미터를 구한다.
- Log likelihood를 최대화 시키는 파라미터를 구한다.
- Negative log likelihood를 최소화 시키는 파라미터를 구한다.
- Cross entropy를 최소화 시키는 파라미터를 구한다.

**그리고, likelihood를 최대화 시키기 위해서 gradient ascent를 해야 했지만, negative로 만듦으로써, 이번엔 negative likelihood를 하강해야 하므로, gradient descent를 해야 하는 것이다.**

하지만, 보다시피, gradient descent 방법으로는 theta를 증가시킬지, 감소시킬지는 가르쳐주지만, 어느정도 감소시켜야 할지, 증가시켜야 할지는 알려주지 않는다. 따라서 theta를 조금씩 증감하면서 gradient가 0이 되는 극점을 찾는 것이 gradient descent optimization이고, 극점을 찾는 과정을 training/learning(학습)이라고 부른다.



#### Machine Learning in Frequentist Statistics

앞서 말했듯이, 데이터 샘플을 뽑는 1회 시행은 우리가 알고 있는 normal distribution이나, bernoulli distribution같은 것이 아닐 가능성이 있다. 따라서 우리는 유연하게 "시행"을 모델링해야 할 필요가 있다.

Machine learning이라고 하면, weights $w$로 parameterize되며, 데이터 샘플 하나 $x$를 입력으로 받으면, 그 데이터 샘플이 어느 부류인지에 대한 확률 $P(x|w)$을 계산한다. 즉, machine learning이라는 이름의 방법론 속에 숨어있는 확률 분포로부터 입력으로 넣은 데이터 샘플이 뽑힐 확률을 계산하는 것이다.

이들을 모든 데이터 샘플들에 반복해서 모두 곱하면,

$$
P(x_1|w)P(x_2|w)\cdots P(x_n|w) = P(X|w)​
$$
즉,  likelihood를 모델링한 것이다. machine learning은 데이터 샘플 1개가 샘플링되는 모 확률 분포라고 가정하고, 그 모 확률분포를 모델링한 것에 지나지 않는다.



#### Generalization Issue

머신 러닝으로 데이터 샘플들의 확률 분포를 모델링했다(말 그대로 모델링한 것이지, 진짜 모 확률 분포가 아니다. 추정일 뿐이다.). 그런데, 우리는 수집된 데이터 셋만을 이용해서 모 확률분포를 추정했는데, 우리가 모은 샘플들이 모 분포에서 나올 수 있는 모든 샘플들을 포함할까? 절대 아니다. 그럼, 샘플링되지 않은 놈들이 있을 수 있는데, 이들에 대해서도 잘 작동하는지는 어떻게 보장하나?

지금 우리는 frequentist statistics에서의 machine learning을 말하고 있다. 답은 **frequentist statistics**에 있다. Frequentist statistics에서는 가능한 많은 데이터 샘플을 뽑고 분포를 추정하게 된다. 여기서, **"가능한 많은 데이터 샘플"**이 핵심이다. 가능한 많은 데이터 샘플들을 뽑게 되면, 어쨌든 샘플링될 확률이 높은 데이터샘플은 많이 뽑힐 것이고, 적은 확률로 샘플링된 녀석들도 적은 개수나마 샘플링될 것이다. 즉, 데이터셋을 바탕으로 확률 분포를 모델링할때, 실제로 높은 확률을 가지는 샘플은 수가 많아서 모델링된 분포에도 높은 확률을 가질 것이고, 실제로 낮은 확률을 가지는 샘플은 수가 적어서 모델링된 분포에도 낮은 확률을 가질 것이다. 즉, 데이터 샘플수가 충분히 많다면, 모 확률분포와 매우매우 유사해진다. 그래서 머신러닝에서 **"데이터셋을 많이 모아라~"** 하는 것이다.

또한, 데이터 샘플들의 수가 많아질수록 매우 다양한 샘플들이 샘플링되어 있을 것이며, 이들 만으로도 충분히 모 확률분포에서 나올 수 있는 샘플들을 커버할 수 있다는 것이다. 따라서, 데이터 샘플 수가 적당히 많다면, 이들을 이용하면 모 확률분포와 매우 유사하게 모델링이 가능하고, 그럼, 미처 샘플링되지 못한 샘플들에 대해서도 잘 작동할 것이라는 이론이 있다.



#### Summary

정리해보면, frequentist statistics에서의 machine learning이란, 일단은 **함수**이다. 그런데, 확률값을 반환하는 **확률 함수**이다. 좀 더 정확히 말해보면 데이터 샘플들의 진짜 모 확률분포를 모델링한, **확률분포**이다.

Machine learning은 parameter(weight)를 이용해서 이 확률 분포를 모델링하고, likelihood를 모델링한다. Likelihood를 최대화하는 parameter를 계산(정확히는 "추정")한다. 이때, gradient descent를 이용한다. parameter를 계산하는 과정을 학습이라고 한다. 학습이 끝나면, machine learning은 모분포를 잘 추정한다고 가정하며, 일반화도 잘 이루어 졌을 것이라고 가정한다.
