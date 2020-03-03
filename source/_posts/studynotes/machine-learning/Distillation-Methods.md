---
title: Distillation Methods
toc: true
date: 2020-03-03 22:07:01
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# Distillation Methods



다음을 참고했다.

[Distillation as a Defense to Adversarial
Perturbations against Deep Neural Networks](https://arxiv.org/pdf/1511.04508.pdf)

[Distilling knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)



## Distilling Knowledge in a Neural Network

Distillation이란, [Distilling knowledge in a Neural Network]라는 논문에서 등장한 것으로 보이며, 이 논문에서는 다음과 같은 과정을 통해 **네트워크의 지식**을 다른 네트워크에게 전달해 줄 수 있다고 한다.

1. 먼저, 데이터 $(X, Y)$를 충분히 잘 학습할 수 있도록 큰 네트워크 $F$를 충분히 학습한다. 학습은 $F$의 정확도를 최대한 올리도록 진행한다.

2. 작은 네트워크 $F_d$를 만들고, 같은 데이터셋 $(X, Y)$를 이용해서 그 네트워크를 학습하는데, 다음과 같은 과정을 거친다.

   1. 데이터 $X$를 큰 네트워크 $F$에 통과시켜서 softmax에 들어가기 바로 전 값, 즉, logit $F(X)$를 얻는다. 큰 네트워크의 파라미터는 모두 고정시킨다.

   2. 데이터 $X$를 작은 네트워크 $F_d$에 통과시켜서 logit 값 $F_d(X)$를 얻는다.

   3. $\sigma(F_d(X))$를 ground  truth인 $Y$와 가깝게 학습시키는 loss를 정의한다. $\sigma$는 softmax이다.
      $$
      L_{CE}(F_d(X), Y)
      $$

   4. 또, 작은 네트워크가 예측한 결과는 큰 네트워크가 예측한 결과를 최대한 따라가도록 학습하도록 한다. 그에 맞는 loss를 정의한다.
      $$
      L_{CE}(\sigma(\frac{F_d(X)}{T}), \sigma(\frac{F(X)}{T}))
      $$
      이때, logit을 하이퍼파라미터 $T$로 나눠줌으로써, 조금 약하게 한다.

      이것은, 작은 네트워크가 큰 네트워크의 데이터셋 $(X, Y)$를 학습한 결과를 최대한 따라가도록 만드는 효과가 있으며, 큰 네트워크의 지식을 작은 네트워크에게 전수한다고 볼 수 있다.

이러한 방법으로, 매우 유사한 성능을 내는 compact한 네트워크를 만들 수 있으며, 큰 네트워크 대신 작은 네트워크를 이용하면 computation complexity를 크게 줄일 수 있을 것이다.



## Generalization using Distillation

Distillation은 모델을 generalization하는 방법으로도 응용할 수 있다. 이 방법으로 상당한 adversarial attack 또한 방어가 가능하다(한때는 adversarial attack에 대한 state-of-the-art 기술이었다고 하는 듯 하다).

방법은 다음과 같다.

1. 똑같은 구조를 가지지만 weight를 공유하지 않는 두 네트워크 $F, F_d$를 생성한다.

2. 먼저, 데이터셋 $(X,Y)$를 이용해서 $F$를 충분히 학습한다. 이후, $F$의 파라미터는 고정시킨다.

3. 같은 데이터셋 $(X, Y)$를 이용해서 $F_d$를 다음과 같이 학습한다.

   1. 데이터 $X$를 $F$에 통과시킨, softmax 결과 $F(X)$를 구한다.

   2. 데이터 $X$를 $F_d$에 통과시킨, softmax 결과 $F_d(X)$를 구한다.

   3. $F(X)$과 $F_d(X)$를 가깝게 학습한다.
      $$
      \text{argmin} ~ KLD(F(X)||F_d(X))
      $$
      (KL-divergense말고 다른걸 써도 됨)

이 방식은, 첫 번째 네트워크 $F$를 학습할 때, one-hot label $Y$를 이용하지만, 두 번째 네트워크 $F_d$를 학습할 때는, one-hot label이 아니라 $F$의 softmax값을 사용하게 된다. One-hot label $Y$를 이용하게 되면, 해당 정답 라벨에 모델이 over-confident하게 된다. Softmax값을 이용하게 되면, 정답 라벨이 될 확률이 크게 학습되는것은 같다. 그러나, 덜 confident하게 되어 overfitting확률이 줄어든다. 이 방법으로 학습된 네트워크는 adversarial attack을 매우 효과적으로 막아냈으며,  generalization이 그 이유라고 분석되고 있는 듯 하다.