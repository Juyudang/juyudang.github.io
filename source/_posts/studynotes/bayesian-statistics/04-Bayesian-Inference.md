---
title: 04. Bayesian Inference
toc: true
date: 2020-03-01 21:08:04
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



Frequentist inference의 단점은, 어떤 개인의 사전 지식과 상관없이 오직 데이터만으로 파라미터를 추정한다는 것에 있다. 하지만, 파라미터를 추정하는 데 있어서, 이미 알고 있는 사전 지식이 있다면 그것을 파라미터의 추정에 같이 이용하면 좋을 것이다.

본 섹션에서는 이를 가능하게 하는 Bayesian inference 프레임워크를 알아볼 것이다.

# Bayesian Inference

Bayesian inference에서는 Frequentist inference와 매우 비슷한 과정으로 이루어진다.

1. 데이터셋 $\mathcal{X}$이 있다고 가정한다.
2. 데이터가 생성되었을 법한 분포를 모델링하고 그 분포의 파라미터를 $\theta$라고 하자.

    $$X_i \sim p(x|\theta)$$

3. 파라미터 $\theta$에 대한 prior $p(\theta)$를 모델링하는데, 이때, prior 분포 모양은 개인이 파라미터에 대해 알고있는 사전 지식에 따라 정한다.

    $$\theta \sim p(\theta)$$

4. 위 두 식을 이용하여 파라미터에 대한 새로운 분포 $p(\theta \vert x)$를 계산한다. 이 분포는 개인의 사전 지식과 데이터로 얻은 지식을 모두 반영한 새로운 분포이다.

    $$p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)}$$

Frequentist inference와 다른 점은 data generating function $p(x \vert \theta)$의 파라미터 $\theta$에 대해 사전 지식을 삽입할 수 있다는 것이다.

Bayesian inference의 목적은 결국, 모델링한 분포의 파라미터 $\theta$의 density를 추정하는 것인데, 처음에 사전 지식으로 parameter density $p(\theta)$를 만들어놓고, 데이터로 이를 $p(\theta \vert x)$로 업데이트하는 과정을 거치게 된다. 즉, Bayesian inference의 최종 계산 결과는 posterior인 $p(\theta \vert x)$가 되는 것이다.

한 가지 주의할 점은 prior $p(\theta)$를 설정할때, 0이나 1의 값은 함부로 부여하면 안 된다는 것이다. 즉, $P(\theta = a) = 0$으로 하고싶은 어떤 값 $a$가 있다고 해서 함부로 그 확률에 0을 부여하면 안되고, 진짜 이론적으로 $\theta = a$이 될 확률이 불가능한지 확인하고 0을 부여해야 한다.

Prior에서 0으로 할당된 값은 데이터로 posterior를 계산한 이후에도 posterior에서 해당 값이 될 확률은 0가 되기 때문에 0을 함부로 사용하면 잘못된 결과를 가져올 수 있다.

예시를 한 번 들어보자.

동전던지기를 베르누이 분포로 모델링하고 싶다고 가정하자. 이때, 모델링을 하기 위해서는 파라미터 $\theta$가 필요하다.

$$X \sim \text{Bern}(x|\theta)$$

그런데, 우리는 이 동전이 fair하지 않다고 생각한다. 즉, 앞면이 나올 확률이 0.7 정도로 높다고 생각한다. 이러한 사전 지식을 $\theta$의 prior $p(\theta)$에 반영한다.

$$p(\theta) = \text{Beta}(7, 3)$$

그리고, 동전을 100번 던져서 앞면이 62회, 뒷면이 38회 나왔다고 가정하자. 데이터가 모였으니, likelihood 계산이 가능하다.

$$p(x|\theta) = \prod_{i=1}^{100} \text{Bern}(x_i|\theta) = \theta^{62}(1 - \theta)^{38}$$

이제, posterior를 계산해서 파라미터 $\theta$에 대한 신뢰도를 업데이트 해야 한다.

$$p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)} \\
= \frac{ \theta^{62}(1 - \theta)^{38} \theta^{6}(1 - \theta)^2}{Z} \\ 
= \frac{1}{Z} \theta^{68} (1 - \theta)^{40} \\ 
= \text{Beta}(68, 40)$$

(이러한 계산은 beta distribution이 bernoulli distribution의 conjugate prior이기 때문에 가능하다)

따라서, 우리는 사전 확률분포 $\text{Beta}(7, 3)$에서, 데이터를 사용하여 사후확률 분포 $\text{Beta}(68, 40)$으로 $\theta$에 대한 신뢰도를 업데이트 할 수 있다.