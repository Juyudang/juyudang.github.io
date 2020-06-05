---
title: 03. Frequentist Inference
toc: true
date: 2020-03-01 21:08:02
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---

# Frequentist Inference



Frequentist statistics에서는 sample space의 분포 파라미터 $\theta$를 추정할 때, 다음과 같은 방법으로 추론할 수 있다.

1. 일단 데이터를 많이 모은다. $X_1 = d_1, X_2 = d_2, ..., X_n=d_n$
2. Central limit theorem을 이용해서 데이터의 평균치 또는 합을 계산한다. $\bar{X} = \frac{1}{n}\sum\limits_i X_i$
3. 이 평균치는 $\theta$에 대한 함수일 것이고(애초에 $X_i$가 $\theta$에 대한 함수임) 이 평균치는 $\mathbb{N}(\bar{X}, \frac{\sigma}{\sqrt{n}})$의 분포를 이룬다. (합의 경우는 $\mathbb{N}(n\bar{X},\sigma)$)
4. 이 분포에 대해 confidence interval을 계산하고, $\bar{X}$주위 그 interval 안에 해당 confidence ($p-value$) 의 자신감으로, 진짜 $\mu$가 있다고 가정한다.

주의할 점은, $p-value$는 $\mu$가 그 confidence의 확률로 interval안에 있다는 것이 아니다. $\mu$는 고정되어 있는 값이라서 그 interval 안에 있을 확률은 0 아니면 1이다. 다만, $\mu$가 거기에 있을 것이라는 95%($p-value=95$)의 자신감이 있을 뿐이다.



## Confidence Interval



동전던지기 시행에서 앞면이 나올 확률 $p$를 알고 싶다.

100번 던져본다. 각 시행은 $X_i$이다. 이때, 100번의 시행을 모두 더한 random variable $Y=\sum\limits_i X_i$를 정의한다. 그럼 $Y$는 다음의 분포를 따른다.
$$
Y \approx \mathbb{N}(100p, 100p(1-p))
$$
$Y = \frac{1}{n} \sum\limits_i X_i$라고 정의했다면, $Y \approx \mathbb{N}(p, \frac{p(1-p)}{\sqrt{n}})$가 되겠다.

어쨌든, 55번의 H, 45번의 T이 나왔다면, frequentist statistics의 확률 정의에 의해 $\hat{p}=0.55$이고 이 추정치는 95%, 97%, 99% confidence interval로 어느정도 true $p$에 가깝다고 확신을 내릴 수 있다. 95%를 예로 들면,
$$
55 - 1.96 * 100 * 0.55 * 0.45 \leq 100p \leq 55 + 1.96 * 100 * 0.55 * 0.45
$$
로 $100p$에 대한 confidence interval을 계산할 수 있다.



## Maximum Likelihood Estimation



데이터를 확률분포 $p(\mathbb{D}|\theta)$로부터 샘플링했을 때, 가지고 있는 데이터가 샘플링 됬을 확률을 $p(D|\theta)$라고 표현한다면, 이를 liklihood라고 한다. 이 likelihood를 최대화하는 파라미터 $\theta$를 찾으면, 즉, likelihood를 최대화하는 분포를 구하면, 그것이 sample space분포인 $p(\mathbb{D}|\theta)$와 매우 유사할 것이라는 것이라고 가정한다. 따라서 likelihood를 최대화하는 파라미터 $\theta$를 찾고, 나아가 sample space distribution을 추정하는 방법을 MLE(Maximum likelihood estimation)라고 부른다.

Likelihood를 최대화하는 $\hat{\theta}$를 구하는데 이용하는 방법은 미분하고 derivatives를 0으로 하는 $\theta$를 구하는 것이다. 즉, 극점을 구하는 것이다.



예를 들어, 동전이 fair한지, loaded인지 구하려고 한다. 만약, fair한 동전이라면 앞 뒷면이 나올 확률은 0.5로 같다. loaded 동전이라면 앞면이 나올 확률은 0.7, 뒷면이 나올 확률은 0.3이라고 하자.

동전을 다섯 번 던져서 5개의 데이터를 얻었다. 이때, 2번은 앞면, 3번은 뒷면이 나왔다.

이때, liklihood는 동전이 fair일때와, loaded일때에 대해서 각각 구할 수 있다.
$$
p(D|\theta) = \begin{cases} \begin{pmatrix} 5 \\ 2 \end{pmatrix} * 0.5^5 & \text{if } \theta \text{ is fair} \\ \begin{pmatrix} 5 \\ 2 \end{pmatrix} * 0.7^2 * 0.3^2 & \text{if } \theta \text{ is loaded} \end{cases}
$$
결과를 구해보면, $\theta$가 fair일때의 $p(D|\theta)$가 더 높다는 것을 알 수 있다. 즉, $\theta$가 fair일때, likelihood가 더 높다. 따라서 MLE에 의해 likelihood가 최대화되는 $\theta=\text{fair}$ 이라고 추정할 수 있다.



그런데, 동전은 물리적인 물체이므로 데이터가 주어졌을 때의 동전이 fair할 확률 $p(\theta=\text{fair}|D)$은 $p(\theta=\text{fair})$와 같다. 동전이 fair한지 안하는지는 변하지 않는 것이고 데이터셋과 상관없이 결정된 것이기 때문이다. 따라서 다음과 같다. 
$$
p(\theta=\text{fair}|D) = p(\theta=\text{fair}) \in \{0, 1\}
$$


즉, frequentist inference는 다음과 같이 정리할 수 있다.
$$
\hat{\theta} = argmax_{\theta} ~p(D|\theta)
$$


다른 예시로, 개와 고양이를 구분하는 classifier를 구현하고 싶다고 하자. MLE 방법에서는 $\theta \in \{개, 고양이\}$이고, 사진을 보여주고 frequentist inference를 한다고 하자. 만약, 개라면 사진처럼 생겼을 확률과 고양이라면 사진처럼 생겼을 확률을 비교하고, 개라면 사진처럼 생겼을 확률이 높으면 개라고 판단하고, 고양이라면 사진처럼 생겼을 확률이 높다면 고양이로 판단한다.