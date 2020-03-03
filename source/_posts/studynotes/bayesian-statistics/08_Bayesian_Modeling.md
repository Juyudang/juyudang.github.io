---
title: 08. Bayesian Modeling
toc: true
date: 2020-03-01 22:08:08
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Bayesian Modeling



## Statistical Modeling

Bayesian modeling은 statistical modeling의 일종이다. Statistical modeling이란, 데이터가 생성/샘플링되는 프로세스를 모델링하는 것을 의미한다.

Bayesian modeling은 이러한 모델링을 할때, 베이지안 방법론을 적용한 것을 말한다. 전체적인 모델링 프로세스는 bayesian modeling이나 frequentist modeling이나 같다.

1. 문제 이해
2. 데이터 수집
3. 데이터 관찰
4. 모델 구성
5. 모델의 구현 및 fit
6. 샘플공간 분포 파라미터 추정
7. 테스트 및 예측성능 검사
8. 5~7번 반복
9. 모델의 이용.

이때, frequentist modeling과 bayesian modeling에서의 차이는 모델의 구현과 fit, 파라미터 추정에 있다.



## Model Specification - 모델 구성

모델의 구성은 계층적으로 적어 내려가면서 파악하는게 어느정도 쉽다. 일단, likelihood를 적고 likelihood에 영향을 미치는 random variable 또는 parameter를 찾는다.

어느 학교의 학생들의 키(height)의 분포를 예로 들자. 키의 분포는 normal distribution을 따른다고 가정하고 likelihood를 만든다.
$$
f(y|\theta) = \mathbb{N}(\mu, \sigma^2)
$$
그리고 $\mu$, $\sigma^2$의 분포가 필요한데, 이들의 prior를 정한다.
$$
\mu \approx \mathbb{N}(\mu_0, \sigma_0^2)
$$
$$
\sigma^2 \approx \mathbb{IG}(\nu_0, \beta_0)
$$



여기서 $\mathbb{IG}$는 inverse-gamma distribution을 뜻한다. 그리고 각 prior는 독립이라고 가정하면, $p(\mu,\sigma^2) = p(\mu)p(\sigma^2)$일 것이다. 모델로 그려보면 다음과 같다.

![image-20200303093449313](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200303093449313.png)

이렇게, 우선 데이터가 어떻게 생성되었을지에 대해 그 생성 과정을 모델링하는데, likelihood부터 적고, 아래 파라미터까지 노드를 뻗어 나간다.