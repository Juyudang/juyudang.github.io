---
title: 14. Predictive Simulations
toc: true
date: 2020-03-01 22:08:14
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Predictive Simulation



어떤 예측해야 할 변수 $\lambda$의 분포에서 샘플링하는 시뮬레이션을 말한다. $\lambda$를 데이터를 관찰하기 전의 분포 $p(\lambda)$에서 시뮤레이션하는가, 관찰한 후의 분포 $p(\lambda|D)$에서 시뮬레이션 하는가에 따라 다음과 같이 나뉜다.

- Prior Predictive Simulation
- Posterior Predictive Simulation



## Prior Predictive Simulation

Prior를 설정한 후, predictive simulation을 수행하는 것을 말한다. 다음처럼 모델링한 모델이 있다고 하자.
$$
 y_i|\lambda_{j} \sim \text{Pois}(\lambda_{j}) \\
 \lambda_j|\alpha,\beta \sim \text{Gamma}(\alpha, \beta) \\
 \alpha \sim p(\alpha), \beta \sim p(\beta)
$$
이때, $\alpha,\beta$에 대한 prior를 각각 설정했다면, 그 prior를 바탕으로 $\alpha^*, \beta^*$를 샘플링할 수 있다. 그런 다음, $\lambda^*$를 샘플링한다. 이때, $\lambda^*$를 샘플링하는 확률분포는 다음처럼 표시할 수 있다.
$$
p(\lambda^*) = \int p(\lambda^*|\alpha,\beta) p(\alpha) p(\beta) ~d\alpha ~d\beta
$$
위 확률 분포에 따라 $\lambda^*$를 샘플링하는 것을 prior predictive simulation이라고 부르고, 위 확률 분포를 prior predictive distribution이라고 부른다. 이 분포는 likelihood와 prior의 곱의 합으로 이루어진다.

$\lambda^*$를 샘플링했다면, $\lambda$와 마찬가지로 $y^*$를 샘플링할 수 있다. 일단 $\lambda^*$를 얻었다면, 다음 식에 의해 $y^*$를 샘플링할 수 있다.
$$
p(y^*) = \int p(y^*|\lambda)p(\lambda) ~d\lambda
$$
이렇게 계층을 올라가면서 각 파라미터와 예측값에 대해 prior predictive simulation을 할 수 있다.



## Posterior Predictive Simulation

데이터를 관측해서 prior를 credential distribution(posterior)로 수정한 이후, predictive simulation하는 것을 말한다. 이때, $\lambda$에 대한 시뮬레이션은 다음과 같다.
$$
p(\lambda|D) = \int p(\lambda|D,\alpha,\beta)p(\alpha|D)p(\beta|D) ~d\alpha ~d\beta
$$
위 식을 posterior predictive distribution이라고 부르는데, prior predictive distribution과 다른 점은 각 파라미터 분포에 prior 대신 posterior가 쓰였다는 점이다.

$y$의 경우도 마찬가지.
$$
p(y|D) = \int p(y|D,\lambda)p(\lambda|D) ~p\lambda
$$
시뮬레이션할 때는, posterior로부터 $\alpha^*,\beta^*$를 샘플링하고, 그 $\alpha^*, \beta^*$를 이용해서 $\lambda^*$를 샘플링한다. 그리고 그 $\lambda^*$를 이용해서 $y^*$를 샘플링하면 된다.