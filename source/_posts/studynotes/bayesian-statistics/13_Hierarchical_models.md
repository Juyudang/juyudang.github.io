---
title: 13. Hierarchical Models
toc: true
date: 2020-03-01 22:08:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Hierarchical Models



데이터 생성 프로세스를 계층적으로 모델링한 것을 의미한다. 즉, likelihood의 파라미터는 또 다른 파라미터를 갖는 분포를 가지는 형태. 예를들어 likelihood는 poisson 분포를 따른다고 모델링하고, 그 파라미터 $$\lambda$$는 또 다른 파라미터 $$\alpha, \beta$$를 가지는 gamma 분포를 따른다고 모델링했다고 하자. 이 경우가 계층적 모델링에 속한다.
$$
y_i|\lambda_{j} \sim \text{Pois}(\lambda_{j}) \\
\lambda_j|\alpha,\beta \sim \text{Gamma}(\alpha, \beta) \\
\alpha \sim p(\alpha), \beta \sim p(\beta)
$$
이때, 각 $$\lambda$$는 여러개가 있고, 그중 하나에서 $$y$$가 생성되지만, $$\lambda$$는 모두 같은 분포에서 나온 녀석들이라고 모델링 한 것이다. $$p(\alpha),p(\beta)$$는 각각 $$\alpha,\beta$$의 prior이다.

이 경우의 장점은, 데이터가 모두 독립이지 않고 같은 성질을 갖는 놈들(같은 $$\lambda$$에서 나온 놈들에 해당)은 비슷하고 다른 성질은 갖는 놈들은 조금 다르다는, 약간의 correlation이 있는 데이터를 모델링할 수 있다.



$$\alpha,\beta$$는 고정적으로 줘도 될 것인데, 왜 궂이 prior를 할당해서 샘플링하는가? 이건 $$\alpha,\beta$$에 대한 uncertainty(불확실성) 때문이다.

$$\alpha,\beta$$는 독립적으로 샘플링된다. 그러나, 샘플링된 $$\lambda$$들 끼리는 독립이 아니다. 대신, $$\alpha,\beta$$가 주어진다면, 각 $$\lambda$$끼리는 해당 특정한 $$\alpha, \beta$$를 파라미터로 하는 분포에서 독립적으로 샘플링됬을 것이므로 조건부 독립이다. $$y$$끼리도 독립이 아니지만, $$\lambda$$가 주어진다면 $$y$$들 끼리 독립(조건부 독립)이다. $$\lambda$$가 주어졌다는 의미는 어느 한 그룹으로 좁혔고, 그 그룹 내에서 샘플들끼리는 독립이기 때문이다(그렇게 모델링 했으니까).

이렇게 함으로써, 다른 그룹은 다른 모델로 모델링하기 보다는 계층적인 하나의 모델로 모델링할 수 있다.