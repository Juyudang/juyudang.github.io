---
title: Appendix 2. Empirical Bayes
toc: true
date: 2020-03-01 22:08:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Empirical Bayes



**하이퍼파라미터 또한 데이터로 구하자.**

베이지안 모델링을 하게 되면, likelihood를 모델링하고, 그 파라미터를 prior로 모델링을 하게 된다.
$$
D \sim \text{Lieklihood}(\theta) ~~~ [P(D|\theta)]\\
\theta \sim \text{prior}(\lambda)
$$
그런데, 이때, prior의 파라미터(위에서는 $\lambda$)는 하이퍼파라미터로, 사용자가 직접 constant로 세팅해 주게 된다. Empirical bayes는 이 하이퍼파라미터를 사용자 대신, 데이터를 이용해서 MAP으로 추론하는 것을 말한다(point estimation).

즉,
$$
\hat{\lambda} = \underset{\lambda}{\text{argmax}} ~ P(\lambda|D)
$$
하이퍼파라미터의 posterior를 구하는 방법은 다음과 같다.
$$
P(\lambda|D) \approx P(\lambda) \int P(D|\theta)P(\theta|\lambda) d\theta
$$
그런데, 이때, $\lambda$의 prior가 필요해지는데, 그냥 uniform prior로 둔다. 그럼 다음과 같다.
$$
P(\lambda|D) \approx \int P(D|\theta)P(\theta|\lambda) d\theta \approx P(D|\lambda)
$$
 그리고, 이놈을 최대화하는 $\lambda$를 구하는 것이다.
$$
\hat{\lambda} = \underset{\lambda}{\text{argmax}} ~ \int P(D|\theta)P(\theta|\lambda) d\theta
$$


## Examples: beta-binomial Model

어떤 데이터에 대해 다음처럼 모델링했다 치자.
$$
x_i \sim \text{binom}(x_i|N_i, \theta_i) \\
\theta_i \sim \text{beta}(\theta_i|a, b)
$$
이때, 각 데이터 샘플이 서로 다른 binomial distribution에서 왔다는 것에 주목한다. 즉, $N_i, \theta_i$가 데이터마다 모두 다르다.

따라서, likelihood는 다음과 같다.
$$
\text{Likelihood}(X|\Theta) = \prod_i \text{binom}(x_i|N_i, \theta_i)
$$
그럼 다음처럼 EB(Empirical Bayes)를 이용해서 $a, b$에 대한 posterior에 비례(approximate)하는 함수를 구할 수 있다.
$$
P(a, b|D) \approx \prod_i \int \text{binom}(x_i|N_i, \theta_i) \cdot \text{beta}(\theta_i|a, b) ~ d\theta_i
$$
이 식을 최대화하는 $a, b$를 구하면 된다.
$$
= \prod_i \frac{\text{beta}(a + x_i, b + N_i - x_i)}{\text{beta}(a, b)}
$$

(왜 저렇게 나오지??)