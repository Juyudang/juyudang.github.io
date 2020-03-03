---
title: Appendix 1. Maximize a Posterior
toc: true
date: 2020-03-01 22:08:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Maximize a Posterior



**사후확률 최대화 추정법(MAP Estimation).**

Likelihood를 최대화하는 추정법(MLE - Maximum Likelihood Estimation)은 오직 데이터만을 이용해서 파라미터를 point estimation하는 방법이다. 하지만, MAP는 선행 지식 또는 개인의 믿음을 파라미터 추정에 집어 넣고, 데이터의 정보와 합해서 파라미터를 point estimation하는 방법이다.



방법은 다음과 같다.

1. 데이터에 대한 likelihood $P(D|\theta)$를 모델링한다.

2. Likelihood $P(D|\theta)$의 파라미터 $\theta$의 prior $P(\theta)$를 모델링한다.

3. Prior $P(\theta)$와 likelihood $P(D|\theta)$를 이용해서 posterior $P(\theta|D)$를 직접 계산할 수 있으면 계산하되, 불가능하다면, posterior와 비례하는 함수 $g(\theta|D)$를 계산한
   $$
   P(\theta|D) = \frac{P(D|\theta)P(\theta)}{\sum_{\theta'}P(D|\theta')P(\theta')}
   $$
   분모인 normalization constant가 계산 불가능할 수도 있다. 그럼, $g(\theta|D)$를 대신 구한다.
   $$
   P(\theta|D) \approx g(\theta|D) = P(D|\theta)P(\theta)
   $$
   보통, 파라미터의 prior $P(\theta)$가 likelihood $P(D|\theta)$와 conjugate를 이루면, posterior를 직접 계산할 수 있을 것이다. 그렇지 못하면서 가능한 $\theta$의 개수가 많다면, $g(\theta|D)$를 계산해야 하는 경우가 많다.

4. Posterior $P(\theta|D)$가 가장 커지는 $\theta$를 구한다.
   $$
   \hat{\theta} = \underset{\theta}{\text{argmax}} ~ P(\theta|D)
   $$
   또는 $g(\theta|D)$가 가장 커지는 $\theta$를 구한다.
   $$
   \hat{\theta} = \underset{\theta}{\text{argmax}} ~ g(\theta|D)
   $$
   미분을 통한 극점을 이용하자. (Gradient descent 같은)



### Is MAP a Bayesian Method?

Bayes 통계의 특징은, 파라미터를 하나의 값이 아닌 분포로 본다는 것이다. 하지만, MAP는 파라미터의 분포를 추정하는게 최종 목표가 아니라 파라미터를 **point estimation**하는 것이 최종 목표이다.

MAP의 계산 과정상 파라미터의 posterior를 계산하게 되지만, 결국은 point estimation으로, 파라미터를 하나의 값으로 본다는 점에서 완전한 베이지안 통계적 방법이라고 보지 않는 경우가 많다고 한다.