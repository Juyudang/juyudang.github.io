---
title: 06. Prior Predictive Distribution
toc: true
date: 2020-03-01 21:08:06
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---

# Prior Predictive Distribution



Prior정보만으로, 즉, 데이터 없이 데이터가 어떻게 분포할지,즉 $p(D)$를 추정해본 분포이다.

데이터 분포 $p(D)$는 다음과 같이 쓸 수 있다.
$$
p(d) = \int_0^1 p(d|\theta)p(\theta) ~d\theta
$$
이때, 데이터 없이 사전 정보만으로 $p(D)$를 추정하는데, 사전정보로 추정한 데이터 분포 $p(D)$를 추정한 것을 prior predictive distribution이라고 부른다.

Prior predictive distribution은 데이터 수집 전에, prior정보만을 이용해서 데이터 sample space distribution을 추정한 것이라고 할 수 있다.



# Posterior Predictive Distribution



데이터를 수집한 후, 데이터의 분포를 추정한 분포를 말한다.

데이터 $d_1$를 수집했다고 치자. 그럼 다음에 샘플링될 $d_2$의 확률분포는 다음과 같다.
$$
p(d_2|d_1) = \int_0^1 p(d_2|d_1,\theta)p(\theta|d_1)d\theta
$$
이때, $d_1 \perp d_2$이므로, 다음과 같다.
$$
p(d_2|d_1) = \int_0^1 p(d_2|\theta)p(\theta|d_1)d\theta
$$
Prior predictive distribution과 다른 점은 prior 자리에 posterior가 들어갔다는 점이다.

Posterior predictive distribution은 데이터를 관찰한 후, 그 정보를 이용해서 데이터 sample space 분포를 추정한 것이라고 할 수 있다.