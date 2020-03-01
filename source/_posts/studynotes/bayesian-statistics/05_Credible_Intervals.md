---
title: 05. Credible Intervals
toc: true
date: 2020-03-01 21:08:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---

# Credible Intervals



Prior로 $\theta$의 distribution $p(\theta)$를 정한 후, 데이터를 이용해서 bayesian inference과정을 거쳤다고 하자. 따라서, $p(\theta|D)$를 구했다. 그리고, 이 posterior속에서 파라미터 $\theta$가 어디쯤에 위치할지, credible interval을 계산할 수 있다. 이는 frequentist statistics에서의 confidence interval과 매우 유사하지만, 다음의 차이점이 있다.

1. Confidence interval은 $\theta$는 고정되어 있고 bound 경계가 random variable이다. 반면, credible interval에서는 $\theta$가 random variable이고 bound가 고정된 값이다.
   - confidence interval은 이 구간 사이에 모 파라미터 $\theta$가 있을 것이라는 자신감이 있을 뿐, $\theta$가 그 구간에 위치할 확률이 p-value가 되는 것이 아니다. Frequentist statistics에서는 $\theta$는 고정되어 있고 변하지 않는다. 
   - 반면, credible interval은 그 구간 내에 $\theta$가 있을 확률을 의미한다.

Posterior를 충분한 데이터로 구했다면, 사전 지식과 합쳐서 credible interval을 계산하고 $\theta$가 어느 범위에 있을 확률을 구하는 것이다.



### Equal-Tailed Interval

95%의 credible interval을 구하고 싶다면, 한쪽 끝에서 2.5%의 bound를 계산하고 다른 한 쪽 끝에서 2.5%의 bound를 계산한다. 그리고 그 사이가 equal tailed interval이 된다.



### Highest Posterior Density

양쪽 끝을 같은 확률로 자르지말고, 확률이 높은 구간을 최대한 포함하자는 것이다. 만약, $p(\theta|D) = 2\theta$의 경우, 오른쪽 꼬리는 매우 확률이 높은 구간인데, 자르기 아깝다는 것이다. 따라서 확률이 낮은 왼쪽 꼬리만 잘라서 interval을 구한다.