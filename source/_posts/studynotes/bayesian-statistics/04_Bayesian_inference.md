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



# Bayesian Inference



우리가 추정하고자 하는 parameter $\theta$에 대해 $\theta$가 어떻게 분포되어있는지 사전 지식 또는 정보가 있다면 그것을 이용하는게 좋을 것이다. 하지만, frequentist inference에서는 사전 정보를 이용하기 어렵다.

사전 정보를 이용해서 $p(\theta)$를 초기화한후(prior), 데이터를 수집하면서 얻은 정보(posterior)를 이용해서 $p(\theta)$분포를 $p(\theta|D)$로 업데이트한다. 이렇게 $p(\theta)$을 추정해 가는 방식을 **bayesian inference**라고 한다. 그리고, 얻은 데이터를 바탕으로 $p(\theta|D)$를 최대화하는 $\hat{\theta}$를 선택하는 것을 **Maximize A Posterior(MAP)** 추정이라고 한다.

즉, 다음과 같다.
$$
\hat{\theta} = argmax_{\theta} ~p(\theta|D)
$$


개와 고양이를 판별하는 classifier를 만든다고 치자. 역시 $\theta \in \{개, 고양이\}$이고, 사진을 주고 bayesian inference로 개인지, 고양이인지 판단을 한다고 하면,  사진처럼 생겼을 경우 개일 확률과 사진처럼 생겼을 경우 고양이일 확률을 비교한다. 사진처럼 생겼을때, 고양이일 확률이 개일 확률보다 높으면 고양이라고 추정하는 방식이 MAP이다.

그런데, $p(\theta|D)$는 다음과 같이 계산한다.
$$
p(\theta|D) = \frac{p(D|\theta)p(\theta)}{\sum_ip(D|\theta_i)p(\theta_i)}
$$
즉, 사후확률 $p(\theta|D)$는 관찰된 데이터의 likelihood $p(D|\theta)$와 사전확률 $p(\theta)$을 이용해서 계산된다. 그리고 계산된 사후확률 $p(\theta|D)$를 이용해서 $p(\theta)$를 업데이트한다(단순 대입, $p(\theta) \leftarrow p(\theta|D)$). 이렇게 데이터를 모은 정보를 바탕으로 prior를 posterior로 업데이트 해 가면서 $\theta$에 대한 분포 $p(\theta)$를 추정해 나가는 방식을 bayesian inference라고 한다.



주의할 점은 prior $p(\theta)$를 어느 특정 지점에서 0 또는 1로 설정하면, posterior에서도 그 지점은 0 또는 1이 된다. 따라서 왠만하면 0 또는 1을 어떤 지점에 할당하지 않도록 한다.
$$
p(\theta|D) \propto p(D|\theta)p(\theta) = p(\theta)
$$
