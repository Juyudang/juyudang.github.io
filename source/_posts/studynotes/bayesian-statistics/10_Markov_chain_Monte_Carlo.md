---
title: 10. Markov Chain Monte Carlo
toc: true
date: 2020-03-01 22:08:10
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Markov Chain Monte Carlo



MCMC라고도 불린다. 파라미터 $\theta$의 분포 $p(\theta)$를 추정하고자 한다. 그러기 위해, bayesian inference를 하려고 하는데, 그러려면, 데이터를 수집한 후 posterior $p(\theta|Y)$를 계산해야 한다. 그러나, 이 posterior를 계산하기에 상당히 어려울 수 있기 때문에(특히, normalization constant) 대신, posterior를 추정하기로 한다. 이 posterior를 추정할때 쓰일 수 있는 알고리즘 중 하나가 MCMC이다.



### Background

우리가 추정하고자 하는 파라미터 $\theta$의 분포인 $\mathbb{p}(\theta)$를 추정하고자 한다. 그래서 $\theta$를 파라미터로 하는 어떤 데이터 분포 $y|\theta$로부터 $y_1,...y_k$를 샘플링했다. 이를 바탕으로 $\theta$의 prior $p(\theta)$를 설정하고, 데이터로 posterior $p(\theta|y_1,...y_k)$를 계산해서 $\mathbb{p}(\theta)$에 대한 베이지안 추론을 하고자 한다. 그런데, posterior인 $p(\theta|y_1,...,y_k)$를 계산하는게 매우 어렵거나 불가능한 경우가 있다. 이때, posterior 분포 $p(\theta|y_1,...,y_k)$를 추정하기 위한 방법으로 MCMC가 사용될 수 있다.



### MCMC

MCMC는 posterior의 분포 $p(\theta|y_1,...y_k)$를 추정하기 위해 마치 이 posterior로부터 샘플링됬을 법한 샘플 $\theta^*_1,...,\theta^*_m$을 생성해 준다. 이들은 posterior로부터 샘플링 되었을 거라고 가정하고 posterior를 monte carlo estimation으로 추정한다.

MCMC의 알고리즘으로 여러 개가 있다고 하는데, 대표적으로 Metropolis-Hastings 알고리즘이 있다.



## Matropolis-Hastings Algorithms

알고리즘은 다음과 같다.



1. 시작하기에 앞서 posterior $p(\theta|y_1,...,y_k)$를 정확히 계산할 수는 없더라도, 이 posterior에 비례하는 어떤 함수는 알고 있어야 한다. 즉, 다음을 만족하는 $g(\theta)$는 알고 있어야 한다.
   $$
   p(\theta|y_1,...,y_k) \propto g(\theta)
   $$

2. $\theta$와 도메인이 같거나 최대한 비슷한 분포 아무거나 고른다. 이 분포는 마르코프 체인을 만족하면 좋다. 즉, $q(\theta^*|\theta_{i-1})$.

3. 적당히 큰 수 $m$번을 반복하는데, $m$개의 $\theta^*$를 1개씩 샘플링할 것이다.

   1. $\theta^*$를 $q(\theta^*|\theta_{i-1})$로부터 1개를 샘플링한다.

   2. 다음을 계산한다.
      $$
      \alpha = \frac{g(\theta^*)q(\theta^*|\theta_{i-1})}{g(\theta_{i-1})q(\theta_{i-1}|\theta^*)}
      $$

   3. $\alpha \geq 1$이면, $\theta_i \leftarrow \theta*$로 accept한다. $0 \leq \alpha < 1$이면, $\alpha$의 확률로 $\theta_i \leftarrow \theta^*$로 accept하고, reject되면 $\theta_i \leftarrow \theta_{i-1}$한다.



분자에 $g(\theta^*)$가 있고, 분모에 $g(\theta_{i-1})$가 있어서, 이전에 뽑은 $\theta$보다 현재 뽑은 $\theta$가 더 $p(\theta|y_1,...,y_k)$에서 확률이 높다면, $\alpha \geq 1$이 되어서 accept된다. $g$가 $p$에 비례하기 때문에 그렇다.



이렇게 뽑은 $\theta^*$는 초반 샘플링된 놈들을 제외하면, posterior $p(\theta|y_1,...,y_k)$에서 샘플링된 것처럼 역할을 할 수 있다. 분포로부터 샘플링된 놈이 있으므로 posterior에 대해 monte carlo estimation이 가능해진다.



## Random Walk Algorithm

Matropolis-hastings 알고리즘에서, proposal distribution $q(\theta^*|\theta_{i-1})$를 $\theta_{i-1}$을 평균으로 하는 normal distribution으로 놓은 것을 말한다. Normal distribution은 대칭 분포이기 때문에, $\alpha=\frac{g(\theta^*)}{g(\theta_{i-1})}$이 된다.



## Gibbs Sampling

파라미터가 여러개라면, gibbs sampling이 Metropolis-hastings 알고리즘 보다 편할 수 있다. Metropolis-hastings 알고리즘에서는 파라미터 $\theta_1, ..., \theta_k$에 대해 proposal distribution을 각 파라미터마다 정의하고, accept, reject과정을 거칠 테지만, gibbs sampling과정에서는 이 과정을 없앴다. 대신 다음의 과정이 있다.

이때, parameter $\theta_1,...,\theta_k$를 모두 업데이트 1번씩 하는 과정을 1번의 iteration이라고 하자.

1. 일단, $p(\theta_1,...,\theta_k|y) \propto g(\theta_1,...,\theta_k)$를 만족하는 $g(\theta_1, ..., \theta_k)$를 알고 있어야 한다. $p(\theta_1,...,\theta_k|y) \propto p(y|\theta_1,...,\theta_k)p(\theta_1,...,\theta_k)$를 활용.

2. 하나의 parameter에 대한 full conditional distribution의 proportion을 계산해야 하는데, 다음과 같이 posterior 분포에 비례하므로(Bayes' rule에 의해), $g$에 비례한다.
   $$
   p(\theta_i|\theta_1,...,\theta_{i-1},\theta_{i+1},...,\theta_k,y) \propto p(\theta_1,...,\theta_k|y) \propto g(\theta_1,...,\theta_k)
   $$
   그리고, 나머지 파라미터는 모두 주어진 것으로 가정한다. 나머지 파라미터는 초기값이거나 가장 최근에 업데이트한 값으로 들어간다.

3. 그렇게 되면, $g$에서 $\theta_i$에 의해 parameterize되지 않는 항은 모두 constant로 취급할 수 있으며, proportion에서 제외할 수 있다. 그럼 $g$가 간소화된다.

4. 이렇게 되면, $\theta_i$에 대한 full conditional distribution이 우리가 아는 분포, 즉 샘플링이 가능한 분포가 되는 경우가 있다. 이럴 경우, 그냥 그 분포에서 샘플링하면 되기 때문에 accept, reject과정이 필요가 없다. 하나를 샘플링하고 $\theta_i$를 업데이트한다.

5. 파라미터 $\theta_{i+1}$에 대해 같은 과정을 반복하는데, $\theta_{1,...i}$은 이전 iteration의 값이 아니라, 현재 iteration값을 이용한다.

6. 만약, 4번 과정에서 샘플링이 가능한 표준적인 분포가 아니라면, 그 안에서, $\theta_i$ 하나에 대해서 matropolis-hastings 알고리즘의 방식을 사용해서, 하나의 샘플을 accept 혹은 reject로 업데이트한다.

7. 업데이트 이전 값은 어디다가 저장해두자. 그 값들이 샘플들이다.



## Assessing Convergence of MCMC

MCMC알고리즘에서 샘플링한 샘플들 $\theta^*_1, ..., \theta^*_k$의 평균값 $\bar{\theta^*}$이 $\theta$의 posterior 분포 $p(\theta|Y)$를 잘 추정하려면, 마르코프 체인이 충분히 수렴해야 하고, 수렴한 체인으로부터 $\theta^*$가 충분히 샘플링되어야 한다. 하지만, 마르코프 체인이 언제 수렴할지를 모르기 때문에 몇 개의 샘플까지가 수렴이 안된 상태의 샘플인지, 몇 개가 유용한 샘플인지 알 수가 없다.



### Stationary Distribution

마르코프 체인이 추정하고자 하는 target distribution(parameter의 posterior가 된다)을 최대한 추정한 distribution을 의미하며, 마르코프 체인이 충분히 수렴한 상태에서의 distribution을 의미한다. 당연히 알 수 없으며, 여기서 마르코프체인으로 샘플링만 가능하다.



### Monte Carlo Effective Sample Size

진짜 Stationary distribution으로부터 독립적으로 샘플링한 샘플을 $\theta_{eff}$이라고 하자. 즉, 이들은 실제로 posterior로부터 샘플링한 샘플과 매우 유사할 것이다.

우리가 마르코프 체인으로부터 샘플링한 샘플의 개수를 $n$이라고 하자. 하지만, 수렴이 제대로 되지 않은 상태에서 뽑은 것은 독립적인 샘플일 수가 없고, 마르코프 체인이기에, 완전히 독립적이기는 어렵다. 따라서, 유용한 샘플들은 일부일 것이다.

이 $n$개의 샘플이 가지고 있는 정보가 과연 몇 개의 $\theta_{eff}$들이 가지는 정보와 같은지를 나타내는게 monte carlo effective sample size이다. 즉, $n=1000000$개의 샘플을 뽑았는데, effective sample size $n_{eff}=500$이라고 하면, 이 100만개의 샘플들은 실제로 posterior에서 500개를 샘플링한 것과 같은 정보를 가진다.

이는, 마르코프 체인이 posterior를 완전히 추정하지 못하기 때문이다. 또한, $n_{eff}$이 너무 작다면, 수렴 속도가 느린 것일수도 있다.



### Auto-correlation

마르코프 체인에서 샘플링한 한 샘플이 과거 샘플들과 얼마나 많은 dependency가 있는가를 나타낸다. [-1,1] 범위의 값을 가지며, 0에 가까울수록 그 샘플은 과거 샘플들과 관계없는, 독립적인 샘플들이다. monte carlo sample size를 증가시키려면, 이 독립적인 샘플들이 필요하다.

마르코프 체인으로부터 샘플링을 하면, 초기 몇 개의 샘플까지는 수렴이 되지 않아서 correlation이 높다. 초기 correlation이 0에 가까운 값이 되는 지점까지의 샘플은 버리는 것도 방법(**burn-in 이라고 부름**).

![1567820594976](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/1567820594976.png)

Auto-correlation이 0과 가까운 값이 적으면 effective sample size가 감소한다.



### Gelman-Rubin Diagnostic

마르코프 체인으로부터 샘플링한 샘플들을 주면, 실수값을 반환하는데, 1에 가가우면 수렴이 된 것이고, 1보다 많이 크면, 수렴이 아직 안된 것이다.

