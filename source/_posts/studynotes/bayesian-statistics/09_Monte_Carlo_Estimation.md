---
title: 09. Monte Carlo Estimation
toc: true
date: 2020-03-01 22:08:09
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Monte Carlo Estimation



쉽게 말하면, 몬테 카를로 추정법이란, 어떤 특정한 파라미터를 얻기 위해서, 파라미터의 distribution으로부터 많은 샘플링 시뮬레이션을 한 후, 그 샘플들의 평균을 계산한 것을 파라미터의 기댓값으로 추정하는 것이다.

예를 들어, 어떤 파라미터 $\theta$는 어떤 분포 $p(\theta)$를 따른다. 우리는 파라미터 $\theta$의 기댓값 $E[\theta]$를 계산하고 싶다. 이를 계산하기 위해서는 원래 $E[\theta] = \int p(\theta) \cdot \theta ~~ d\theta$를 계산해야 한다. 하지만, 이 계산은 불가능하거나 매우 힘들 수 있다.

$E[\theta]$를 계산하는 대신 추정하는 방법으로 몬테 카를로 추정법을 이용한다. 우선, 컴퓨터로 $p(\theta)$로부터 $\theta$를 많이 샘플링한다. 그리고 그들의 평균 $\bar{\theta} = \frac{1}{m}\sum_i^m \theta_i$를 계산하고 $\bar{\theta}$를 $E[\theta]$로 추정하는 것이다.

분포 $p(\theta)$로부터 높은 확률의 $\theta$가 많이 샘플링되고 낮은 확률의 $\theta$는 적게 샘플링 되었을 것이다. 따라서 이 추정법은 유효할 수 있다. 다른 방식으로 해석하면, central limit theorem에 의해 샘플평균 $\bar{\theta}$는 실제 평균인 $E[\theta]$를 평균으로 하고 $\frac{1}{m}Var[\theta]$를 분산으로 하는 normal distribution을 따른다. 특히, 샘플수가 많아질수록, 계산한 샘플평균은 실제 평균값과 매우 유사할 확률이 높다.



$h(\theta)$의 기댓값 $E[h(\theta)]$를 추정하고 싶다. 그러면, $\theta$를 많이 샘플링해서 각 샘플로 $h(\theta)$를 계산하고 평균을 내면 $E[h(\theta)]$의 추정값이 된다.



### Monte Carlo Error

CLT(Central Limit Theorem)에 의해 파라미터 $\theta$에 대해 모은 샘플들은 $\mathbb{N}(E[\theta],\frac{Var[\theta]}{m})$를 따른다. $Var[\theta]$는 $\theta$의 분산으로, 다음으로 대체한다.

$$
Var[\theta] = \frac{1}{m}\sum_i (\bar{\theta} - \theta_i)^2
$$

그리고, $\frac{Var[\theta]}{m}$값을 **monte carlo error**라고 한다. Monte carlo estimation 값($E[\theta]$의 추정값인 $\bar{\theta}$)이 진짜 $E[\theta]$로부터 어느정도로 오차가 있을지에 대한 term이라고 볼 수 있다.



### Monte Carlo Marginalization

Paremter가 hierarchical하게 연결된 경우도 있다. 예를들어, 데이터 $Y$는 베르누이 분포 $\text{Bern}(\phi)$를 따르는데, 이 $\phi$가 또 베타분포 $\text{Beta}(2, 2)$를 따른다고 하자. 데이터 $Y$의 기댓값 $E[Y]$를 몬테 카를로 추정법으로 추정하기 위해서는 다음의 과정이 필요하다.

1. $\text{Beta}(2, 2)$로부터 $\phi$를 샘플링한다.
2. 샘플링한 $\phi$를 가지고 $Y|\phi$를 샘플링한다.
3. 이제, ($Y,\phi$)한 쌍이 생성되었다.
4. 반복한다.

이 과정의 특징이, 샘플 ($Y,\phi$)가 자연스럽게 $P(Y,\phi)$의 joint distribution을 반영한다는 것이다.

그런데, 위에서 샘플링한 $\phi$를 그냥 무시하고 $Y$만 취하면 그게 $\phi$에 대해 marginalization한 것과 같다. 즉, prior predictive distribution을 취한 것이다.

