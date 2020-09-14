---
title: Appendix 3. Variational Inference
toc: true
date: 2020-09-14 21:08:16
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Summary

Latent variable 또는 hidden variable의 posterior approximation을 위한 optimization 기반의 variational inference 방법론 및 응용 분야 소개

# Motivations of VI

Hidden variable(latent variable, mixture variable) $z$의 posterior density $p(z|x)$를 계산하기 쉽지 않거나, 불가능에 가까울 경우에 사용하는 방법으로 MCMC(Markov chain Monte Carlo) 방법(approximation)이 존재했다. Stationary distribution이 posterior $p(z|x)$인 Markov chain을 모델링하고 그 체인으로부터 얻은 샘플들로 posterior $p(z|x)$에 대한 empirical(경험적) 추정을 하는 것이 기존의 방법이었다.

하지만, MCMC 방법은 매우 느려서 복잡한 모델이나 빨리 inference 를 수행해야 하는 작업에는 적합하지 못했다.

VI(Variational inference)는 MCMC만큼 정확하지는 않지만, 나름 큰 스케일의 문제에 적용이 가능한 대안으로 제시되었다.

VI는 MCMC와 같은 approximation 방법이지만, 샘플링 기반 방법이 아닌 optimization 기반의 방법이다.

# Variational Inference

Variational inference는 다음과 같은 문제를 푸는 것이다.

$$q^*(z) = \underset{q(z) \in Q}{\text{argmin }} \text{KL}(q(z)||p(z|x))$$

**Goal**: Posterior $p(z|x)$와 매우 근접한 approximation $q(z)$를 찾는 것

$p(z|x)$에 매우 근사하는 함수 $q^*(z)$를 찾기 위해 $q(z)$를 parameterize한 후, $q(z)$를 위 식에 맞추어 최적화하게 된다. 이에따라, $q(z)$는 적절히 flexible하게 parameterize되어야 하며, true posterior $p(z|x)$를 잘 추정하도록 충분히 간단한 함수여야 한다.

## Problem of Bayesian Inference

Bayesian inference에서는 어떤 파라미터 $\theta$의 분포를 추정하고자 할 때, likelihood $p(x|\theta)$와 parameter에 대한 prior $p(\theta)$를 이용하여 posterior $p(\theta|x)$를 계산하게 된다. 이 posterior가 파라미터의 density에 대한 추정이 된다.

그러나, $p(\theta|x)$를 계산하기 위해서는

$$p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)}$$

를 계산해야 하는데, $p(x)$는 보통의 경우 계산이 불가능하다.

따라서, 일반적인 Bayesian inference(위 식을 푸는 것)로는 posterior를 추정할 수가 없는 경우가 많고, 이를 해결하기 위해 MCMC나 VI가 존재한다. VI에서는 $p(\theta|x)$를 계산이 가능한 어떤 다른 함수 $q(\theta)$로 근사시키고, $q(\theta)$를 대신 사용하게 된다.

## VI vs MCMC

MCMC와 VI는 같은 문제를 푸는 inference 알고리즘이다. 둘 다 true posterior를 계산하는게 힘들때 사용되며, **approximation**을 통해 true posterior를 **추정**한다. 다만, MCMC는 샘플링을 통해 추정하고 VI는 optimization을 통해 추정한다.

- MCMC
  - VI에 비해 posterior 추정이 정확한 편
  - 복잡한 모델일수록, 대규모 데이터셋을 사용할수록 적용하기가 어려움(느림)
  - Target distribution(보통은 posterior를 가리킴)으로부터 상당히 정확한(진짜 그 distribution에서 샘플링한 듯한) 샘플을 얻을 수 있음
- VI
  - MCMC에 비해 posterior 추정이 부정확할 수 있음
  - Optimization 방법이라서 복잡하고 큰 문제(큰 데이터셋)에 잘 작동함(빠름)
  - Target distribution의 density만 추정할 뿐, 샘플을 얻을 수는 없음

이러한 특성 때문에 MCMC와 VI의 사용 용도는 미세하게 다르다. **MCMC는 density를 시뮬레이션하는 용도**이고, **VI는 density를 추정하는 용도**이다.

## Evidence Lower Bound

Variational inference에서는 다음과 같은 $q^*(z)$를 찾는 것이다.

$$q^*(z) = \underset{q(z) \in Q}{\text{argmin }} \text{KL}(q(z)||p(z|x))$$

하지만, 이 식 역시 true posterior $p(z|x)$가 있기 때문에 계산할 수가 없다. argmin 안의 KL divergence식을 조금 변형해볼 것이다.

$$\text{KL}(q(z)||p(z|x)) = \mathbb{E}_{z \sim q(z)}[\text{log } q(z) - \text{log } p(z|x)] \\
= \mathbb{E}_{z \sim q(z)}[\text{log } q(z) - \text{log } p(x,z) + \text{log }p(x)]$$

이때, KL divergence는 항상 0보다 크거나 같으므로, 다음이 성립한다.

$$\mathbb{E}_{z \sim q(z)}[\text{log } q(z) - \text{log } p(x,z) + \text{log }p(x)] \geq 0$$

그리고, evidence $p(x)$는 $z$와 무관하므로 밖으로 나올 수 있다. $p(x)$를 제외한 나머지 식을 우변으로 이항해보면 다음과 같은 식이 만들어진다.

$$\text{log }p(x) \geq \mathbb{E}_{z \sim q(z)}[\text{log } p(x,z) - \text{log } q(z)] = \text{ELBO}(q)$$

이때, 우변의 식은 log evidence의 lower bound가 된다. 우변의 식을 evidence lower bound라고 부르며, ELBO라고도 부른다.

한편, 다시 최적화 식으로 돌아와서 보면,

$$q^*(z) = \underset{q(z) \in Q}{\text{argmin }} \text{KL}(q(z)||p(z|x))$$

이 식을 풀면 다음과 같다.

$$q^*(z) = \text{arg}\underset{q^*}{\text{min }} \mathbb{E}_{z \sim q(z)}[\text{log } q(z) - \text{log } p(x,z) + \text{log }p(x)]$$

그리고, evidence $p(x)$는 $z$와 무관한 항이므로 최적화에서 제외할 수 있다.

$$q^*(z) = \text{arg}\underset{q^*}{\text{min }} \mathbb{E}_{z \sim q(z)}[\text{log } q(z) - \text{log } p(x,z)] \\
= \text{arg}\underset{q^*}{\text{min }} -\text{ELBO}(q)$$

따라서, variational inference는 ELBO를 최대화하는 optimization을 푸는 문제로 approximation할 수 있다.