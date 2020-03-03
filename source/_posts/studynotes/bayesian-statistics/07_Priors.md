---
title: 07. Priors
toc: true
date: 2020-03-01 21:08:07
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---

# Priors



Prior를 어떤 분포로 선택할지에 대해서는 conjugatation을 고려해야 한다. prior으로 선택한 distribution이 likelihood와 곱해져서 posterior가 되었을 때도 그 distribution이 되어야 한다는 의미이다.



## Effective Sample Size

Effective sample size란, 설정한 prior의 영향이 posterior에 영향을 50%만 미치는 순간의 샘플 개수를 말한다. 샘플 개수가 적으면 prior는 posterior에 영향을 많이 미칠 테지만, effective sample size이상의 샘플을 모은다면, prior가 posterior에 미치는 영향이 50% 이내일 것이다.



## Priors in Binomial Likelihood

예를들어, binomial distribution으로 likelihood를 모델링하는 경우, 즉, 제품 생산과정에서 불량의 빈도수 같은 경우는 prior를 beta distribution으로 불량일 확률 $\theta$를 모델링한다.

이때, beta distribution의 parameter $\alpha$와 $\beta$를 어떻게 정해야 할까, beta distribution의 평균은 $\frac{\alpha}{\alpha+\beta}$라는 것을 기억한다. 만약, 우리가 0.3%확률로 불량이 나올 것이라고 믿는다면, $\frac{\alpha}{\alpha + \beta}=0.3$이 되게끔 정하면 된다. 다만, 이러면 경우의 수가 많은데, $\alpha$와 $\beta$값이 커지면 그 믿음에 자신감이 있는 것이다.



30%의 확률로 불량이 있다고 생각해서 불량률 $\theta$에 대해 $\theta \approx \text{beta}(6, 14)$으로 prior를 설정했다고 하자. 그리고 10번의 생산 후 6개의 불량이 나왔다. 이때, $\theta$의 posterior는 $\text{beta}(6+6, 14+4) = \text{beta}(12, 18)$이 된다. **$\alpha$는 불량인 것의 개수와 관련있고, $\beta$는 불량이 아닌 것과 관련이 있는 것이다.** 실제로 계산해봐도 그렇다.
$$
P(\theta) = \frac{\Gamma(6 + 20)}{\Gamma(6)\Gamma(14)}\theta^{6-1}(1-\theta)^{14 - 1} 
$$
$$
P(X|\theta) = \begin{pmatrix} 10 \\ 6 \end{pmatrix}\theta^6(1-\theta)^4
$$


$$
P(\theta|X) = P(X|\theta)P(\theta) = \frac{25!10!}{5!19!6!4!}\theta^{12-1}(1-\theta)^{18-1}
$$

$$
P(\theta|X) \propto \text{beta}(12, 18)
$$





앞의 상수들은 다 상수일 뿐. 어쨌든 beta distribution에 근사된다.



### Effective Sample Size

Binomial likelihood에서 beta distribution을 $\theta$의 prior로 했을 경우, effective sample size는 $\alpha+\beta$가 된다.

Posterior는 다음과 같다.
$$
\text{Posterior}(\theta|X) = \text{Beta}(\alpha+\sum_i^n x_i, \beta + n - \sum_i^n x_i)\\
$$
이때, posterior mean은 $\frac{\alpha + \sum_i^n x_i}{\alpha + \beta + n}$인데, 이를 더 decompose해보면, 다음 식이 나온다.
$$
\frac{\alpha + \sum_i^n x_i}{\alpha + \beta + n} = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \frac{\alpha}{\alpha + \beta} + \frac{n}{\alpha + \beta + n} \cdot \frac{\sum_i^n x_i}{n}
$$
이것은 prior mean과 data mean과의 weighted sum으로 해석할 수 있다. 즉, posterior mean은 prior mean과 data mean에 의해 영향을 받는다. 그런데, 이때, 샘플 개수 $n$이 작으면, prior의 영향력이 커진다. 반면, $n$이 커지면, data의 영향력이 커진다. $n \geq \alpha+\beta$ 일때, prior보다 데이터의 영향력이 커진다. 따라서, effective sample size는 $\alpha + \beta$이다.

Prior를 정의할때, $\alpha, \beta$를 크게 잡던, 작게 잡던, $\alpha$와 $\alpha+\beta$의 비율이 같으면, prior mean은 같지만, 값들이 크면, prior의 영향력이 강해지기 때문에 sample 개수를 많이 모아야 한다.



## Priors in Poisson Distribution

Poisson distribution을 likelihood로 취하는 experiment에 대해서는 parameter가 $\lambda$가 된다. 즉, $\lambda$에 대한 prior가 필요한데, 이때는 Gamma distribution으로 $\lambda$의 prior를 모델링한다. Poisson distribution으로 likelihood를 모델링 할 수 있는 경우, Gamma distribution이 conjugate한 distribution이다.

이때, Gamma distribution의 두 파라미터 $\alpha$와 $\beta$를 정할때, gamma distribution의 평균은 $\frac{\alpha}{\beta}$인 것을 생각하자. **~~$\alpha$는 event 발생 횟수, $\beta$는 총 시행 횟수와 관련이 있다.~~**

이때, poisson이므로, 1번의 시행에서 event가 여러번 발생할 수 있다. 특정 시간 안에 몇 번의 버스가 오는가?



### Effective Sample Size

Beta distribution을 prior로 삼고, posterior도 역시 beta distribution이기 때문에, effective sample size는 $\alpha+\beta$이다.



## Priors in Exponential Distribution

Exponential distribution도 역시 $\lambda$를 파라미터로 하며, gamma distribution을 prior로 하면 conjugate인 prior를 만들 수 있다.



### Effective Sample Size

Gamma prior로 conjugate인 likelihood의 경우, posterior도 gamma distribution이다. 이런 경우, effective sample size는 $\beta$이다.
$$
\text{Posterior}(\lambda|X) = \text{Gamma}(\alpha + \sum_i^n x_i, \beta + n)
$$

$$
\text{mean}(posterior) = \frac{\alpha + \sum_i^n x_i}{\beta + n}
$$

$$
\frac{\alpha + \sum_i^n x_i}{\beta + n} = \frac{\beta}{\beta + n} \cdot \frac{\alpha}{\beta} + \frac{n}{\beta + n} \cdot \frac{\sum_i^n x_i}{n}
$$




## Priors in Normal Distribution

Normal distribution의 파라미터 $\mu$는 $\sigma$에 의존함과 동시에 normal distribution prior에서 conjugate한다. $\sigma$는 주어젔다고 가정하는 경우가 많으며, 그렇지 않을 경우, inverse-gamma distribution에서 conjugate한다.



