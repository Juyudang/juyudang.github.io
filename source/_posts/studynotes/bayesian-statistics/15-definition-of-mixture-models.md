---
title: 15. Definition of Mixture Models
toc: true
date: 2020-08-01 15:26:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# 15. Definition of Mixture Models



Normal distribution, poisson distribution, exponential distribution 등 여러가지 기본 distribution들로 모델링이 가능한 데이터가 있고, 그렇지 않은 데이터가 있다. 예를 들어, 남자와 여자의 키의 분포를 생각해보면, 여자의 키는 160근처에서 평균을 이루면서 normal distribution을 이루고, 남자의 분포는 175쯤에서 평균을 이루면서 normal distribution을 이룬다고 해 보자.

이때, 사람들의 키 분포는 두개의 봉우리가 있는 multi-modal distribution이 되고, 일반적인 distribution으로는 모델링이 불가능하다. 데이터에 사실상 두 개의 population이 존재하기 때문이다.

이번 section에서는 이러한 상황에서 데이터를 모델링하는 데 사용할 수 있는 도구인 mixture model에 대해서 다룬다.



## Definition of Mixture Models

Mixture model은 여러개의 population이 합친 데이터를 모델링할 수 있는 도구이다. 하나의 population에 하나의 distribution을 모델링한 후, 각 distribution을 weighted sum한 것이 mixture model이다.

![](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image1.png)

Mixutre model의 probability density(또는 probability mass) function $f(x)$는 다음과 같이 정의한다.

$$
f(x) = \sum_{k=1}^K \omega_k g_k(x)
$$

이 mixture model에는 $K$개의 분포 $g_k(x)$를 합친 형태이며, 이 분포들을 $\omega_k$로 weighted sum한 형태이다.

이때, 다음과 같은 제약조건이 붙는다.

$$
\sum_{k=1}^K \omega_k = 1
$$

모든 $g_k(x)$가 올바른 probability density라고 할때($\int_{-\infty}^{\infty} g_k(x) dx = 1$), 위 제약조건이 있으면 다음을 만족할 수 있게 된다.

$$
\int_{-\infty}^{\infty} f(x) dx = 1
$$

가장 간단한 예제 중 하나인 사람들의 키 분포를 예로 들어보자. 사람들은 남자와 여자 두 가지 성별로 그룹핑할 수 있으며, 남자의 키 분포를 다음과 같다고 가정해보자.

$$
g_{ \text{male} }(x) = \frac{1}{ \sigma \sqrt{2\pi} } \text{exp}\{ -\frac{(x - \mu_{\text{male}})^2}{2\sigma^2} \}
$$

그리고, 여자의 키 분포를 다음과 같다고 가정해보자.

$$
g_{\text{female}}(x) = \frac{1}{\sigma\sqrt{2\pi}} \text{exp}\{ -\frac{(x - \mu_{\text{female}})^2}{2\sigma^2} \}
$$

그럼, 사람들의 키 분포는 다음과 같이 표현할 수 있다.

$$
f_{\text{height}}(x) = \omega_{\text{male}}g_{\text{male}}(x) + \omega_{\text{female}}g_{\text{female}}(x)
$$

이 예제에서는 남, 여의 키 분포의 분산 $\sigma^2$이 같다고 가정했다. 그러나, $\sigma$도 다르게 둘 수도 있다.



## Expectation of Mixture Models

분포의 기댓값은 분포의 특성을 파악하는 데 있어서 가장 중요한 statistics 중 하나이다.

Mixture model의 기댓값은 다음과 같이 정의될 수 있다.

$$
\mathbb{E}__f [X] = \int_{ -{\infty} }^{ \infty } x f(x) dx
$$

이때, $f(x)$를 mixture density로 치환해보면 다음과 같다.

$$
\mathbb{E}__f [X] = \int_{ -\infty }^{ \infty }x \sum_{k=1}^K \omega_k g_k(x) dx \\
= \sum_{k=1}^K \omega_k \int_{ -\infty }^{ \infty }x  g_k(x) dx \\
= \sum_{k=1}^K \omega_k \mathbb{E}_g [X]
$$

즉, mixture density의 기댓값은 각 구성원 분포 기댓값의 weighted sum과 같다.



## Variance of Mixture Models

분포의 분산또한 평균과 마찬가지로 분포의 특성을 파악하는데 아주 중요하다.

Mixture model의 분산은 다음과 같다.

$$
Var[X] = \int_{-\infty}^{\infty}(x - \mathbb{E}[X])^2 f(x) dx
$$

이때, 다시 $f(x)$를 mixture density 식으로 치환해보자.

$$
Var_f [X] = \int_{-\infty}^{\infty}(x - \mathbb{E}_f [X])^2 \sum_{k=1}^k\omega_k g_k(x) dx \\
= \sum_{k=1}^k \omega_k \int_{-\infty}^{\infty} (x - \mathbb{E}_f[X])^2 g_k(x) dx \\
= \sum_{k=1}^k \omega_k \int_{-\infty}^{\infty} (x^2 g_k(x) - 2x\mathbb{E}_f [X] g_k(x) + (\mathbb{E}_f [X])^2 g_k(x)) dx \\
= \sum_{k=1}^k \omega_k \mathbb{E}_g[X^2] - 2\mathbb{E}_f [X]\sum_{k=1}^k \omega_k \int_{-\infty}^{\infty} x g_k(x) + (\mathbb{E}_f [X])^2 \sum_{k=1}^k \omega_k \int_{-\infty}^{\infty} g_k(x) dx \\
= \sum_{k=1}^k \omega_k \mathbb{E}_g[X^2] - 2 (\mathbb{E}_f [X])^2 + (\mathbb{E}_f [X])^2 \\
= \sum_{k=1}^k \omega_k \mathbb{E}_g[X^2] -  (\mathbb{E}_f [X])^2 \\
= \sum_{k=1}^k \omega_k [Var_{g_k}[X] + (\mathbb{E}_{g_k} [X])^2] - (\mathbb{E}_f [X])^2
$$


## Applications of Mixture Models

가장 자주 사용되는 application으로는 다음과 같은 경우가 있다.

- 데이터가 multi-modal인 경우
    - Multi-modal인 경우에는 modal 수만큼 density를 mixture하는 경우가 많다.
    - 0-inflated distribution(0이 크게 솟아있는 경우)인 경우도 0인 경우와 0이 아닌 경우로 distribution을 mixture하게 된다. 이때, 0인 경우는 point-mass function(dirac delta)을 사용한다.
- 데이터가 skewed 형태인 경우
- 데이터가 heavy tail을 가진 normal 형태인 경우