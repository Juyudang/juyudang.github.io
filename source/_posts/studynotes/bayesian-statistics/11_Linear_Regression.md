---
title: 11. Linear Regression
toc: true
date: 2020-03-01 22:08:11
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# Linear Regression



선형 회귀라고도 불리며, 예측해야할 dependent variable이 continuous할때, 유용하다.

선형 회귀는 다음과 같다.
$$
y_i \sim \mathbb{N}(\mu_i, \sigma^2) \\
\mu_i = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k
$$
또는 다음과 같이 표현할 수 있다.
$$
y_i = \beta_0 + \beta_1 x_1 + \cdots + \beta_k x_k + \epsilon \\
\epsilon \sim \mathbb{N}(0, \sigma^2)
$$
즉, 예측값 $y_i$는 일직선 값에 오차값(residual) $\epsilon$을 더한 값이며, 이 residual값은 normal distribution을 따른다.

베이지안 통계에서는 $\beta$들은 분포를 갖는 random variable들이다. 따라서, 선형 회귀를 학습시킬때, 각 $\beta$에 대해 prior를 설정하고 posterior를 계산하게 된다.

각 $\beta$값에는 normal prior를 주는게 보통이지만, 다른 prior도 상관없다.

단, 어떤 variable $x_i$가 $y_i$에 영향을 주는 놈인지 알고 싶다면, $\beta_i$의 prior로 **Laplace prior**를 설정하기도 한다. Laplace distribution은 double exponential 이라고도 불리며, 0점에서 뾰족한 모양이고 $y$축 대칭이다.

만약, 어떤 $i$번째 $\beta_i$의 posterior 분포가 그냥 Laplace처럼 0점에 뾰족한 모양에 가깝다면, 그 $\beta_i$에 대응되는 $x_i$는 영향력이 거의 없다고 할 수 있다. 이런 방법을 **Lasso** 라고 부른다.

JAGS 문법으로 표현하면 다음과 같다.

```R
# rjags

mod.string <- "model {
    # likelihood
    for (i in 1:length(y)) {
        y[i] ~ dnorm(mu[i], prec)
        mu[i] = b0 + b[1]*x1[i] + b[2]*x2[i]
    }
    
    # prior
    # prec의 사후샘플들의 effective size가 5, variability가 2라고 기대하는 경우의
    # inverse-gamma는 다음과 같다.
    prec ~ dgamma(5/2, 5*2/2)
    b0 ~ dnorm(0, 1e-6)
    for (i in 1:2) {
        b[i] ~ dnorm(0, 1e-6)
    }
}"
```





## Poison Regression

선형 회귀의 일종으로, response variable인 $y$가 count 값인 경우에 사용되는 경우가 있다. poison 분포는 0보다 크거나 같은 값을 도메인으로 가지므로, $y$역시 0보다 같거나 큰 값이어야 한다. 반대로, $y$의 범위가 0이상이라면, Poison regression을 생각해 볼 수 있다.

Poison regression은 likelihood가 poison distribution으로 모델링된 형태이다. 그런데, 이 경우, $y$가 0 이상 값이므로, 선형 회귀처럼 $y_i = \beta_0 + \beta_1 x_{1,i}$로 할 수 없다. 대신, $y$를 적절히 변형해서 $[-\infin, \infin]$범위로 만들어 준다면, 선형 함수로 적용이 가능할 것이다. 이때 사용하는 것이 $\text{log}$함수이다. 즉, 다음과 같다.
$$
\text{log}~y_i = \beta_0 + \beta_1x_{1,i} + \beta_2 x_{2,i} + \cdots \\
y_i = \text{exp}(\beta_0 + \beta_1x_{1,i} + \beta_2 x_{2,i} + \cdots)
$$
위의 방법으로 모델링한다. Poison distribution의 파라미터 $\lambda$는 곧 분포의 기댓값이다. 즉, $y_i = \lambda_i$로 생각하면 된다.  JAGS 문법으로 표현하면 다음과 같다.

```R
# rjags

mod.string <- "model {
    # likelihood
    for (i in 1:length(y)) {
        y[i] ~ dpois(lambda[i])
        log(lambda[i]) = b0 + b[1]*x1[i] + b[2]*x2[i]
    }
    
    # prior
    b0 ~ dnorm(0, 1e-6)
    for (i in 1:2) {
        b[i] ~ dnorm(0, 1e-6)
    }
}"
```



