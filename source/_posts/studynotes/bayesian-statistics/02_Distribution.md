---
title: 02. Distribution
toc: true
date: 2020-03-01 21:08:01
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---

# Distribution



## Background



### Indicator Function

어떤 조건이 만족했을때, 1을 반환하고, 만족하지 못하면 0을 반환하는 함수이다.
$$
\mathbb{I}_{\text{cond}(x)}(x) = \begin{cases} 1 & \text{if cond(x) is True} \\ 0 & \text{if cond(x) is False} \end{cases}
$$
Indicator function은 다음과 같이 동전 던지기에 대한 확률같은 것을 표현할 때 용이하다.
$$
P(X|\theta) = \theta \cdot \mathbb{I}_{\text{head}} + (1 - \theta) \cdot \mathbb{I}_{\text{tail}}
$$



### Expected Values 

**기댓값**이라고도 불리며, 확률적인 관점에서 본 **"평균"**이다.
$$
E[X] = \sum\limits_{x} p_x x
$$
각 샘플들에 비중치를 곱해서 모두 더한 것이다.

일반적으로 생각할 수 있는 평균값은 모든 샘플들에 같은 비중치를 둔 기댓값과 같다.



### Variance

**분산**이라고도 불리며, 샘플들이 평균 또는 기댓값으로부터 얼마나 떨어져서 분포하는가, 즉, **샘플들이 얼마나 넓게 퍼져있는가**를 나타낸다.
$$
\text{Var}[X] = E[(X-\mu)^2] = \sum\limits_{x} p_x(x - \mu)^2
$$
평균과의 거리의 제곱을 평균한 것인데, 단위가 제곱이 된다. 따라서 단위를 일치시키기 위해 제곱근을 씌우는데, 이를 **standard deviation(표준편차)**라고 부른다.
$$
\text{Std.}[X] = \sigma(X) = \sqrt{\text{Var}[X]}
$$



### Scaling Random Variable vs Many Random Variables

Random variable을 scaling 한다는 것은, 분포를 넓게 피는것을 의미한다. random variable $X$를 $c$배 scaling하는 것은 $cX$로 표기한다.

반면, random variable $X$를 여러번 시행하는 것은 $\sum_i^n X_i$로 표기한다. 둘이 분명히 다른데, $cX$의 경우, 1번 샘플링하는 것이고, $\sum X$는 여러번 샘플링 하는 것이다.

예를들어, $X$가 베르누이 분포를 따르고, 1일 확률이 0.7 이라면, $10X$는 0일 확률이 0.3, **10일 확률**이 0.7인 것이다.

반면, $\sum_i^{10} X_i$는 1이 나올 확률이 0.7인 분포에서 10번 샘플링하는 것이다.



## Discrete Distribution



### Bernoulli Distribution

Sample space의 크기가 2(event 개수가 2개)라고 추정되는 random variable $X$가 있을때, 이 $X$는 Bernoulli distribution(베르누이 분포)을 따른다. 베르누이 분포를 따르는 random variable의 1회 experiment을 베르누이 시행이라고 한다.
$$
\text{Bern}(X|\theta) = \theta*\mathbb{I}_{X=1} + (1 - \theta) * \mathbb{I}_{X=0}
$$
**Expected value:**
$$
E[X] = \theta
$$
**Variance:**
$$
\sigma^2(X) = \theta * (1 - \theta)^2 + (1-\theta)*(0 - \theta)^2 = \theta(1-\theta)
$$


### Binomial Distribution

베르누이 분포를 따르는 experiment를 여러번 시행했을 때, 한 결과가 몇 번이 나왔는가에 대한 분포이다. 흔히, 동전을 10번 던졌을때, 앞면이 몇 번 나오는지에 대한 분포라고 이해하면 편하다. 동전 던지기 1회는 베르누이 시행이다.
$$
\text{Binom}(n, x|\theta) = \begin{pmatrix} n \\ x \end{pmatrix} \theta^x(1-\theta)^{n-x}
$$
**Expected value:**
$$
E[X] = n\theta
$$
**Variance:**
$$
\sigma^2(X) = n\theta(1-\theta)
$$


### Geometric Distribution

베르누이 시행을 여러번 반복하는데, 어떤 event가 최초로 일어날때 까지의 시행한 횟수는 geometric distribution을 따른다. (기하분포). $\theta$는 베르누이 시행 1회에서 그 event가 성공할 확률이다.
$$
\text{Geom}(x|\theta) = \theta * (1-\theta)^{x-1}
$$
**Expected value:**
$$
E[X] = \frac{1}{\theta}
$$


### Multinomial Distribution

시행이 베르누이 시행이 아니라, 여러 개의 event가 나올 수 있는 시행일 때의 binomial distribution을 의미한다.
$$
\text{Multinom}(n, x_1,...,x_{n}|\theta_1, \theta_2, ..., \theta_{n}) = \begin{pmatrix} n \\ x_1, x_2, ..., x_n \end{pmatrix}\theta_1^{x_1}*\theta^{x_2}* \cdots*\theta^{x_n}
$$


### Poisson Distribution

포아송 분포는, 어느 시간 간격 내에 그 event가 몇 번 일어날지에 대한 분포이다. 해당 시간 간격동안에 event가 발생하는 횟수를 $\lambda$라고 하면, 다음과 같다.
$$
\text{Poisson}(x|\lambda) = \frac{\lambda^xe^{-\lambda}}{x!}
$$
Expected value:
$$
E[X] = \lambda
$$
(애초에 $\lambda$ 정의가 그냥 기댓값이다.)

여기서, 만약에 binomial distribution을 따르는데, 동전이 앞면이 나올 확률이 너무나도 희박하고, 동전 던지기 experiment를 무한번 한 경우, 그 무한번의 experiment를 일정 기간의 시간이라고 간주하게 되면 poisson distribution와 같다.



## Continuous Distribution



Sample space가 continuous한 경우의 distribution을 말함.



### Exponential Distribution

특정 일이 일어날 때 까지 걸린 시간 또는 기다린 시간의 분포는 exponential distribution을 따른다.
$$
\text{Exp}(\lambda) = \lambda e^{-\lambda x} \mathbb{I}_{ x \geq 0 }
$$
여기서 $\lambda$는 어떤 시간 동안에 사건이 발생하는 횟수의 비율을 말한다. 예를 들어, 10분 동안 버스가 3대 오면 $\lambda = 0.3$이다(시간을 10분 단위로 했을 때).



### Gamma Distribution

버스가 올때까지 걸리는 시간을 측정하는 시행이 여러 번 있고, 그들의 총합 시간은 gamma distribution을 따른다. 쉽게 말해서, Gamma distribution을 따르는 $Y$는 exponential distribution을 따르는 $X_i$의 합과 같다.
$$
Y = \sum_i X_i \\
p(y|\alpha,\beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)}y^{\alpha-1}e^{-\beta y} \mathbb{I}_{y \geq 0}(y)
$$
감마분포는 $\alpha$와 $\beta$를 파라미터로 삼으며, $\alpha = n$, $\beta = \lambda$가 된다.

$$\alpha$$는 shape parameter로, $\alpha=1$이면, exponential distribution이 된다. 또한, $\alpha$가 0에 가까워질수록 right-skewed가 된다. $$\alpha$$가 커질수록 normal distribution에 가까워지면서 skewness가 줄어든다(한쪽으로 치우치지 않는다).

$\beta = \lambda$는 rate parameter로, $\theta = \frac{1}{\beta}$는 scale parameter이다. 서로 역수 관계이며, 감마 분포를 표기할때, $(\alpha, \beta)$로 parameterize하기도 하고 $(k, \theta)$로 parameterize하기도 한다. $\alpha=k$이지만, $\beta = \frac{1}{\theta}$이다. $\theta$는 scale parameter로, rate의 역수이다. Scale parameter는 분산의 scaling 정도이며, 클수록 넓게 퍼진다. 즉, rate가 작을수록 넓게 퍼지며, random variable $X$의 $c$배 scale인 $cX$는 $(k, c\theta)$가 되는 셈.

$\sum_i^{n} X_i$는 $(nk, \theta)$ 효과를 얻는다! 이걸 보면 $\alpha$가 exponential의 횟수와 관련이 있을지도 모른다.



**Expected Value:**
$$
E[X] = \frac{\alpha}{\beta}
$$
**Variance:**
$$
\sigma^2(X) = \frac{\alpha}{\beta^2}
$$




### Uniform Distribution

모든 sample space범위의 단위 interval에서 확률이 같다.
$$
\text{Uni}(X) = \frac{1}{b-a}\mathbb{I}_{ a \leq x \leq b }
$$
**Expected Value:**
$$
E[X] = \frac{a+b}{2}
$$
(a~b까지 나올 확률이 같으므로 샘플링 여러번 하다 보면 평균값은 중앙값인 $\frac{a+b}{2}$이 된다.)

**Variance:**
$$
\sigma^2(X) = \frac{(b-a)^2}{12}
$$




### Beta Distribution

Sample space가 0과 1 사이인 분포. 따라서 확률을 모델링할때 이용하기도 한다.
$$
\text{Beta}(\alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$
여기서, $\Gamma(x) = (x-1)!$이다. 앞의 $\Gamma$ term들을 풀어보면 binomial coefficient와 비슷하게 생겨서 나중에 binomial distribution을 적분할때, gamma distribution을 이용하면 매우 유용하다.

**Expected Value:**
$$
E[X] = \frac{\alpha}{\alpha + \beta}
$$
**Variance:**
$$
\sigma^2(X) = \frac{\alpha\beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}
$$



### Normal Distribution

정규 분포.
$$
p(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$
**Expected Value:**
$$
E[X] = \mu
$$
**Variance:**
$$
\text{Var}(X) = \sigma^2
$$
만약, iid(independent identical distribution)에서 샘플링된 여러 샘플들, 즉, 똑같은 분포로부터 독립적인 시행으로 샘플링한 여러 샘플들의 평균 $\bar{X}$은 정규 분포를 따른다. 샘플의 개수를 $n$, 그 샘플들을 샘플링한, 즉, 하나의 샘플을 샘플링한 분포의 실제 평균을 $\mu$, 분산을 $\sigma^2$이라고 했을 때, 다음을 만족한다.
$$
\bar{X} \sim \mathbb{N}(\mu, \frac{\sigma^2}{n})
$$
이를 central limit theorem(CLT) 이라고 부른다. 평균 $\mu$는 추정 대상이라서 모르지만, $\sigma^2$는 샘플들의 분산으로 대체한다. 즉, 우리가 샘플링한샘플들의 평균은 실제 샘플 평균으로부터 어느정도 가깝다는 것이다. 또한, 분산은 샘플수에 반비레하는데, 이는 샘플이 많을수록, 진짜 샘플 평균에 가까워진다는 것을 알 수 있다.



### t-Distribution

Student-t 분포, test용 분포라고도 한다. 

CLT에서, 샘플 평균의 분포를 standarize시키면 standard normal distribution이 아니라, t-distribution이 나온다. 분산값인 $\sigma^2$가 샘플 분산인 $S^2$으로 대체되기 때문이다.
$$
S^2 = \frac{\sum_i (\bar{X}-X_i)^2}{n-1}
$$
이러면, $\bar{X}$의 분포는 더 이상 normal distribution이 아닌, t-distribution을 따른다. $\nu = n-1$이라고 했을 때,
$$
\text{t}(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu\pi}}(1+\frac{x^2}{\nu})^{-(\frac{\nu+1}{2})}
$$
이때, $\nu$는 자유도, degree of freedom이라고 부른다.

**Expected Value: **
$$
E[X] = 0 ~~~\text{if } \nu \geq 1
$$
**Variance:**
$$
\text{Var}(X) = \frac{\nu}{\nu-2} ~~~~\text{if } \nu \geq 2
$$
