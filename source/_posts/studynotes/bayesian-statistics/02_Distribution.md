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

앞서 확률이 무엇인지 정의하고 어떻게 확률을 계산하는지 알아보았다. 이번에는 각 이벤트가 일어날 확률 분포에 대해 알아볼 것이다.

# Distribution

**확률 분포(probability distribution)**는 sample space $\mathcal{X}$를 0과 1사이의 실수값으로 매핑시켜주는 **함수**이다. 어떤 random variable $X$에 대해 $X$가 sample space $\mathcal{X}$를 가지고 있을 때, random variable $X$에 대한 확률 분포 $\mathbb{P}_X$는 다음과 같은 함수이다.

$$
\mathbb{P}_X: \mathcal{X} \rightarrow \mathbb{R}
$$

이때, $X$가 discrete 이면, probability mass function(PMF)로, continuous이면, probability density function(PDF)로 확률 분포를 표현할 수 있으며, 소문자 $p$로 적는다.

확률 분포의 특징은 도메인 $\mathcal{X}$의 모든 element에 대해 일어날 확률을 더하면 1이 나온다는 것이다.

$$
\sum_{x \in \mathcal{X}} p(x) = 1
$$

또는 continuous random variable의 경우, 다음처럼 표기하기도 한다.

$$
\int_{x \in \mathcal{X}} p(x) dx = 1 \\
\int_{x \in \mathcal{X}} d\mathbb{P}(x) = 1
$$

## Indicator Function

Indicator 함수란, 어떤 조건이 만족했을때, 1을 반환하고, 만족하지 못하면 0을 반환하는 함수이다. 

$$
\mathbb{I}_{\{\text{cond}(x)\}}(x) = \begin{cases} 1 & \text{if cond(x) is True} \\ 0 & \text{if cond(x) is False} \end{cases}
$$

Indicator function은 다음과 같이 동전 던지기에 대한 확률값을 표현할 때 용이하다. 

$$
P(X|θ) = θ * 𝕀_{\text{head}} + (1 − θ) * 𝕀_{\text{tail}}
$$

즉, 앞면(head)이면, $\theta$만 살아남고, 뒷면(tail)이면, $(1 - \theta)$가 살아남는다.

## Expected Values

**기댓값**이라고도 불리며, 확률적인 관점에서 본 **“평균”**이다. 어떤 random variable $X$가 있고, 이 random variable $X$가 sample space $\mathcal{X}$를 가지고 있을 때, $X$의 기댓값 $\mathbb{E}[X]$는 다음과 같이 정의한다.

$$
\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x \cdot p(x) = \int_{x \in \mathcal{X}} x \cdot p(x) dx
$$

즉, 기댓값은 각 샘플에 확률을 곱해서 모두 더해준 값이다. 일반적인 산술 평균을 생각해보면, 수학점수가 98, 국어점수가 88, 영여 점수가 94일때, 이들의 평균은 $(98+88+94)/3 \approx 93$인데, 어떻게 보면, 각 과목이 똑같은 가중치를 주어서 합한 것과 같음을 알 수 있다($0.3 \times 98 + 0.3 \times 88 + 0.3 \times 94$).

기댓값은 확률 분포가 $\mathcal{X}$의 어느 부분에 위치해 있는지(sample space의 어느 부근의 샘플이 주로 뽑혔는지)에 대한 정보, 즉, 위치정보를 알 수 있는 중요한 measure이다.

## Variance

**분산**이라고도 불리며, 샘플들이 평균 또는 기댓값으로부터 얼마나 떨어져서 분포하는가를 나타낸다.

어떤 random variable $X$가 있고, $X$가 sample space $\mathcal{X}$를 가질 때, $X$의 분산 $\text{Var}[X]$은 다음과 같이 정의한다.

$$
\text{Var}[X] = \mathbb{E}[(X-\mu)^2] = \sum_{x \in \mathcal{X}} (x - \mu)^2 \cdot p(x) = \int_{x \in \mathcal{X}} (x - \mu)^2 \cdot p(x) dx
$$

이때, $\mu = \mathbb{E}[X]$이다.

수식으로만 보면, 기댓값으로부터 각 샘플의 거리의 제곱의 기댓값이다. 즉, 각 샘플이 평균으로부터 떨어진 거리의 제곱 평균값이다.

이때, variance의 단위는 sample space 상에서의 **거리의 제곱**이다. 따라서 단위를 sample space와 일치시키기 위해 제곱근을 씌워서 표현하기도 하는데, 이를 **standard deviation(표준편차)**라고 부른다.

$$
\text{Std.}[X] = \sigma(X) = \sqrt{\text{Var}[X]}
$$

분산(variance)은 확률 분포가 sample space상에서 얼만큼 모여있는지, 즉, 분포의 모양을 판단하는데 중요한 measure로 사용된다.

## Scaling Random Variable vs Many Random Variables

Random variable을 scaling 한다는 것은, 분포를 넓게 펼치거나 좁히는 것을 의미한다. random variable $X$를 $c$배 scaling하는 것을 $cX$로 표기한다.

반면, random variable $X$를 여러번 시행하는 것은 $\sum_i^n X_i$로 표기한다. 둘이 분명히 다른데, $cX$의 경우, 1번 샘플링하는 것이고, $∑X$는 여러번 샘플링 하는 것이다.

예를들어, $X$가 베르누이 분포를 따르고, 1일 확률이 0.7 이라면, $10X$는 0일 확률이 0.3, **10일 확률**이 0.7인 것이다.

반면, $\sum_i^{10} X_i$는 1이 나올 확률이 0.7인 분포에서 10번 샘플링하는 것이다.

## Discrete Distribution

지금까지 확률 분포의 정의와 기본적인 measure를 살펴보았다. 지금부터는 random variable $X$이 discrete variable인 경우, 즉, $X$의 sample space $\mathcal{X}$이 discrete space인 경우에 자주 사용되는 확률 분포 몇가지를 살펴보려고 한다.

### Bernoulli Distribution

Sample space의 크기가 2(event 개수가 2개)라고 추정되는 random variable $X$가 있을때 주로 사용되는 확률 분포로 Bernoulli distribution(베르누이 분포)이 있다. 베르누이 분포를 따르는 random variable의 1회 experiment을 베르누이 시행이라고 한다.

예를 들어, 동전 던지기 1회의 sample space는 앞면 또는 뒷면 총 2개의 이벤트로만 구성되어 있으며, 동전을 한 번 던지는 시행을 베르누이 시행이라고 부른다. 동전던지기 1회에 대한 sample space를 매핑하는 random variable $X$에 대해 다음과 같이 베르누이 분포로 모델링할 수 있다.

$$
\text{Bern}(X|θ) = θ * 𝕀_{\{X = 1\}} + (1 − θ) * 𝕀_{\{X = 0\}}
$$

이때, $\theta$는 동전이 앞면이 나올 확률이며, $1- \theta$는 뒷면이 나올 확률이다. 베르누이 분포에서는 파라미터가 $\theta$ 1개이다.

**Expected value:**

Bernoulli distribution의 기댓값은 다음과 같다.

$$
E[X] = θ
$$

동전 던지기를 여러번 시행했을 때, 평균적으로 앞면이 나오는 횟수가 기댓값이므로 $\theta$ 그 자체가 기댓값이 된다.

**Variance:**

Bernoulli distribution의 분산은 다음과 같이 계산한다.

$$
σ^2(X) = θ * (1 − θ)^2 + (1 − θ) * (0 − θ)^2 = θ(1 − θ)
$$

이때, 우변의 첫 번째 term에서 $\theta = p(1)$이고, $(x - \mu)^2 = (1 - \theta)^2$이 된다. 두번째 term 마찬가지로 해석이 가능하다. $1-\theta = p(0)$이고, $(x - \mu)^2 = (0 - \theta)^2$가 된다.

### Binomial Distribution

베르누이 분포를 따르는 experiment를 여러번 시행했을 때, 한 결과가 몇 번이 나왔는가에 대한 분포이다. 흔히, 동전을 10번 던졌을때, 앞면이 몇 번 나오는지에 대한 분포라고 이해하면 편하다. 즉, binomial distribution을 따르는 random variable 은 bernoulli random variable $X_i$를 여러개 더한 것이다.

$$
X = X_1 + X_2 + \cdots
$$

Binomial distribution은 다음과 같이 정의된다.

$$
\text{Binom}(n, x|\theta) = \begin{pmatrix} n \\ x \end{pmatrix} \theta^x(1-\theta)^{n-x}
$$

이때, $n$은 binomial distribution을 구성하고 있는 Bernoulli random variable의 개수이며, 동전을 10번 던져서 앞면이 나올 개수를 모델링하는 것이라면, $n=10$이 된다. $\theta$는 1회의 동전 던지기에서 앞면이 나올 확률이다.

**Expected value:**

Binomial distribution은 $n$번의 Bernoulli 시행에서 한 이벤트가 몇 번 일어나는지에 대한 평균이므로 다음과 같이 정의된다.

$$
E[X] = nθ
$$

**Variance:**

$$
σ^2(X) = nθ(1 − θ)
$$

### Geometric Distribution

베르누이 시행을 여러번 반복하는데, 어떤 event가 최초로 일어날때 까지의 시행한 횟수는 geometric distribution을 따른다. (기하분포). *θ*는 베르누이 시행 1회에서 그 event가 성공할 확률이다. 

$$
\text{Geom}(x|θ) = θ * (1 − θ)^{x − 1}
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

포아송 분포는, 어느 시간 간격 내에 그 event가 몇 번 일어날지에 대한 분포이다. 해당 시간 간격동안에 event가 발생하는 횟수를 *λ*라고 하면, 다음과 같다.

$$
\text{Poisson}(x|\lambda) = \frac{\lambda^xe^{-\lambda}}{x!}
$$

Expected value:

$$
E[X] = λ
$$

(애초에 $λ$ 정의가 그냥 기댓값이다.)

여기서, 만약에 binomial distribution을 따르는데, 동전이 앞면이 나올 확률이 너무나도 희박하고, 동전 던지기 experiment를 무한번 한 경우, 그 무한번의 experiment를 일정 기간의 시간이라고 간주하게 되면 poisson distribution와 같다.

## Continuous Distribution

Sample space가 continuous한 경우의 distribution을 말함.

### Exponential Distribution

특정 일이 일어날 때 까지 걸린 시간 또는 기다린 시간의 분포는 exponential distribution을 따른다. 

$$
\text{exp}(λ) = λe ^{− λx}𝕀_{x ≥ 0} 
$$

여기서 *λ*는 어떤 시간 동안에 사건이 발생하는 횟수의 비율을 말한다. 예를 들어, 10분 동안 버스가 3대 오면 $λ$ = 0.3이다(시간을 10분 단위로 했을 때).

### Gamma Distribution

버스가 올때까지 걸리는 시간을 측정하는 시행이 여러 번 있고, 그들의 총합 시간은 gamma distribution을 따른다. 쉽게 말해서, Gamma distribution을 따르는 *Y*는 exponential distribution을 따르는 *Xi*의 합과 같다.  

$$
Y = \sum_i X_i \\
p(y|\alpha,\beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)}y^{\alpha-1}e^{-\beta y} \mathbb{I}_{\{y \geq0\}}(y)
$$

감마분포는 $α$와 $β$를 파라미터로 삼으며, $α = n, β = λ$가 된다.

$α$는 shape parameter로, $α = 1$이면, exponential distribution이 된다. 또한, $α$가 0에 가까워질수록 right-skewed가 된다. $α$가 커질수록 normal distribution에 가까워지면서 skewness가 줄어든다(한쪽으로 치우치지 않는다).

$β = λ$는 rate parameter로, $\theta = \frac{1}{\beta}$는 scale parameter이다. 서로 역수 관계이며, 감마 분포를 표기할때, $(α, β)$로 parameterize하기도 하고 $(k, θ)$로 parameterize하기도 한다. $α = k$이지만, $\beta = \frac{1}{\theta}$이다. $θ$는 scale parameter로, rate의 역수이다. Scale parameter는 분산의 scaling 정도이며, 클수록 넓게 퍼진다. 즉, rate가 작을수록 넓게 퍼지며, random variable $X$의 $c$배 scale인 $cX$는 $(k, cθ)$가 되는 셈.

$\sum_i^{n} X_i$는 $(nk, θ)$ 효과를 얻는다! 이걸 보면 $α$가 exponential의 횟수와 관련이 있을지도 모른다.

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
\text{Uni}(X) = \frac{1}{b-a}\mathbb{I}_{\{ a \leq x \leq b \}}
$$

**Expected Value:** 

$$
E[X] = \frac{a+b}{2}
$$

($a$~$b$)까지 나올 확률이 같으므로 샘플링 여러번 하다 보면 평균값은 중앙값인 $\frac{a+b}{2}$이 된다.)

**Variance:** 

$$
\sigma^2(X) = \frac{(b-a)^2}{12}
$$

### Beta Distribution

Sample space가 0과 1 사이인 분포. 따라서 확률을 모델링할때 이용하기도 한다. 

$$
\text{Beta}(\alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}
$$

여기서, $Γ(x) = (x − 1)!$이다. 앞의 $Γ$ term들을 풀어보면 binomial coefficient와 비슷하게 생겨서 나중에 binomial distribution을 적분할때, gamma distribution을 이용하면 매우 유용하다.

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
E[X] = μ
$$

**Variance:** 

$$
Var(X) = σ^2 
$$

만약, $iid$(independent identical distribution)에서 샘플링된 여러 샘플들, 즉, 똑같은 분포로부터 독립적인 시행으로 샘플링한 여러 샘플들의 평균 $X̄$은 정규 분포를 따른다. 샘플의 개수를 $n$, 그 샘플들을 샘플링한, 즉, 하나의 샘플을 샘플링한 분포의 실제 평균을 $μ$, 분산을 $σ^2$이라고 했을 때, 다음을 만족한다. 

$$
\bar{X} \sim \mathbb{N}(\mu, \frac{\sigma^2}{n})
$$

이를 central limit theorem(CLT) 이라고 부른다. 평균 $μ$는 추정 대상이라서 모르지만, $σ^2$는 샘플들의 분산으로 대체한다. 즉, 우리가 샘플링한샘플들의 평균은 실제 샘플 평균으로부터 어느정도 가깝다는 것이다. 또한, 분산은 샘플수에 반비레하는데, 이는 샘플이 많을수록, 진짜 샘플 평균에 가까워진다는 것을 알 수 있다.

### t-Distribution

Student-t 분포, test용 분포라고도 한다.

CLT에서, 샘플 평균의 분포를 standarize시키면 standard normal distribution이 아니라, t-distribution이 나온다. 분산값인 $σ^2$가 샘플 분산인 $S^2$으로 대체되기 때문이다. 

$$
S^2 = \frac{\sum_i (\bar{X}-X_i)^2}{n-1}
$$

이러면, $X̄$의 분포는 더 이상 normal distribution이 아닌, t-distribution을 따른다. $ν = n − 1$이라고 했을 때, 

$$
\text{t}(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu\pi}}(1+\frac{x^2}{\nu})^{-(\frac{\nu+1}{2})}
$$

이때, $ν$는 자유도, degree of freedom이라고 부른다.

**Expected Value:** 

$$
E[X] = 0 \text{ if } ν ≥ 1 
$$

**Variance:** 

$$
\text{Var}(X) = \frac{\nu}{\nu-2} ~~~~\text{if } \nu \geq 2
$$