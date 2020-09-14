---
title: 16. Lieklihood for Mixture Models
toc: true
date: 2020-08-04 12:00:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---



# 16. Likelihood for Mixture Models

Mixture model을 정의했으니, 이제 mixture model로 어떤 일을 할 수 있는가를 알아볼 차례이다. Mixture model은 데이터의 생성 프로세스를 모델링하는 데 사용된다. 즉, 데이터를 fitting하는 데 사용된다. 데이터를 생성하는 분포를 모델링한 것을 likelihood distribution이라고 부른다. Mixture model의 fitting은 이 likelihood distribution에서 학습 데이터를 넣었을 때 얻어진 likelihood를 최대화하는 방향으로 이루어진다.



## Hierarchical Representation

Mixture model은 다음처럼 표현이 가능했다.

$$f(x) = \sum_{k=1}^K \omega_k g_k(x)$$

$K$개의 mixture component가 있고, 데이터 $x$는 $K$개중 하나의 component $g_k(x)$로부터 생성된다는 의미의 mixture model이다. 이렇게 바로 표현할 수도 있지만, mixture model이 복잡해지면 이렇게 바로 표현하는 것은 다른사람이 보기에 이해하기도 힘들고, 직관적이지 못할 수도 있다.

Mixture model을 표현하는 또 다른 방법은 hierarchical하게 표현하는 것이다.

$$x_i|c_i \sim g_{c_i} \\
c_i \sim (\omega_1, \omega_2, ..., \omega_K)$$

여기서, $c$는 $K$개중에서 어떤 mixture component를 선택할 것인지에 대한 random variable이고, $g_c$는 $c$번째의 mixture component이다. 이때, $c$번째 mixure component를 선택할 확률은 $\omega_c$이다.

이 표현법으로 $f(x)$를 계산해보면 다음과 같다.

$$f(x) = \sum_{k=1}^K P(x|c_k)P(c_k)  \\ 
= \sum_{k=1}^K g_k(x) \omega_k$$

즉, 맨 처음 식과 일치한다. 이처럼, hierarchical representation은 mixture model의 또 다른 representation이며, 같은 mixture model에 대해 더 편리한 표현법을 제공해준다.



## Sampling from a Mixture Models

Mixture model에서 데이터를 샘플링하기 위해서는(데이터를 생성하기 위해서는) 다음과 같은 과정을 거치면 된다. Mixture model은 hierarchical representation으로 표현되어야 한다.

다음과 같은 mixture model이 있다고 가정해보자.

$$x_i|c \sim g_c \\
c \sim (\omega_1, \omega_2, ..., \omega_K)$$

1. $(\omega_1, \omega_2, ..., \omega_K)$의 distribution으로부터 $c_i$를 샘플링한다.
2. $c_i$에 해당하는 mixture component $g_{c_i}(x)$로부터 $x_i$를 샘플링한다.



## Likelihood Functions

Mixture model은 데이터가 생성되었을 법한 likelihood 모델을 구성하고 그 모델의 파라미터를 학습하는 것을 목적으로 한다. 그러기 위해서는 먼저, 데이터를 수집하고 관찰한다. 그리고, 데이터가 샘플링되어 나왔을 법한 likelihood를 모델링해야 한다.

Mixutre model을 사용하여 모델링할 수 있는 likelihood에는 두 가지가 존재한다.

- Observed data likelihood
- Complete data likelihood



### Observed Data Likelihood

데이터(observation)가 $x_1, x_2, ..., x_n$와 같이 존재한다고 하고, 각 데이터 $x_i$는 같은 distribution에서 왔으며, 독립된 조건 속에서 샘플링되었다고 하자(independent identical distributed; iid). 그리고 데이터 샘플 하나는 다음과 같은 샘플링 과정을 거쳤다고 모델링했다고 해 보자.

$$f(x_i) = \sum_{k=1}^K \omega_k g_k(x_i)$$

이때, likelihood는 모든 data 확률의 joint distribution이므로, 다음과 같이 정의할 수 있다.

$$L(\omega) = \prod_{i=1}^n f(x_i) = \prod_{i=1}^n \sum_{k=1}^K \omega_k g_k(x_i)$$

그러나, 이 형태는 multiplication과 summation이 공존하고, 계산이 다소 힘들다는 단점이 있다. 이 likelihood를 **observed data likelihood**라고 부르는데, 우리가 이상적으로 학습하기 원하는 likelihood는 observed data likelihood지만, 계산의 복잡성 때문에 latent variable을 추가하여 식을 단순화한 **complete data likelihood**를 보통 사용하게 된다.



### Complete Data Likelihood

Observed data likelihood에 latent variable을 추가하여 식을 구성한 likelihood를 말한다. 다음과 같이 데이터가 생성된 과정이 hierarchical model로 모델링되었다고 가정해보자.

$$x_i|c_i \sim g_{c_i} \\
c_i \sim (\omega_1, \omega_2, ..., \omega_K)$$

이때, complete data likelihood는 다음과 같다.

$$L(\omega, c) = \prod_{i=1}^n f(x_i, c_i) = \prod_{i=1}^n \prod_{k=1}^K [g_k(x_i) \omega_k]^{\mathbb{I}(c_i=k)}$$

즉, observed data likelihood에서는 한 개의 데이터 샘플에 대해 likelihood는 다음과 같았는데,

$$P(x_i) = f(x_i) = \sum_{k=1}^K \omega_k g_k(x_i)$$

Complete data likelihood에서는 한 개의 데이터 샘플에 대해 likelihood는 다음과 같다.

$$P(x_i, c_i) = f(x_i, c_i) = \omega_{c_i} g_{c_i}(x_i)$$



## Parameter Identifiability

보통 distribution은 파라미터가 다르면 모양이 서로 다른 distribution이 된다. 예를 들어, 다음과 같은 두 개의 normal distribution은 서로 다른 모양의 distribution이 된다.

$$f_1(x) = \frac{1}{\sqrt{2\pi}} \text{exp}\{ - \frac{x^2}{2} \} \\
f_2(x) = \frac{1}{2\sqrt{2\pi}} \text{exp}\{ -\frac{1}{2} (\frac{x - 1}{2})^2 \}$$

이렇게 파라미터로 두 개의 distribution을 구분할 수 있는 성질을 parameter identifiability라고 부른다.

하지만, 이 성질은 mixture model에는 적용되지 않는 경우가 있다. 즉, 파라미터가 서로 달라도 완전히 일치하는 모양이 나올 수도 있다. 이러한 경우는 세 가지로 나눌 수 있다.

1. Label switching 현상
2. Component split 현상
3. 0-weighted component 현상

Mixture model을 선택할 때는 위 세가지 현상을 주의해야 한다. 똑같은 데이터를 모델링했고, 서로 다른 파라미터를 얻었는데, 모양이 거의 같은 모델을 얻을 수도 있다. 이때는 모델의 mixture component를 줄일 수 없는지도 살펴보아야 하며, 순서만 바뀌지 않았는지 잘 분석해야 한다.



### Label Switching

다음의 두 개의 distribution은 파라미터가 서로 다르지만, 모양이 완전히 같은 분포이다.

$$f_1(x) = 0.7 \cdot \mathbb{N}(0, 1^2) + 0.3 \cdot \mathbb{N}(1, 2^2) \\ 
f_2(x) = 0.3 \cdot \mathbb{N}(1, 2^2) + 0.7 \cdot \mathbb{N}(0, 1^2)$$

얼핏 보면 그냥 파라미터도 똑같은 것처럼 보이지만, 실제로는 다음과 같이 서로 다른 파라미터인 경우가 있다.

$$\begin{cases}
\omega_1 = 0.7, \mu_1 = 0, \sigma _1= 1, \omega_2 = 0.3, \mu_2 = 1, \sigma_2 = 2 \\
\omega_1 = 0.3, \mu_1 = 1, \sigma _1= 2, \omega_2 = 0.7, \mu_2 = 0, \sigma_2 = 1
\end{cases}$$

즉 파라미터가 완전히 스위칭되어 학습된 경우이다.

이럴때는 파라미터가 다르더라도 완전히 모양이 같은 분포가 된다. 이런 현상을 막기 위해 첫 번째 라벨에 대한 평균($\mu_1$)은 두 번째 라벨에 대한 평균 $\mu_2$보다 작아야 한다거나 하는 constraint가 필요할 수도 있다.



### Component Split

다음 두 개의 distribution은 파라미터가 서로 다르지만, 모양이 완전히 같은 분포이다.

$$f_1(x) = 0.7 \cdot \mathbb{N}(0, 1^2) + 0.3 \cdot \mathbb{N}(1, 2^2) \\ 
f_1(x) = 0.7 \cdot \mathbb{N}(0, 1^2) + 0.1 \cdot \mathbb{N}(1, 2^2) + 0.2 \cdot \mathbb{N}(1, 2^2)$$

이것은 흔히 over-estimate때문에 일어나는 현상인데, 실제로는 두 개의 component mixture만으로도 데이터의 distribution을 잡는 데 충분하지만, 세 개의 component로 over-estimate 하려고 하면 이런 현상이 발생할 수 있다.



### 0-weighted Component

다음 두 개의 distribution은 파라미터가 서로 다르지만, 모양이 완전히(거의) 같은 분포이다.

$$f_1(x) = 0.7 \cdot \mathbb{N}(0, 1^2) + 0.3 \cdot \mathbb{N}(1, 2^2) \\ 
f_1(x) = 0.7 \cdot \mathbb{N}(0, 1^2) + 0.2999 \cdot \mathbb{N}(1, 2^2) + 0.0001 \cdot \mathbb{N}(1, 2^2)$$

이러한 현상 역시, mixture component를 많이 잡았을 때 발생할 수 있다.