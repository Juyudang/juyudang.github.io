---
title: 01. Probability
toc: true
date: 2020-03-01 21:08:00
tags:
	- StudyNotes
	- BayesianStatistics
categories:
	- Study Notes
	- Bayesian Statistics
---

# Probability



## Background



### Experiments

시행이라고도 불리며, 데이터 샘플 한개를 샘플링하는 행위를 말한다.



### Sample Space

Sample space란, 데이터의 도메인을 의미한다. **어떤 시행에 의해 나올 수 있는 모든 결과의 집합이다.**

즉, 동전 던지기를 10번 하는 experiment에서 앞면의 개수를 표현하는 sample space는 $\{0, 1, 2, ..., 10\}$을 포함한다. 주의할 점은, sample space가 정확히 저거다 라고 말할 수는 없다. Sample space에서 샘플링될 확률이 높아서 샘플링된 데이터도 있지만, 확률이 극도로 낮아서 샘플링되지 못하고 있는 데이터도 있기 마련이다.

한마디로 말하면, sample space는 인간이 정확히 정의할 수 없는 영역이다.



### Events

사건 이라고도 불리며, 어떤 experiment로부터 나올 수 있는 결과 하나하나를 의미한다.

동전을 10번 던지는 시행 1회에서는,  다음과 같은 event들이 있을 수 있다.

- 앞면이 0번 나오는 event.
- 앞면이 1번 나오는 event.
- ...



### Random Variables

확률 변수라고도 불리며, 어떤 experiment를 수행하면, 확률적으로 어떤 event가 발생할 것이다. 이러한, experiment <-> event 와의 확률 매핑 관계를 random variable이라고 한다.

하버드 어느 통계학 강의에 따르면, random variable을 sample space의 event들을 integer number 값으로 매핑하는 함수라고 정의한다.

확률 변수는 변수이고, 그 확률 변수의 sample space내의 어떠한 event가 될 수 있다.
$$
P(X=x_k)​
$$
이때, $x_k$는 어떤 이벤트이고, $X$는 random variable이다.



하나의 experiment는 하나의 sample space를 가지며, 하나의 random variable과 대응된다. 그 sample space안에 여러개의 event가 있을 수 있다.



## Definition of Probability



**확률은 불확실성을 정량화하는 도구이다.**

0과 1사이 값을 가지는 값이다.

어떤 experiment(시행)에 대한 random variable(확률변수) $X$가 있고, $X$는 $x_1,...,k$의 event를 가질 수 있을 때,
$$
0 \leq P(X=x_i) \leq 1
$$
를 $x_i$가 일어날 확률이라고 정의한다.



일단, 기본적인 확률의 정의는 위와 같다.



### Odds

어떤 event a에 대한 odds는 $O(a) = \frac{P(a)}{P(a^C)}$라고 정의한다. 즉, 동전던지기에서 앞면이 나올 확률이 0.3이라면, 앞면이 나올 event에 대한 odds는 $O(X=h)=\frac{P(X=h)}{P(X \not = h)} = \frac{0.3}{0.7} = \frac{3}{7}$이다.



## How to Compute Probability



Frequentist statistics와 bayesian statistics, 두 통계학에서는 확률을 계산하는 방법에 차이가 있다.



### Classical Method

> Equally Likely Probability

Sample space에서 모든 event들은 일어날 확률이 같다고 정의한다.

동전 1번 던지는 시행에서 sample space는 앞면, 뒷면만 있다고 가정한다. 그럼 앞면이 나올 확률은 0.5이고, 뒷면이 나올 확률 역시 0.5이라고 정의한다.



### Probability in Frequentist Statistics

> Relative Rates of Events in Infinite Sequence

Frequentist statistics에서는 어떤 event의 확률을 "수많은 시행 가운데 그 event가 일어난 비율"이라고 정의한다.

즉, 동전 던지기에서 앞면이 나올 확률을 계산하고 싶다면, 일단 동전을 무수히 많이 던져본다. 1000번을 던진 후, 651번의 앞면이 나왔다면, 동전 던지기 시행에서 앞면이 나올 확률은 0.651 로 정의한다.



### Probability in Bayesian Statistics

> Personal Perspective

Frequentist statistics에서 확률의 문제점은 샘플링이 가능해야 확률을 정의할 수 있다는 점이다. 그런데, 실제로는 샘플링 없이 확률을 정의해야 하는 경우가 많다.

예를 들어, 내일 비가 올 확률은 frequentist statistics 에서는 설명이 불가능하다. 샘플링을 하기 위해서는 타임머신이 필요해 보인다.

또 다른 예시로, 주사위가 fair할 확률 $P(fair)$을 보자. frequentist statistics에서는 주사위는 물리적인 물체이기 때문에 공장에서 만들어졌을 때 부터 fair한것이면 fair한것이고 아니면 아닌 것이다. 즉, deterministic한 요소이며, 확률적으로 어쩔때는 fair하고 어쩔때는 unfair하고 그러지 않는다는 것이다. 따라서 $P(fair) = \{0,1\}$이다.

Bayesian statistics에서는 모든 것은 deterministic하지 않다. 즉, 0%, 100%는 존재하지 않으며 불확실성이 항상 존재한다고 가정한다.

Bayesian statistics에서의 확률이란, 개인의 믿음에 관련되어 있다. 예를 들어, 동전 던지기 시행에서 앞면이 나올 확률은 "내가 생각하기에는 0.3일 것이다."이면, 0.3인 것이다. 다만, 이 확률을 정할 때, fair bet을 만족해야 한다.

예를 들어, 만약, 앞면이 나오면 7달러를 얻고, 뒷면이 나오면 3달러를 잃는다고 하자. 그럼 다음을 만족해야 한다.
$$
E[gain] = p*7 + (1-p) * (-3) = 0​
$$
즉, 내가 앞면이 나올 확률에 어느정도 베팅을 할 수 있는지에 대한 자신감을 고려해서 최대한 공평하게 확률을 선정해야 한다. 이때 $p=0.3$이 되야 한다.



### Bayes' Rule 

다음을 bayes' rule이라고 정의한다.
$$
P(A=a_1|B) = \frac{P(B|A=a_1)P(A=a_1)}{\sum\limits_{a \in A} P(B|A=a)P(A=a)}​
$$
