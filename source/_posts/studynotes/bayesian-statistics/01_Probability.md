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



통계학에서는 확률론을 기초로 한다. 확률론은 확률에 대한 이론을 다루며, 확률은 불확실성을 수치화(정량화)함으로써, 불확실성을 수학으로 다룰 수 있도록 도와준다.



# Background

확률론을 하기 앞서서, 필요한 용어들을 먼저 정리할 필요가 있다.

- **Experiments**

  시행이라고도 불리며, 하나의 데이터 샘플을 얻는 행위를 말한다.

- **Sample space**

  Sample space란, 어떤 시행에 의해 나올 수 있는 모든 경우의 **집합**을 의미한다. 예를들어, 동전던지기라는 시행에서는 뒷면 또는 앞면만이 나올 수 있다. 이때, 동전던지기라는 experiment의 sample space는 $\{\text{HEAD}, \text{TAIL}\}$이 된다. 주사위를 던지는 experiment에 대해서의 sample space는 $\{1, 2, 3, 4, 5, 6\}$이 될 것이다.

- **Events**

  사건이라고도 불리며, **Sample space의 부분집합**이다. 예를 들어, 주사위를 던지는 시행에서, sample space는 $\{1, 2, 3, 4, 5, 6\}$이다. 그리고, 다음과 같은 event $A$를 정의할 수 있다.

  - $A$: 짝수가 나오는 경우

  이때, $A$는 $\{2, 4, 6\}$이 된다.

- **Random variables**

  확률변수라고도 불리며, 각 **sample space를 어떤 다른 라벨로 매핑하는 함수**를 의미한다. 예를들어, 주사위를 던지는 experiment가 있다고 가정해보자. 이때, sample space는 $\{1, 2, 3, 4, 5, 6\}$이 된다.

  이때, 우리는 random variable $X$를 다음과 같이 정의할 수 있다.

  - $X = x_1$: 주사위가 짝수인 경우
  - $X=x_2$: 주사위가 홀수인 경우
  - $X=x_3$: 주사위가 4보다 크거나 같은 경우

  Random variable은 여러개의 이벤트중 하나의 이벤트를 취할 수 있으며, 주사위 던지기의 기존 라벨 $\{1, 2, 3, 4, 5, 6\}$을 $\{x_1, x_2, x_3\}$으로 매핑하게 된다. 이때, random variable의 sample space는 $\{x_1, x_2, x_3\}$이 된다.

  각 이벤트 $x_1, x_2, x_3$은 서로 겹치는 부분이 있어도 상관없다(예제에서는 $x_3$과 $x_2$가 모두 4를 가지고 있다.



# Definition of Probability

**확률은 불확실성을 정량화하는 도구이다.**

어떤 experiment(시행)에 대한 random variable(확률변수) X가 있고, X는 $x_1,...,x_k$의 event(사건, 경우)를 취할 수 있을 때, $x_i$가 일어날 확률은 대문자 $P$를 이용하여 $P(X=x_i)$로 정의한다. 또는 $p_X(X=x_i)$로 표기하거나 간단하게 $p(x_i)$로 표기하기도 한다(세 가지 표현법 모두 같은 의미임). 어떤 사건이 일어날 확률은 항상 0보다 크거나 같으며 1보다 작거나 같다.

$$
0 \leq P(X=x_i) \leq 1
$$




## Odds

어떤 event a에 대한 odds는 $O(a) = \frac{P(a)}{P(a^C)}$라고 정의한다. 즉, 동전던지기에서 앞면이 나올 확률이 0.3이라면, 앞면이 나올 event에 대한 odds는 $O(X=h)=\frac{P(X=h)}{P(X \not = h)} = \frac{0.3}{0.7} = \frac{3}{7}$이다.



## How to Define Probability

방금전까지 확률은 불확실성을 정량화해주는 도구라고 정의했다. 그런데, 어떻게 정량화를 해야 할까. 불확실성을 정량화하는 방법은 크게 3가지로 나눌 수 있다.

- Classical method
- Frequentist method
- Bayesian method



### Classical Method

**Equally Likely Probability**

Sample space에서 모든 event들은 일어날 확률이 같다고 정의하는 방법이다.

동전 1번 던지는 시행에서 sample space는 앞면, 뒷면만 있다고 가정한다. 그럼 앞면이 나올 확률은 0.5이고, 뒷면이 나올 확률 역시 0.5이라고 정의한다.

하지만, 이러한 정의에는 문제가 있는데, 내일 날씨가 비가오거나, 맑거나, 우박이 내리는 3가지 경우만 있다고 가정해보자. 이때, classical method에 따르면, 맑을 확률은 0.33, 비가 올 확률도 0.33, 우박이 내릴 확률도 0.33이 된다.

따라서, classical method 방법은 매우 조심스럽게 사용해야 한다.



### Probability in Frequentist Statistics

**Relative Rates of Events in Infinite Sequence**

어떤 event의 확률을 “수많은 시행 가운데 그 event가 일어난 비율”이라고 정의하는 방법으로, Frequentist statistics에서 확률을 정의하는 방법이다.

즉, 동전 던지기에서 앞면이 나올 확률을 계산하고 싶다면, 일단 동전을 무수히 많이 던져본다. 1000번을 던진 후, 651번의 앞면이 나왔다면, 동전 던지기 시행에서 앞면이 나올 확률은 0.651 로 정의한다.

하지만, 이러한 정의에도 문제점이 있다. 어떤 이벤트가 일어날 확률을 계산하기 위해서는 많은 수의 샘플이  필요하고, experiment를 많이 수행해야 한다. 하지만, experiment가 가능한 경우는 실제로 그렇게 많지가 않다.

예를들어, 내일 비가 올 확률을 구하고 싶다고 해 보자. Frequentist statistics에 따르면, 내일 날씨를 여러번 샘플링 해야 한다. 즉, 내일 날씨를 확인하고, 다시 오늘로 돌아온 후 내일이 되면 날씨를 확인하고, 다시 오늘로 돌아와서 내일이 되면 날씨를 확인하고를 반복해야 한다. 하지만, 알다시피, 이건 타임머신이 있어야 가능하다.



### Probability in Bayesian Statistics

**Personal Perspective**

Bayesian statistics에서 확률을 정의하는 방법으로, 그 event가 일어날것 같다는 개인의 견해와, 필요하다면 이전의 데이터를 바탕으로 확률을 정의한다.

예를 들어, 내일 비가 올 확률을 구하기 위해 오늘의 날씨를 보고 내일 비가 올 확률이 0.7정도 되겠구나 하는 개인의 믿음을 확률에 반영하게 된다.

Bayesian statistics에서 확률의 특징은 모든 것은 deterministic하지 않다. 즉, 0%, 100%는 존재하지 않으며 불확실성이 항상 존재한다고 가정한다.

그렇다면, "개인의 견해"를 얼만큼 반영하고 어떻게 설정해야 할까. 이때 사용하는 방법 중 하나가 fair bet이라는 방법이다.

예를 들어, 만약, 앞면이 나오면 7달러를 얻고, 뒷면이 나오면 3달러를 잃는다고 하자. 그럼 다음을 만족해야 한다. 

$$
E [ \text{gain} ]=p*7 + (1−p)*(−3) = 0
$$

즉, 내가 앞면이 나올 확률에 어느정도 베팅을 할 수 있는지에 대한 자신감을 고려해서 최대한 공평하게 확률을 선정해야 한다. 이때 $p = 0.3$이 되야 한다.



## Bayes' Rule

두 random variable $X,Y$가 있을 때, 다음을 Bayes' rule이라고 정의한다.

$$
p(X=x|y) = \frac{p(Y|X=x)p(X=x)}{p(y)}
$$

Bayes rule의 특징이자 장점은, conditioning variable의 위치를 뒤바꿀 수 있다는 점이다. 이 Bayes rule은 Bayesian statistics에서 매우 중요한 역할을 차지한다.