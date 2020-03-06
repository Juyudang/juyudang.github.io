---
title: Hidden Markov Models 2
toc: true
date: 2020-03-03 22:28:57
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# Hidden Markov Models



Udemy 강좌: https://www.udemy.com/course/unsupervised-machine-learning-hidden-markov-models-in-python



Hidden Markov model(HMM)은 다음과 같이 maximum likelihood estimation을 이용해서 파라미터를 추정하게 된다.
$$
\theta^* = \underset{\theta}{\text{argmax}} ~ p(x|\theta)
$$
다음으로, HMM의 파라미터가 무엇인지 적어본다. $\theta = ?$



### Parameters of HMM

Markov model에서의 parameter는 initial distribution vector $\pi$와 state transition matrix $A$였다. HMM에서는 state-to-observation matrix $B$가 추가된다. 즉, $\pi, A, B$가 학습 parameter가 된다.

- $\pi$

  Initial distribution. row vector이며, hidden state 개수가 $M$개일때, $\pi$는 $(1, M)$ 모양이다. $\pi(i)$하면, $i$번재 state의 initial 확률이다.

- $A$

  Hidden state transition matrix. 간단하게 state transition matrix이라고도 하며, $t$에서의 hidden state가 주어졌을 때, $t+1$에서의 hidden state의 확률분포이다. 즉, $p(s_{t+1}|s_t)$을 표현한다. 따라서, observation의 종류가 $D$개일때, $M \rightarrow D$이므로, $(M, D)$ 모양이다. $A(i, j)$의 원소는 $p(s_{t+1} = j | s_t = i)$를 의미한다.

- $B$

  Observation transition matrix이며, $t$에서의 hidden state가 주어졌을 때, $t$에서의 observation의 확률분포이다. $p(x_t|s_t)$를 표현한다. $B(j, k)$의 원소는 $p(x_t = k|s_t = j)$를 의미한다.

이들을 이용한 연산의 예를 잠깐 몇개 들어보면, (Sequence의 시작 index는 1이다.)

- $\pi B = \sum_i \pi(i) B(i,:) = \sum_{z_1} p(z_1)p(x_1|z_1) =  p(x_1)$이다.
- $\pi A B = \sum_{i,j} \pi(i) B(i,j) A(j,:) = \sum_{z_1, z_2} p(z_1)p(z_2|z_1)p(x_2|z_2) = p(x_2)$이다.



## Algorithms of HMM

HMM에서도 다른 확률 모델과 마찬가지로 forward propagation, backward propagation과정이 존재한다.



### Forward Algorithms

HMM의 forward 알고리즘은 데이터셋의 확률, 즉, likelihood를 계산하는 알고리즘으로 대표된다. Markov model과는 달리, HMM의 likelihood는 곧바로 파라미터로 나타낼 수가 없어서 likelihood를 적절히 변형해야 한다. 그리고 단순히 변형해도, 그 계산의 time complexity가 매우 커서 계산 최적화를 위한 작업을 해 줘야 한다.



### Problem 1: Find Likelihood Distribution

파라미터 $\pi, A, B$를 바탕으로 likelihood를 계산할 수 있어야 한다. Likelihood가 있어야 ML 추정법을 적용할 수 있기 때문. Likelihood는 $p(x|\pi, A,B)$와 같으며, observation $x$의 joint distribution에 해당한다.

파라미터는 수식의 모든 term에 존재하므로, 생략한다. 먼저, likelihood는 다음과 같다. $T$는 전체 sequence 길이이다. 이 likelihood를 파라미터에 대한 식으로 바꿔주어야 한다.
$$
p(x) = p(x_1, x_2, ..., x_T)
$$
이것을 hidden Markov model 구조(확률 그래프 모델이니까)에 따라 factorize하기 위해, hidden variable $z$를 삽입한다 (Marginalize).
$$
p(x) = \sum_{z_1} \sum_{z_2} \cdots \sum_{z_T}p(x_1, x_2, ..., x_T, z_1, z_2, ..., z_T)
$$
이제 factorize한다.
$$
p(x) = \sum_{z_1} \sum_{z_2} \cdots \sum_{z_T} p(z_1) p(x_1|z_1) \prod_{t=2}^{T} p(z_{t}|z_{t-1})p(x_t|z_t)
$$
이제 parameter에 대한 식으로 바꿀 수 있다.
$$
p(x) = \sum_{z_1} \sum_{z_2} \cdots \sum_{z_T} \pi(z_1) B(z_1, x_1) \prod_{t=2}^T A(z_{t-1}, z_t) B(z_t, x_t)
$$
그런데, 이 식의 time complexity를 봐야 한다. 위 식은 결국, 모든 hidden state 조합을 더하는 것이다. Hidden state의 개수는 $M$개이고, 이게 $T$-time 만큼 있으므로, $M^T$개의 hidden state조합이 존재한다. 또한, 하나의 hidden state 조합을 구하기 위해서는 $O(T)$시간이 걸리며, 총 $O(TM^T)$ 시간이 걸리게 된다. 이것은 exponential한 time으로, 매우 비효율적이다.



### Answer to Problem 1: Forward/Backward Algorithm

그런데, 위 $p(x)$에는 겹치는 연산이 상당히 많다. 이것을 factorize해서(인수분해) 좀 더 효율적으로 $p(x)$를 계산할 수 있을 것 같다.

우선, $T=2, M=2$인 경우를 생각해본다.
$$
p(x) = \sum_{z_1} \sum_{z_2} \pi(z_1)B(z_1, x_1)\prod_{t=2}^T A(z_{t-1}, z_t)B(z_t, x_t)
$$

$$
= \pi(1)B(1, x_1)A(1, 1)B(1, x_2) +
$$

$$
\pi(1)B(1, x_1)A(1, 2)B(2, x_2) +
$$

$$
\pi(2)B(2, x_1)A(2, 1)B(1, x_2) +
$$

$$
\pi(2)B(2, x_1)A(2, 2)B(2, x_2) +
$$

그런데, 중복된 연산이 너무 많다. 따라서, factorize를 해 주자.
$$
p(x) =
$$
$$
\pi(1)B(1, x_1)[A(1, 1)B(1, x_2) + A(1, 2)B(2, x_2)]
$$

$$
\pi(2)B(2, x_1)[A(2, 1)B(1, x_2) + A(2, 2)B(2, x_2)]
$$

$T=3, M=2$인 경우도 마찬가지로 할 수 있다. 수식으로 보면 다음처럼 표현할 수 있다.
$$
p(x) = \sum_{z_1} \sum_{z_2} \sum_{z_3} p(z_1) p(x_1|z_1) \prod_{t=2}^3 p(z_t|z_{t-1})p(x_t|z_t)
$$
$$
= \sum_{z_1} \sum_{z_2} \sum_{z_3} p(z_1)p(x_1|z_1)p(z_2|z_1)p(x_2|z_2)p(z_3|z_2)p(x_3|z_3)
$$

$$
= \sum_{z_3} p(x_3|z_3) \sum_{z_2} p(x_2|z_2)p(z_3|z_2) \sum_{z_1} p(z_1)p(x_1|z_1)p(z_2|z_1)
$$

위 식을 다음처럼 변형해본다.
$$
\sum_{z_3} p(x_3|z_3) \sum_{z_2} p(z_3|z_2) [p(x_2|z_2) \sum_{z_1} p(z_2|z_1)[p(x_1|z_1) p(z_1)]]
$$
여기서, $\alpha$라고 하는 놈을 정의한다.
$$
\alpha(3, z_3) = p(x_3|z_3) \sum_{z_2} p(z_3|z_2) \alpha(2, z_2)
$$
$$
\alpha(2, z_2) = p(x_2|z_2) \sum_{z_1} p(z_2|z_1) \alpha(1, z_1)
$$

$$
\alpha(1, z_1) = p(x_1|z_1)p(z_1)
$$

이때, $p(x)$는 다음처럼 된다.
$$
p(x) = \sum_{z_3}\alpha(3, z_3)
$$
즉, 다음처럼 일반화가 가능하다.
$$
p(x) = \sum_{z_T} \alpha(T, z_T)
$$
$$
\alpha(t, z_t) = p(x_t|z_t) \sum_{z_{t-1}} p(z_t|z_{t-1}) \alpha(t-1, z_{t-1})
$$

$$
\alpha(1, z_1) = p(x_1|z_1)p(z_1)
$$

이렇게 되면, likelihood $p(x)$를 계산하는데, $O(MT)$면 끝이 난다.






### Problem 2: Find the Most Likely Sequence of Hidden States

Likelihood를 구했다면, 이번엔 가장 probable한 hidden states의 sequence를 찾을 수 있어야 한다. 즉,
$$
z^* = \underset{z}{ \text{argmax} } ~ p(z|x)
$$

를 만족하는 hidden states $$z$$의 joint distribution을 계산할 수 있어야 한다.

그런데, 이때, 위 식은 다음처럼 정리가 가능하다.
$$
z^* = \underset{z}{ \text{argmax} } ~ p(z|x) = \underset{z}{ \text{argmax} } ~ \frac{p(x,z)}{p(x)} = \underset{z}{ \text{argmax} } ~ p(x, z)
$$
그런데, 여기서, $p(x, z)$는 $p(x)$를 구하는 식에서 marginalization만 빼면 된다. 즉,
$$
p(x, z) = p(z_1)p(x_1|z_1) \prod_{i=2}^T p(z_{t}|z_{t-1})p(x_t|z_t)
$$
이다. 하나의 joint probability를 계산하려면 $O(T)$시간이 걸리는 셈. 그러면, observations들에 맞게 가장 그럴듯한 hidden state들을 찾으려면, hidden state의 모든 조합을 저 식에 넣어보고 가장 큰 확률값을 주는 조합을 고르면 될 것이다. 그러나, 이 방법은 $O(TM^T)$가 걸린다.



### Answer to Problem 2: Viterbi Algorithm

지금, $p(x, z)$가 가장 큰 $z$조합을 구해야 한다. HMM은 Markov model이기 때문에 $t-1$까지 최적의 $z$ sequence를 구해놨다면, $t$에서의 $z_t$는 greedy하게 선택하면 $t$까지의 $z$ sequence는 optimal이다. 즉, $t=1$에서, $p(z_1)p(x_1|z_1)$이 최대가 되는 $z_1$를 구하고, $t=2$에서, $p(z_2|z_1)p(x_2|z_2)$이 최대가 되는 $z_2$를 구하고 이런식으로 앞에서부터 greedy하게 선택해도 된다는 것이다.








### Problem 3: Training HMM

다음을 만족하는 parameter $$\pi, A, B$$를 계산한다.
$$
A^*, B^*, \pi^* = \underset{A,B,\pi}{ \text{argmax} } ~ p(x|A,B,\pi)
$$
