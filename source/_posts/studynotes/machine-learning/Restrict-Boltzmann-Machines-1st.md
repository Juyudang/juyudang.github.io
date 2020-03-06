---
title: Restrict Boltzmann Machines 1
toc: true
date: 2020-03-03 22:28:52
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# Restrict Boltzmann Machines



Restrict Boltzmann machines은 Boltzmann machine에다가 일종의 제약조건을 추가한 형태의 neural network를 말한다.

보통 Boltzmann machine이라고 하면 각 뉴런이 bernoulli random variable인 경우가 많다.



## From Boltzmann Machine to RBM

### Statistical Mechanics

입자가 매우 많은 환경에서 입자의 운동을 deterministic하게 정의할 수 없을떄, 통계적으로 물리적 현상을 해석하는 학문을 의미한다.

Boltzmann machine은 statistical mechanics의 영향을 많이 받은 neural network이다. Statistical mechanics에서는 **평형 상태(Equilibrium)**를 중요하게 여기는데, 자연계의 모든 것은 평형 상태로 가기를 원한다. 평형 상태에서는 에너지가 한곳에 치우치지 않고 고루 분포되어 있고, 시스템 전체의 에너지를 낮출 수 있으면 낮은 에너지를 갖게 된다.

Statistical mechanics에서는 어떤 상태 $s_i$가 될 확률은, 그 상태가 가지는 에너지 $\epsilon_i$와 그 상태에서의 온도 $T_i$에 의해 결정된다고 한다. $k$는 Boltzmann constant이다.
$$
p(s_i) = \frac{exp\{\epsilon_i/(kT_i)\}}{\sum_j exp\{\epsilon_j/(kT_j)\}}
$$
위 수식을 **Boltzmann Distribution** 또는 **Maxwell-Boltzmann Distribution**이라고 부른다. 통계 역학에서 물리적 특성에 따른 각 상태의 확률 분포를 정의한다. 분자에는 하나의 상태에 대한 식이 들어가며, 분모에는 모든 상태의 식을 합한 normalization constant가 있다.

즉, 상태 $s_i$의 에너지가 높으면 $p(s_i)$는 작아진다. 즉, 자연계가 에너지가 높은 상태로 전이될 확룰을 작다. 반대로, 에너지가 낮은 상태로 전이될 확률은 높다. 이것은 일상 생활에서도 찾아볼 수 있는데, 대기중의 공기들이 갑자기 좁은 공간으로 모이는 상태가 될 확률은 매우 작다. 인간이 강제적으로 박스 내에 많은 공기를 가두는 것이 아닌 이상 말이다. 그리고, 만약, 박스 내에 많은 공기를 가두었다고 가정하자. 이 박스 내의 에너지는 높으며 불안정하다. 그리고 박스를 없애버리면 공기는 자연스럽게 퍼져나가고, 공기 밀도는 작아진다. 왜냐하면, 펴저나간 상태가 안정적이고 확률이 높기 때문이다. 그리고 그 상태는 작은 에너지를 가지는 상태이다.



### Neural Networks and Neuroscience

Neural nework는 뇌를 모방한 확률 모델이다. 우리 뇌는 외부 신호를 받는 뉴런과 뇌 내부 속의 뉴런까지 모두 연결되어 있다. 따라서, Neuroscientist들은 가장 완벽한 형태의 neural network로 Boltzmann machine을 정의했다.

![image-20200114145755116](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200114145755116.png)

위 그림은 Boltzmann machine을 보여주고 있으며, 모든 뉴런이 서로 연결되어 있는 형태를 가진다. 그리고, Boltzmann machine의 뉴련은 외부와 맞닿아 있는 visible unit들과 내부에 숨겨저 있는 hidden unit들로 나뉜다. Visible unit들이 외부로부터 신호를 받는 input layer의 역할을 하며, hidden unit이 latent variable을 발견하고 pattern recognition 역할을 하는 hidden layer로 작동한다.



### Boltzmann Machines

Boltzmann machine의 특징은 모든 유닛이 모두와 연결된 네트워크이며, 각 edge는 양방향이다. 즉, visible unit이 hidden unit으로 정보를 propagation하기도 하지만, hidden unit이 visible unit에게 propagation하기도 한다. 이건 visible unit들 사이 관계, hidden unit들 사이 관계에서도 마찬가지이다.



### Systems in BM

Boltzmann machine은 **하나의 시스템**이며, **한 순간의 상태**를 저장한다. 각 뉴런이 bernoulli variable이라고 하면, 각 variable의 값에 따라 전체 시스템의 상태가 정의된다. 그리고 visible unit개수가 $D$개, hidden unit 개수가 $M$개라고 하면, 이 Boltzmann machine에 의해 표현 가능한 시스템의 상태 개수는 $2^{D+M}$이 될 것이다.

Boltzmann machine의 propagation 목적은 가장 낮은 에너지 값을 가지는 상태를 찾는 것이며, 낮은 에너지 값으로 수렴할 때 까지 edge를 오가며 unit들이 신호를 주고받는다.

주의할 것은, Boltzmann machine에서, 낮은 에너지 상태를 찾는다는 것은 weight가 변한다는 것이 아니다. 뉴런의 값이 변하는 것이다. 현재 가지고 있는 weight를 가지고, 유닛들이 서로 신호를 보내면서 유닛의 값을 고치면서 찾을 수 있는 가장 낮은 에너지 상태로 수렴하게 된다.



### Propagation in BM

BM에서의 propagation은 생략하고 뒤에서 RBM의 propagation을 말하고자 한다.



### Convergence of BM

Boltzmann machine에서는 각 edge에 weight를 두고 각 뉴런은 bias를 가지고 있다. 그리고, Boltzmann machine이 현재 상태에서의 에너지는 다음과 같이 계산한다.
$$
E(s_i) = -[\sum_i \sum_j W_{i,j} z_i z_j + \sum_k b_k z_k]
$$
이때, $z_i$는 Boltzmann machine의 $i$번 뉴런이다. Edge는 양 끝에 연결된 뉴런이 모두 1일 때, 활성화 되는 형태이다.

위 energy function을 Boltzmann distribution에 넣으면, 다음과 같을 것이다. (BM에선 온도는 사용하지 않는다. 다르게 말하면, 온도는 고정된 상수로 간주한다.)
$$
p(s_i) = \frac{exp\{-E(s_i)\}}{\sum_j exp\{-E(s_j)\}}
$$




### Intractability of BM

하지만, 여기서 가장 큰 단점이자, BM이 현재로서는 사용할 수 없는 인공신경망이라는 것을 드러내 주는 것이 있다. 바로, **Intractability(계산불가능)** 특성이다. Boltzmann distribution을 보면, 분모를 계산하기 위해서는 모든 상태의 exponential을 계산해야 한다. 하지만, visible unit개수와 hidden unit개수가 증가하면, 가능한 상태의 개수는 exponential하게 증거하게 된다. 이 증가량으로 인해 실생활에 응용할 수 있는 neuron 개수만큼 unit을 생성하게 되면, Boltzmann machine이 가지는 상태 수는 엄청나게 증가한다.

따라서, 현실적으로 계산 가능한 모델을 구현하기 위해 BM에 제약(restrict)을 걸어서 사용하게 되는데, 이를 **Restrict Boltzmann machine(RBM)**이라고 한다.



## Restrict Boltzmann Machine

BM에 다음의 제약을 준 신경망이다.

- Visible unit끼리는 직접 연결되지 않는다.
- Hidden unit끼리는 직접 연결되지 않는다.

![image-20200114172903349](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200114172903349.png)

그 외에, 기본적인 이론은 BM과 모두 같다. 다만, visible unit과 hidden unit을 구분해서 쓴 energy function은 다음과 같다.
$$
E(v, h) = -[\sum_i \sum_j w_{i,j} v_i h_j + \sum_i b_i v_i + \sum_j c_i h_i]
$$
$c$는 hidden unit에 있는 bias이다.



### Propagation in RBM

RBM에서 propagation은 다음과 같이 두 가지가 있을 수 있겠다.

- $p(h|v)$
- $p(v|h)$

즉, visible unit이 주어진 후, hidden unit으로의 propagation과 hidden unit이 주어진 후 visible unit으로의 propagation이다. 일단 결론부터 말하면, propagation은 다음의 식으로 이루어진다.
$$
p(h=1|v) = \sigma (W^T v + c)
$$

$$
p(v=1|h) = \sigma (W h + b)
$$

놀랍게도 하나의 unit이 1이 될 확률은 그냥 일반적인 neural network처럼 sigmoid 결과 형태이다. 그리고, 실제로 energy function을 바탕으로 유도한 결과 위 식처럼 나온다.

이제, 위 식을 유도해 보고자 한다.

먼저, 우리가 필요한 건 propagation을 위한, $p(h|v), p(v|h)$이다. 이 두 가지 식을 유도하는 과정은 정확히 일치하므로, $p(h|v)$만 유도하고자 한다.

먼저, $p(h|v)$는 Bayes rule을 통해 다음처럼 적을 수 있겠다.
$$
p(h|v) = \frac{p(v,h)}{p(v)}
$$
Boltzmann distribution의 분모를 $Z$라고 하자. 분모는 normalization constant이다. 이때, Boltzmann machine의 상태는 visible unit과 hidden unit의 조합으로 표현이 가능하므로, $p(s) \leftrightarrow p(v,h)$로 대체하기로 한다.
$$
p(v,h) = \frac{1}{Z} exp\{\sum_i \sum_j W_{i,j} v_i h_j + \sum_i b_i v_i + \sum_j c_j h_j \}
$$
그리고 $p(v) = \sum_h p(v,h)$이므로, (by Marginalization)
$$
p(v) = \sum_j \frac{1}{Z} exp \{ \sum_i \sum_j W_{i,j} v_i h_j + \sum_i b_i v_i + \sum_j c_j h_j \}
$$
그리고, 위 두개의 식을 $p(h|v)$에 넣어보면 $Z$가 약분되어 사라진다.
$$
p(h|v) = \frac{exp\{ \sum_i \sum_j W_{i,j} v_i, h_j + \sum_i b_i v_i + \sum_j c_j h_j \}}{\sum_j exp\{ \sum_i \sum_j W_{i,j} v_i, h_j + \sum_i b_i v_i + \sum_j c_j h_j \}}
$$
그리고, 이 식의 분모는 또 다른 normalization constant이다. 이를 $Z'$라고 하자.
$$
p(h|v) = \frac{1}{Z'} exp\{ \sum_i \sum_j W_{i,j} v_i, h_j + \sum_i b_i v_i + \sum_j c_j h_j \}
$$
그리고 지수 법칙(?)을 이용해서 다음처럼 변형한다. ($e^{a+b} = e^a e^b$)
$$
p(h|v) = \frac{1}{Z'} exp\{ \sum_i b_i v_i \} exp\{ \sum_i \sum_j W_{i,j} v_i, h_j + \sum_j c_j h_j \}
$$
여기서, $p(h|v)$는 $h$의 함수이며, $v$는 이미 주어져 있다($h \text{ given } v$이니까). 따라서, $exp\{\sum_i b_i v_i\}$또한 constant로써, $Z'$와 합체할 수 있다. 이것을 $Z''$라고 하자.
$$
p(h|v) = \frac{1}{Z''} exp\{ \sum_i \sum_j W_{i,j} v_i, h_j + \sum_j c_j h_j \}
$$
그리고, 다시 지수법칙을 이용해서 $j$와 관련된 항을 밖으로 뺀다. (역시 $e^{a+b} = e^a e^b$)
$$
p(h|v) = \frac{1}{Z''} \prod_j exp \{ \sum_i W_{i,j} v_i h_j + c_j h_j \}
$$
그런데, RBM 아키텍처를 다시 한번 소환해보자.

![image-20200114175704355](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200114175704355.png)

보다시피, visible unit이 주어진 상태에서는 hidden unit끼리는 independent이다. 즉, conditional independent이다. 따라서, $p(h|v) = \prod_j p(h_j|v)$가 성립한다. 위 식을 $h_j$에 대한 확률로 변경해보자.
$$
p(h_j|v) = \frac{1}{Z''} exp \{ \sum_i W_{i,j} v_i h_j + c_j h_j \}
$$
$h$끼리는 다 독립이고, 독립일때 joint distribution은 각 확률의 곱이므로 $\prod_j$가 존재했으나, $h_j$하나만 가저올려면 $\prod_j$를 없애면 된다. 그런데, 이때, Boltzmann machine의 각 뉴런은 bernoulli random variable이라고 했으므로, 다음과 같이 $p(h_j|v)$를 나눌 수 있다.
$$
p(h_j|v) = 
\begin{cases}
	p(h_j = 1|v) \\
	p(h_j = 0|v)
\end{cases}
$$
그리고,  다음과 같을 것이다.
$$
\begin{cases}
p(h_j=1|v) = \frac{1}{Z''} exp \{ \sum_i W_{i,j} v_i + c_j \} \\
p(h_j=0|v) = \frac{1}{Z''} exp \{ 0 \} = \frac{1}{Z''}
\end{cases}
$$
그리고 확률의 합은 1이다. 즉, 
$$
p(h_j=1|v) + p(h_j=0|v) = 1
$$
이 식에 대입해서 $Z''$에 대해 정리해보면,
$$
Z'' = exp \{ \sum_i W_{i,j} v_i + c_j \} + 1
$$
그렇다면, $p(h_j=1|v)$는 다음과 같이 정리할 수 있다.
$$
p(h_j = 1|v) = \frac{exp\{ \sum_i W_{i,j} v_i + c_j \}}{exp\{ \sum_i W_{i,j} v_i + c_j \} + 1}
$$
그리고 이것은 sigmoid 함수 형태이며, 최종적으로 다음처럼 정리가 가능하다.
$$
p(h = 1|v) = \sigma (W^T v + c)
$$
지금까지 propagation에 쓰이는 확률을 유도해봤는데, 아직, weight 업데이트에 해당하는 "학습"은 시작도 안 했다. 최적화를 위한 미분 과정도 intractable하기 때문에 좀 복잡하다.

RBM에서 propagation의 문제는 normalization constant의 계산이 불가능하다는 것이었는데, 어떻게어떻게 normalization constant를 trick으로 계산한 후 최종적으로 유도한 것을 보았다.

