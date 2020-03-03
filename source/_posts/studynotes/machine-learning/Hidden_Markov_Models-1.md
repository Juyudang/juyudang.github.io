---
title: Hidden Markov Models 1
toc: true
date: 2020-03-01 22:28:55
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning
---



# Hidden Markov Models



Udemy 강좌: https://www.udemy.com/course/unsupervised-machine-learning-hidden-markov-models-in-python



## Markov Assumption

Markov property라고도 부르며, time-series 데이터나, 상태 기반 데이터에서, 현재의 상태는 오로지 바로 이전 상태만으로부터 영향을 받는다는 가정이다. 즉 다음과 같다.
$$
P(s_t|s_{t-1}s_{t-2}\cdots s_1) = P(s_t|s_{t-1})
$$
이전 상태들이 주어졌을 때, 현재 상태의 확률 분포는 오로지 바로 앞전 상태만으로부터 영향을 받는다. 즉, $s_{t-1}$이 주어진다면, $s_t$는 $s_{t-2},...,s_1$와 독립이다(Conditional independence).

Markov assumption은 상당히 강력한 가정으로, 많은 분야에 응용되지만(자연어와 같은 time-series, state machine 기반 모델 등), 바로 이전 상태를 제외한 그 이전 상태들을 모두 무시하므로, 성능에 한계가 있다.

보통 markov assumption하면 first-order markov assumption을 의미하며, 이전 몇 개의 데이터로부터 영향을 받게 할 것인가에 따라 second-order, third-order 등이 있다.

Second-order markov assumption은 다음과 같다.
$$
P(s_t|s_{t-1}, \cdots, s_1) = P(s_t|s_{t-1}, s_{t-2})
$$
Third-order markov assumption은 다음과 같다.
$$
P(s_t|s_{t-1},\cdots,s_{1}) = P(s_t|s_{t-1},s_{t-2},s_{t-3})
$$
그런데, 예상하다시피, 마르코프 가정으로 구현한 모델은 이전 모든 상태에 영향을 받게 모델링한 모델보다 성능이 떨어질 가능성이 높다. 그럼에도 불구하고 사용하는 이유는, 우리가 관심있는것은 지금까지 지나온 상태들의 joint distribution인데, 마르코프 가정이 없다면, joint distribution계산 과정이 매우 복잡해진다. 그래서, 쉽게 모델링하기 위해 마르코프 가정을 사용하며, 성능도 쓸만한 편이다.



## Markov Models

마르코프 가정(Markov assumption)을 바탕으로 모델링한 모델을 말한다. 다음과 같이 state machine도 마르코프 모델 중 하나이다.

![image-20191120062127996](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20191120062127996.png)

State machine은 일반적으로 다음 상태의 확률은 오직 현재 상태에 의해 영향을 받고 결정된다. 위와 같은 state machine에서의 transition probabilities는 행렬로 표현이 가능하며 이러한 행렬을 **state transition probability matrix**라고 부른다. 마르코프 모델에서는 현재에 기반한, 다음 상태 또는 다음 무언가의 확률 분포를 matrix로 표현이 가능하며, $M$개의 노드가 있을 때, transition probability matrix는 $M$x$M$행렬로 표현한다.

State transition probability matrix의 한 행 원소의 합은 1이어야 한다. $i$번째 행이 의미하는 것은 $s_i$를 기반으로 다음 state의 확률분포이기 때문이다.



### Starting Position

지금까지 지나온 상태들의 joint distribution을 계산해 보면 다음과 같다.
$$
P(s_t,s_{t-1},\cdots,s_1) = P(s_1) \prod_{i=2}^{t} P(s_i|s_{i-1})
$$
State transition probability matrix를 정의했다면, $P(s_i|s_{i-1})$는 알 수 있다. 그런데, 초기 상태인 $P(s_1)$는 행렬에 없다.

따라서, initial distribution을 정의해 주어야 하며, $1$x$M$ 벡터로 구성된다.



### Training of Markov Models

마르코프 모델의 학습은 MLE(Maximum Likelihood Estimation)으로 이루어진다. 즉, $s_j \rightarrow s_i$로의 확률 분포는 데이터셋에서 $s_j$ 다음으로 $s_i$가 얼만큼의 비율로 등장하느냐에 따라 결정된다.

예를들어, 다음의 문장이 있다.

"I like cats"

"I like dogs"

"I love kangaroos"

그럼 상태의 집합은 $\{\text{I}, \text{like}, \text{cats}, \text{dogs}, \text{love}, \text{kangaroos}\}$이렇게 6개의 원소로 구성되어 있으며, initial distribution은 $P(\text{I})=1$이고, 나머지 단어의 경우, 0이다.

또한, $P(\text{like}|\text{I})=0.66, P(\text{love}|\text{I})=0.33, P(else|\text{I})=0$이다.



### Smoothing

그런데, 확률이 0이라는 것은 매우 위험하다. 반대로, 어떤 것의 확률이 1이라는 것 또한 매우 위험하다. MLE에 의해 트레이닝 데이터에 나오지 않은 것들은 모두 0이 되버리는데, 이는 오버피팅을 야기한다. 따라서 학습 데이터에 모든 경우의 수가 다 들어있기를 바래야 하는데, 이는 비현실적이다. 따라서 어떤 것에 1 또는 0의 확률을 할당하는 것을 피해야 하는데, 방법으로는 **smoothing**이라는 것이 있다.

smoothing이란, 0확률을 막아주는 기법을 의미하는데, 다음의 경우가 있다.

- No smoothing

  기본적인, smoothing을 적용하지 않은 경우.
  $$
  P(s_i|s_j) = \frac{\text{count}(s_j \rightarrow s_i)}{\text{count}(s_j \rightarrow *)}
  $$

- Add-one smoothing

  분자에 +1, 분모에 +$M$을 해 준다.
  $$
  P(s_i|s_j) = \frac{\text{count}(s_j \rightarrow s_i) + 1}{\text{count}(s_j \rightarrow *) + M}
  $$
  이때, $M$은 상태의 개수(자연어의 경우엔, 단어 개수)이다. 이러면, 모든 확률은 1 또는 0이 되지 않으며, $\sum_i P(s_i|s_j)=1$이 유지된다.

- Add-epsilon smoothing

  분자에 +1이 아니라, +$\epsilon$을 해 준다. 분모에는 +$\epsilon M$을 해 준다.
  $$
  P(s_i|s_j) = \frac{\text{count}(s_j \rightarrow s_i) + \epsilon}{\text{count}(s_j \rightarrow *) + \epsilon M}
  $$
  이때, $\epsilon$은 학습 파라미터로써, 추론해도 되고 hyper parameter로 해도 된다. Add-one 스무딩이 때로는 너무 강하거나 너무 약할때가 있다. 따라서, 스무딩의 강도를 조정하겠다는 이야기가 된다.



## Markov Chains

마르코프 모델이면서, 확률 과정(stochastic process)을 모델링한 것을 의미한다. 보통 통계에서샘플링이라 함은 샘플 하나를 얻는 과정을 말하지만, stochastic(random) process에서의 샘플링은 sequence of random variables을 얻는 과정이고, 하나의 샘플이 time series이다. 마르코프 체인 역시 stochastic process이며, 하나의 샘플은 time-series이다.

State transition probability distribution matrix를 $A$라고 하고, initial distribution을 $\pi$라고 했을 때, $t$번째 상태에서의 marginal distribution은 다음과 같다.
$$
P(s_t) = \pi A^{t}
$$
이때, $A$의 $i$번째 row는 $i$번 상태에서 다른 상태로 갈 확률분포이며, $A$는 $M$x$M$ 행렬이고, $\pi$는 1x$M$벡터이다. 따라서, 위 식은 1x$M$벡터가 나온다.

Marginal distribution에 대해 잠깐 설명해보면, 예를들어, 첫번째 상태 $s_1$의 확률분포는 다음과 같다.
$$
P(s_1) = \sum_{s_0} P(s_1,s_0) = \sum_j \pi_j A_{j,i} = \pi A
$$




### Stationary Distribution

그런데, $A$를 반복해서 곱하다 보면(확률 과정을 반복), 어느 순간 marginal distribution의 변화가 다음과 같은 상태가 된다.
$$
P(s_t) = \pi A^t = P(s_{t-1}) = \pi A^{t-1}
$$
이때, $p(s)=p(s)A$를 만족한다. 이때, $p(s)$를 **stationary distribution**이라고 부른다. 이 stationary distribution $p(s)$을 보면, 행렬 $A$의 전치행렬인 $A^T$의 eigenvector($p(s)$는 벡터이다)와 같은 성질이다는 것을 알 수 있다. 다만, 그에 상응하는 eigen value는 1이다.



### Limiting Distribution

그래서, 어떤 stochastic process의 최종 distribution은 무엇일까. 이 최종 distribution은 **limiting distribution** 또는 **equilibrium distribution**이라고 부른다. 즉, 다음과 같다.
$$
p(s_\infty) = \pi A^\infty
$$
그런데, 이건 stationary distribution과 같은가?

일단, **limiting distribution은 stationary distribution이다. 하지만, 모든 stationary distribution이 다 limiting distribution이 되는 건 아니다.** Eivenvector는 최대 $A$의 차원만큼 개수가 존재하며, 그중에서 eigen value가 1인 eigen vector는 여러개 일 수 있다. 이들 중 어느놈이 limiting distribution일까..

일단, limiting distribution이 구해지면, 그 stochastic process를 통해 앞으로 나올 time series를 샘플링할 수 있다(MCMC의 원리?).



### Perron-Frobenius Theorem

선형 대수학에서의 어떤 이론인데, stochastic process에 맞아떨어지는 이론이다.

어떤 행렬 $A = (a_{i,j})$에 대해, $A$는 $n$-by-$n$ matrix이고, 모든 원소가 양수이면, $A$의 가장 큰 양수 eigenvalue $r$이 존재하고 그와 상응하는 eigenvector의 모든 원소는 양수이다. 그리고, 모든 원소가 양수인 eigenvector는 이 eigenvector가 유일하며, 다른 eigenvector는 반드시 음수가 하나이상 포함되어 있다.

Stochastic process에서 다음 두 가지 조건을 만족시킨다면, 그 Markov chain은 반드시 유일한 stationary distribution을 가지며, 따라서, 해당 stationary distribution은 limiting distribution이라고 확신할 수 있다. Transition matrix $A$에 대해,

- $\sum_j a_{i,j} = 1$, 즉, 한 row의 모든 원소 합이 1이다. 하나의 row는 probability distribution이다.
- $a_{i,j} \not = 0$, 어떠한 원소도 0이 아니다.

여기서, transition matrix $A$의 eigenvector는 distribution으로써의 역할을 해야 하므로 모두 양수여야 하는데, 그런 조건을 만족하는 eigenvector는 오직 하나밖에 없으므로, 이놈이 limiting distribution이라고 확신할 수 있다.



## Application of Markov Models

### Language Models‌

Second-order language model을 예로 들자. 먼저, 문장의 첫 두 단어에 대한 initial distribution을 만들고 앞 두 단어가 주어졌을 때, 현재 단어에 대한 transition matrix를 만든다.‌

학습은 실제 문장들로 학습하며, 문장에서 앞 두 단어가 주어졌을 때, 현재 자리에 오는 단어의 비율을 transition matrix로 한다. 만약, 현재 단어가 끝 단어라면, 이 단어가 끝 단어일 확률 계산에 추가해준다.‌

앞 $$k$$개의 단어를 바탕으로 현재 단어를 추정하는 Markov model이다.



### Google's PageRank Algorithms‌

Google의 페이지랭크 알고리즘은 각 페이지를 방문할 확률인 stationary distribution(정확히는 limiting distribution)이 높은 순서대로 랭크를 매기는 것을 말한다. 한 페이지에서 다른 페이지로 가는 링크가 있을 것이고, $A$페이지에서 $M$개의 링크가 있고, $B$페이지로 가는 링크가 존재한다면, $A→B$ 로의 transition probability는 $\frac{1}{M}$이 된다. 이렇게 transition matrix를 정의하고, matrix에서 0인 원소들을 smoothing을 이용해서 없앤 후, stationary distribution을 계산한다.

현재 페이지에서 다음 페이지로 갈 확률이 존재하는 Markov model이다.



## Hidden Markov Models

마르코프 모델에서 hidden state상태를 추가한 형태. hidden state가 markov chain을 이루고 hidden unit에서 visible variable이 컨디셔닝 되어 나온다. 다음 그림은 markov chain과 hidden markov chain을 표현한 것인데, 노란색이 hidden unit들, 파란색이 visible unit을 표현한 것이다.

![image-20191201145914442](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20191201145914442.png)

Hidden markov model에서는 observable state $o_t$가 이전 observable state $o_{t-1}$에 영향을 받지 않는다. 대신 같은 시간의 hidden state인 $h_t$에 의해서만 영향을 받는다는 가정을 한다.

Markov model은 initial distributoin $$\pi$$와 transition probability matrix $A$가 존재하지만, hidden markov model에서는 initial distribution $$\pi$$와 hidden state transition matrix $A$, hidden state로부터 visible state로의 변환을 의미하는 transition matrix $B$가 존재한다.



### Application of HMM

다음과 같은 application이 존재할 수 있다.

- Parts of Speech (POS) Tagging Systems
- Stock Price Models



#### Parts of Speech (POS) Tagging Systems

각 단어를 visible unit으로, 명사인지 동사인지, 형용사인지 등을 hidden state로 삼아서 HMM을 모델링하는 것을 말한다.

크게, 음성 시그널을 최외곽 visible variable, 단어를 hidden state로 삼아서 markov chain을 구성하는데, 이 애들이 다시 다른 HMM에 들어가는 방식이라고 생각하면 된다.



#### Stock Price Models

HMM이 hidden time series($$z$$들)를 캐치할 수 있다는 것에 주목해서 stock price의 hidden factor를 HMM으로 캐치하게 한 모델을 말한다.

![image-20191201151728545](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20191201151728545.png)

이때, visible variable은 deterministic한 것이 아니라 generative하게 distribution으로 모델링할 수도 있다(위 그림처럼).

