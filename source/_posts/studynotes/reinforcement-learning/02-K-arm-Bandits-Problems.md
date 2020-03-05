---
title: 02. K-arm Bandits Problems
toc: true
date: 2020-03-03 10:00:01
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# K-arm Bandits Problems



참고: Coursera Reinforcement Learning (Alberta Univ.)

Reinforcement learning에서 가장 기본적인 문제로, K-arm bandit problem을 들 수 있다. K개의 arm이 있고, agent는 그 arm 중 하나를 골라야 하며, 하나를 골랐다면, environment는 그 arm의 reward를 반환한다.



### Multi-armed Bandits Problem

슬롯 머신문제, 의사의 치료법 선택 문제 등을 통칭해서 multi-armed bandits 종류의 문제라고 부른다.

- 슬롯 머신 문제

  내가 어떤 슬롯 머신을 선택하면 분명히 해당 슬롯 머신의 stationary distribution에 의해 어떤 결과를 얻을 것이고 그 결과를 바탕으로 reward를 얻게 될 것이다. 하지만, stationary distribution은 인공지능이 알 수 없다. 하지만, 어떤 슬롯 머신을 많이 돌려봤다면 해당 머신에 대해서는 stationary distribution을 어느정도 추정할 수 있다. 추정한 놈들중 높은 reward를 주는 결과를 뱉는 머신을 선택할 수도 있고 새로운 머신을 돌려서 그 머신의 stationary distribution을 추정해 볼 수 있다. 그 머신이 지금까지 알고 있는 머신보다 더 좋은 결과를 줄 수도 있기 때문에 새로운 머신도 조사해봐야 한다.

- 의사의 치료법 선택 문제(이름은 신경쓰지 말자)

  의사는 심각한 환자에게 신생 치료법을 제안할 수도 있고 최선이라고 알려진 방법을 선택할 수도 있다. 신생 치료법은 최선이라고 알려진 방법보다 나을수도, 쪽박일 수도 있다.

이런 종류의 문제를 K-armed bandits 문제라고 부른다.



### K-arm Bandit vs Reinforcement Learning

K-arm bandit problem은 reinforcement learning에서 다음과 같은 제약조건을 걸어버린, 쉬운 문제에 속한다.

- 오직 하나의 state만 존재한다. 즉, 어떤 arm을 선택하는 액션을 취한다고 해서 state가 변하지는 않는다.
- 취할 수 있는 action 목록은 항상 동일하다. 시간이 지난다고 해서 가능한 액션이 늘어난다거나 줄어들지 않는다.
- 항상 optimal action이 일정하다. 시간에 따라 optimal action이 변하지 않는다.
- 유한한 환경이다. 즉, 선택 가능한 액션 개수, state개수 등이 finite하다.

일반적으로 reinforcement learning은 위 제약조건들이 없다.



### Terminology

$A_t$: 어떤 시간 $t$에서 취한 액션

$R_t$: 어떤 시간 $t$에서 취한 액션으로 얻은 reward

$q_* (a)$: 어떤 임의 액션 $a$를 취해서 얻는 reward의 기댓값 즉,
$$
q_* (a) = \mathbb{E}(R_t|A_t=a)
$$
(당연히 $q_* (a)$는 인공지능이 알 수 없는 값이다. 이걸 알면 기댓값이 높은 길만 선택하면 된다.)

$Q_* (a)$: 인공지능이 exploitation, exploration을 바탕으로 추정해낸 분포로 계산한, 임의 액션 $a$를 취했을 때의 reward 기댓값. 인공지능은 적절한 exploration을 통해 $Q_* (a)$를 업데이트해서 $q_* (a)$와 가깝게 추정해야 한다.



## Exploitation vs Exploration

Value function을 말하기 앞서서, exploitation과 exploration은 서로 균형을 이뤄야 한다. 이 둘의 균형을 맞춰야 적절히 최대 reward를 찾아가는 결정을 하면서 새로운 길을 개척할 수 있다.

이것을 달성하는 방법으로 다음과 같은 것들이 있다.

- $\epsilon$-greedy methods (Epsilon-greedy)

- Optimal Initial Values
- UCB(Upper Confidence Bound) Methods
- Bayesian/Thompson Sampling



### Trade-off Between Exploration and Exploitation

일단, agent는 어떤 결정을 할때, exploration과 exploitation을 동시에 수행할 수 없다. 반드시 exploration을 할지, exploitation을 할지 선택하고 액션을 취해야 하는데, exploration을 하게 되면 당장 최선의 결과를 얻을 수 없으며, exploitation을 하게 되면 당장 최선의 결과를 얻을 수 있으나, 더 최상이 되는 다른 것을 찾아나설 수 없다.



### $\epsilon$-greedy Methods

매번 action을 선택할 때, $\epsilon$의 확률로 exploration을 하고, $1-\epsilon$의 확률로 exploitation을 하게 구현한 방법이다. 이 방식의 단점은 충분히 수렴한 뒤에도 $\epsilon$의 확률로 exploration을 하게 된다는 것이다(하지만 non-stationary distribution인 환경에서는 이게 더 도움이 된다).

$\epsilon$이 크면 빠르게 optimal한 action 기댓값을 찾도록 exploration을 할 수 있지만, 지나치게 exploration을 많이 하고 수렴 이후에는 greedy action을 해야 하는데 exploration을 하고 있는 경우를 볼 수 있다.

그렇다고 $\epsilon$이 작으면 global optimal을 찾는 속도가 너무 느리다.



### Optimistic Initial Values

$\epsilon$-greedy 방법에서, 각 액션의 초기 기댓값을 어떻게 정하느냐에 따라 알고리즘의 성능이 달라지기도 한다. 가장 기본적인 방법은 초기 기댓값을 0으로 두는 것으로, 이렇게 되면, $\epsilon$값을 0보다 크게 둬서 어느정도 exploration을 유도해야 한다.

하지만, 최대 reward를 알고, initial value값을 최대 reward보다 높은 값을 설정해 두면, 오직 greedy하게 선택하게 하는 것으로도 초기 iteration에서 적절히 exploration이 가능하다. 하지만, 어느정도 지나게 되면 exploration을 하지 않는다.

![image-20191230212326720](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20191230212326720.png)

초기 reward 기댓값을 높게 설정하면, 아무리 좋은 reward를 얻는 액션이라 하더라도, 기댓값이 감소하기에, 다른 액션의 기댓값이 높아지게 된다. 따라서 자연스럽게 exploration을 하게 된다. 하지만, 초창기 iteration이 거의 exploration이 차지하기 때문에 수렴이 다소 느릴 수 있으나, 그 다음부터는 매우 빠르게 optimal에 수렴한다.

optimal에 수렴한 이후부터는 exploitation만 수행하게 되기 때문에(이미 최적 expectation을 계산해서 낮은 기대치를 갖는 액션은 취하지 않는다.) 만약, optimal reward expectation이 변하는 환경, 즉, stationary distirbution이 변하는 환경에선 이 방법은 적합하지 않다. 시간이 지남에 따라 최적의 액션이 바뀌게 되면 이 방법이 소용이 없어진다.

다음을 만족하는 action을 선택한다.
$$
a^* = \underset{a}{\text{argmax}} ~ q^* (a)
$$
Initial value estimation이 높기 때문에 오직 greedy하게 액션을 선택한다.



### UCB (Upper Confidence Bound)

Confidence interval을 이용해서 액션을 선택하는 방법으로, 각 action의 value 기댓값을 추정한 후, 그 기댓값의 confidence interval를 계산한다. 그리고, 특정한 p-value에 대해 confidence interval의 upper bound를 구한 후, 가장 높은 upper bound를 가지는 action을 선택하는 방식이다. 

이 방법의 장점은, 굳이 $\epsilon$의 확률을 정해놓고 exploration을 하게 하는 것이 아니라, exploration이 얼마 이루어지지 않아 불확실한 confidence boundary를 가지는 것을 알아서 선택하게 하고, 많이 exploration되어 확실하지만, 높은 기댓값으로 확실한 값을 선택하게 함으로써, exploration과 exploitation을 자동으로 조절해서 선택하게 한다.

시간이 지날수록 수렴하게 되면 exploration은 자동으로 줄어들게 된다.

![image-20200105142655161](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200105142655161.png)

![image-20200105142946483](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200105142946483.png)

즉 ,다음을 만족하는 action을 선택한다.
$$
a^* = \underset{a}{\text{argmax} } [q^* (a) + c\sqrt{\frac{\text{ln} ~ N} {N_a} } ]
$$
이때, $\underset{a}{\text{argmax}} ~ q^* (a)$는 greedy 한 선택을 위한 term, 즉, exploitation을 위한 term이고, $a^* = \underset{a}{ \text{argmax} } ~ c \sqrt{ \frac{ \text{ln} N } {N_a} }$은 exploration을 위한 term이다. 즉, confidence interval인데, 다음 식을 이용해서 유도할 수 있다고 한다.
$$
P(|\bar{X} - \mu| \geq \epsilon) \leq 2e^{-2 \epsilon^2 N}
$$


### Bayesian/Thompson Sampling

UCB와 매우 유사하지만, value 기댓값의 confidence interval의 upper bound가 가장 큰 액션을 선택하는 것이 아니라, value의 posterior를 구하고 거기서 샘플링 한 후, 가장 큰 샘플을 가지는 액션을 선택하게 된다.
$$
a^* = \underset{a}{\text{argmax}} ~ [\bar{x} \sim \text{Posterior}(q^*(a))]
$$
이것은 UCB와 마찬가지로 confidence/credible interval과 관련되어 있는데, value 샘플링을 할 때, 가장 높은 value가 나온 경우는 두 가지로 생각할 수 있다.

- 그 action에 대한 value estimation이 매우 불확실한 경우. 즉, credible interval이 매우 넓은 경우.
- 그 action의 value estimation이 그냥 높은 경우. 즉, optimal인 경우.

하는 방법은 다음과 같다.

1. 한 액션의 value를 파라미터로 생각한다.
2. Value의  likelihood를 모델링하고, conjugate prior를 설정한다.
3. Value의 posterior를 계산한다.
4. Posterior를 계산했으면, posterior를 이용해서 value하나를 샘플링한다. Conjugate prior가 아니라면? MCMC를 써서 수렴시킨 후에 나온 샘플 하나를 가저오면 되나?
5. 이것을 각 액션에 대해서 반복하고 가장 높은 value 샘플을 가지는 액션을 취한다.



## Action-value Methods

어떤 액션을 골라야 할 때, 지금 현재 가지고 있는 지식만으로 각 액션을 취했을 때의 얻어지는 기댓값을 각각 계산하고, 가장 높은 기댓값을 가지는 액션을 취하는 방식이다. 즉, value function을 "가장 큰 액션의 기댓값을 가지는 액션"이라고 정의하는 것이다.

액션에 따른 reward의 기댓값 $Q_* (a)$을 계산하는 방법은, 다음과 같이 할 수도 있고, 다른 방법을 사용할 수도 있다.
$$
Q_* (a) \approx \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}_{A_i=a} } {\sum_{i=1}^{t-1} \mathbb{I}_{A_i=a} }
$$
즉, 이때까지 $a$라는 액션을 취했을 때, 얻었던 reward들의 평균값으로 $a$의 reward 기댓값이라고 삼는 것이다.

그리고, 다음을 만족하는 액션 $A_t$를 선택한다.
$$
A_t = \text{argmax}_a ~ Q_t(a)
$$
Action-value methods는 greedy한 방식으로, $\epsilon$-greedy와 함께 사용해서 exploration과의 균형을 맞추려고 시도해 볼 수 있다.



## Associative Search (Contextual Bandits)

K-arm bandit problem은 매우 간단한 reinforcement learning 예제이다. 보통 흔히 이야기하는 reinforcement learning에 다음과 같은 제약조건을 걸면 K-arm bandit problem이 된다.

- 단 한 가지의 situation만 존재한다.
- 그에 따라, 액션이 situation에 영향을 미치지 않는다.(액션을 취한다고 해서 다음 time의  situation이 다른 situation으로 바뀌지는 않는다.)

**상황(situation 또는 state)**이란, agent가 상호작용하는 environment의 한 객체라고 생각해도 되며, 상황이 바뀌면 reward를 샘플링하는 value function도 바뀐다. 따라서, 이 문제는 non-stationary problem이며, 각 상황들에 할당되어 있는 value function 역시 non-stationary일 수도 있다.

Associate search를 통한 reinforcement learning은 contextual bandit problem이라고도 부른다.



반대로 이야기하면, full reinforcement learning은 다음과 같은 점때문에 K-arm bandit problem이랑 다르다.

- Environment안에 여러개의 situation이 state로 존재한다.
- 각 state에는 서로 다른 value function이 있다. 즉, 어떤 state에서는 액션 $A_i$을 취하는게 optimal이지만, 다른 state에서는 액션 $A_j$를 취하는게 optimal이 되기도 한다. state마다 취할 수 있는 액션 집합이 다를 수도 있다.
- 하나의 state입장에서, value function은 non-stationary distribution이 될 수 있다.
- Agent가 취하는 액션은 본인이 받는 reward에도 영향을 미치지만, 다음 time에서의 state에도 영향을 줄 수 있다. 즉, 액션이 state transition을 야기하기도 한다.