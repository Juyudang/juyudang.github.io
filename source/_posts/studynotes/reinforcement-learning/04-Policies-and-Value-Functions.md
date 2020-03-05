---
title: 04. Policies and Value Functions
toc: true
date: 2020-03-03 10:00:03
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning

---



# Policies and Value Functions



참고: Coursera Reinforcement Learning (Alberta Univ.)

RL에서의 학습이란, action을 결정하는 policy와 value function을 추정하는 과정이라고 볼 수 있을 만큼, 두 과정의 RL에의 영향은 절대적이다.



### Policies

우리말로, **정책**이라고 하며, 말 그대로, 주어진 현재 상황에서 agent가 어떤 action을 선택할지 결정해주는 정책을 의미한다.

**Policy는 각 state에서 가능한 action들의 probability distribution이다.**

Value function과 그 의미가 매우 유사해보인다. 하지만, Value function은 reward의 기댓값을 계산해주는 것일 뿐, action을 결정하는 것은 policy에게 달려 있다.

- Deterministic Policies
  $$
  \pi(s)= a
  $$
  위 식처럼, deterministic policy란 state를 입력으로 받아서 action 1개를 반환하는 함수이다. 즉, state를 action으로 매핑하는 함수라고 볼 수 있다.

- Stochastic Policies
  $$
  \pi(a|s) = \text{ {Probability distribution matrix} }
  $$
  Stochastic policy는 한 state를 주게 되면, 그 state에서의 action probability distribution을 반환하는 함수이다. 따라서, 한 state를 주면, 여러 action이 나오게 된다.

  이때, action probability distribution matrix는 다음을 만족해야 한다.
  $$
  \sum_a \pi(a|s) = 1
  $$
  $$
  \pi(a|s) \geq 0
  $$
  
  즉, 행렬의 각 row가 적법한 probability distribution이어야 한다.

action probability distribution은 각 state마다 다르다.



### Valid & Invalid Policies

MDP에서 policies는 반드시 현재 타임에서의 상태에 존재하는 정보를 이용해서 action을 결정해야 한다. 즉, policies의 인자는 반드시 time $t$에서는 $s_t$만이 되어야 하며, 따라서 다음과 같다.
$$
\pi(s_t) \rightarrow a_t
$$
만약, policy가 이전 time의 state나 action을 이용해서 action을 결정한다면, 적법한 policy라고 부르지 않는다.



### Value Functions

다음 두 가지로 나뉜다.

- State Value Function $v_{\pi}(s_t)$

  한 상태 $s_t$에서, 앞으로 어떤 policy $\pi$를 따른다고 했을 때, **앞으로 얻을 수 있는** reward 기댓값을 의미한다.

$$
v_{\pi}(s_t) = \sum_{a_{t} } \pi(a_t|s_t) \sum_{s_{t+1},r_{t+1} } p(s_{t+1},r_{t+1}|s_{t},a_t)[r_{t+1} + \gamma \cdot v_{\pi}(s_{t+1})]
$$

- Action Value Function $q_{\pi}(s_t, a_t)$

  한 상태 $s_t$에서, 어떤 액션 $a_t$를 취하고 난 후, 앞으로 어떤 policy $\pi$를 따른다고 했을 때, **앞으로 얻을 수 있는** reward의 기댓값을 의미한다.

$$
  q_{\pi} (s_t, a_t) = \sum_{s_{t+1}, r_{t+1}} p(s_{t+1}, r_{t+1}|s_t, a_t) [r_{t+1} + \gamma \cdot \sum_{a_{t+1} } \pi(a_{t+1}|s_{t+1})q_{\pi}(s_{t+1}, a_{t+1})]
$$

둘 다 현재 어떤 상황에서 앞으로 얻을 수 있는 reward의 기댓값을 의미한다.



## Bellman Equation

**현재 시간 $t$에서의 value와 다음 시간 $t+1$에서의 value와의 관계식**을 의미한다. State-value Bellman equation과 action-value Bellman equation이 존재하며, reinforcement learning 알고리즘 구현에 있어서 가장 중요한 알고리즘 중 하나이다.



### Bellman Equation vs Value Function

두 용어의 개념에 대한 차이는 거의 없다. Bellman equation도 value function이다. 다만,  $t$에서의 value function을 $t+1$에서의 value function에 대한 식으로 나타냈다 뿐. Recursive하게 표현한 value function을 Bellman equation이라고 부를 뿐이다.



### State-value Bellman Equation

State-value function에 대한 Bellman equation으로, state-value function입장에서, $t$에서의 state value와 $t+1$에서의 state value와의 관계식이다.
$$
v_{\pi} (s_t) = \sum_a \pi(a|s_t) \sum_{s_{t+1},r} p(s_{t+1}, r|s_{t}, a)[r + \gamma \cdot v_{\pi}(s_{t+1})]
$$


### Action-value Bellman Equation

Action-value function에 대한 Bellman equation으로, action-value funciton입장에서, $t$에서의 actionvalue와 $t+1$에서의 action value와의 관계식이다.
$$
q_{\pi} (s_{t}, a_{t}) = \sum_{s_{t+1}, r} p(s_{t+1}, r|s_t, a_t)[r + \gamma \cdot \sum_{a_{t+1} } \pi(a_{t+1}|s_{t+1}) \cdot q_{\pi}(s_{t+1}, a_{t+1})]
$$


### Compute Value using Bellman Equation

Bellman equation의 가장 큰 장점은, value function을 매우 효율적으로 계산할 수 있게 해 준다는 것이다. Value function은 정의에서 보다시피, 미래의 reward의 기댓값이다. 즉, 현재 시간 $t$이후의 모든 시간에서의 reward 기댓값인데, 이 정의로는 value를 계산할 수 없다. Bellman equation은 이 무한 수열 계산문제를 단순한 linear equation으로 바꿔준다.

다음 board를 생각해 보자.

![image-20200119160231033](../../../../../../../GoogleDrive/Notes/note-images/04-Policies-and-Value-Functions/image-20200119160231033.png)

보드에는 $A,B,C,D$라는 4개의 공간이 있으며, 말 하나를 이 공간 내에서 움직이려 한다. 즉, 각 공간이 곧 state이며, 총 4개의 state가 있는 environment이다.

Action은 상,하,좌,우 4개의 움직임이 존재한다. Policy는 총 4개의 움직임에 대해 uniform distribution이다. 말이 $B$$로 들어오거나 B$에 머무는 움직임에 대해서만 reward +5를 부여하고 나머지는 0을 부여한다. Discount factor는 0.7로 하자.

State $A$에서의 value는 무한 수열식이지만, Bellman equation을 이용한다면, 다음 state의 value를 이용해서 계산이 가능하다.
$$
V(A) = \sum_{a} \pi(a|A) \sum_{s',r} p(s',r|a,A)[r + \gamma \cdot V(s')]
$$
그런데, action이 정해지면, state는 확정(deterministic)이므로, 위 Bellman equation을 다음처럼 변경할 수 있다.
$$
V(A) = \sum_a \pi(a|A) [0 + 0.7 \cdot V(s')]
$$
$$

V(A) = \frac{1}{4} \cdot 0.7 \cdot V(C) + \frac{1}{2} \cdot 0.7 \cdot V(A) + \frac{1}{4} \cdot (5 + 0.7 \cdot V(B)))
$$



$V(B), V(C), V(D)$도 유사하게 $V(A), V(B), V(C), V(D)$에 대한 식으로 표현이 가능하며, 일차 연립방정식으롤 표현이 가능하다. 즉, 무한 수열을 푸는 문제가 일차 연립 방정식을 푸는 문제로 바뀐 것이다.

하지만, 현실에서는 approximation방법을 많이 이용한다. State개수가 많아서 그런가?



## Optimality

Reinforcement learning의 목적은 단순히 value function과 policy를 계산하는게 아니라, optimal policy와 optimal value function, 즉, reward를 최대화하는 policy와 value function을 찾는 것이다.



### Optimal Policies

Optimal policy란, 모든 state에서 가장 높은 value를 반환하게 하는 policy를 말한다. 즉, 다음 그림처럼 어떤 여러개의 policies들보다 항상 큰 value를 반환하게 하는 policy는 항상 존재한다.

![image-20200119163026023](../../../../../../../GoogleDrive/Notes/note-images/04-Policies-and-Value-Functions/image-20200119163026023.png)

즉, $\pi_1, \pi_2$보다 항상 크거나 같은 value를 반환하는 policy는 항상 존재한다는 건데, 방법은 간단하다. $\pi_1 \leq \pi_2$인 state에서는 $\pi_2$의 policy를 따르고, $\pi_1 > \pi_2$인 state에서는 policy $\pi_1$을 따르도록 하는 새로운 policy $\pi_3$를 만들면 된다.

이와 같은 방법으로, 언제나 모든 policy보다 크거나 같은 value를 반환하는 policy를 만들 수 있으며, 이런 policy는  unique할 수도 있고 여러개가 될 수도 있다(모든 state에 걸쳐서 똑같은 value를 반환하는 policy가 여러개일 수 있음).

어쨌든, 이런 과정을 거쳐서 가장 높은 value를 모든 state에 걸쳐서 반환하는 policy를 optimal policy라고 부른다. 방금 말했듯이, optimal policy는 반드시 존재하며, 여러개일 수 있다.

또 하나 생각할 점은, 위 그림에서 $\pi_3$은 분명히, $\pi_1, \pi_2$중 하나를 선택한 policy에 불과하므로, $\pi_1, \pi_2$둘 중 value가 높은 policy의 value와 같아야 할 것이데, 어느 지점에서는 $\pi_1, \pi_2$ 모두의 value보다 높다. 이것은, future value까지 반영해서 생기는 현상으로, 미래 state에서도 최선 policy인 $\pi_3$을 따르므로, value는 재료가 된 policy들보다 커질수도 있다.



### Optimal Values

보통 optimal policy는 unknown으로, 바로 계산할 수 없다. 애초에 reinforcement learning의 목적은 optimal policy를 찾는 것이다. Optimal policy를 계산할때는 opimal value function를 이용하게 된다.

Optimal value function이란, 현재 state에서 가능한 모든 액션과 그에 다른 다음 value를 보고, 다음 value가 가장 높은 action을 deterministic하게 선택했을 때의 value function을 의미한다.
$$
v_* (s) = \underset{a}{\text{max} } ~ \sum_{s',r} p(s',r|s,a)[r + \gamma \cdot v_*(s')]
$$
보다시피, action의 분포(policy)가 사라지고, 그냥 다음 state인 $s'$의 value $v_* (s')$가 가장 높은 action을 무조건(deterministically) 취하게 한다. 또한, 이 value $v_*(s')$만으로 $v_* (s)$를 계산하도록 한다.

앞서, value function은 두 가지가 있고, 두 가지 value function 모두 Bellman equation 형태로 바꿀 수 있었다. Optimal value function도 마찬가지이며, optimal한 value function을 Bellman equation형태로 바꾼 것을 Bellman optimality equation이라고 부른다.

- **Bellman optimality equation for state value function**

$$
  v_* (s) = \underset{a}{\text{max} } ~ \sum_{s',r} p(s',r|s,a) [r + \gamma \cdot v_* (s')]
$$
- **Bellman optimality equation for action value function**

$$
  q_* (s,a) = \underset{a}{\text{max} } \sum_{s',r} p(s',r|s,a)[r + \gamma \cdot \underset{a'}{\text{max} } ~ q_* (s',a')]
$$

  