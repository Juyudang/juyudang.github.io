---
title: 03. Markov Decision Process
toc:true
date: 2020-03-03 10:00:02
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning

---



# Markov Decision Process



참고: Coursera Reinforcement Learning (Alberta Univ.)

마르코드 결정 과정.

이름에서 유추할수 있다시피, Markov assumption에 기반한 decision process로, $t+1$에서의 상태 $s_{t+1}$는 오직 현재 $t$에서의 상태인 $s_{t}$에 있을 때, agent의 decision인 $a_{t}$에 의해서만 결정된다는 것이다.

### Markov Decisoin Process vs K-arm Bandit Problems

Markov decision process에서는 K-arm bandit에서 가정했던 여러가지 조건을 해제한 application에 적용이 가능하다.

- K-arm bandit problem에서는 항상 하나의 state만 존재했지만, Markov decision process는 액션을 취함에 따라 state가 변하는 application에도 적용이 가능하다.
- K-arm bandit problem에는 언제나 선택할 수 있는 action 리스트가 고정되어 있었다. 하지만, Markov decision process가 적용되는 application은 그럴 필요가 없다. 각 state에 서로 다른 action list가 있을 수 있다.
- K-arm bandit problem에는 state와 선택 가능한 액션 리스트가 하나임과 동시에 매 time마다 optimal action은 항상 고정되어 있었다. 하지만, 이번에 이야기할 reinforcement learning environment는 state마다 optimal action이 다를 수 있다.



### Finite Markov Decision Process

Agent와 상호작용하는 environment에는 여러 state가 있을 수 있는데, 이 state의 개수가 finite 하며, 각 state에서 존재하는 action 개수도 finite한 경우에 적용되는 Markov decision process를 finite Markov decision process라고 한다. 물론 finite 하지 않는 경우가 매우 많다.



현재 상태를 $s$, 이 상태를 기준으로 내린 decision $a$, 그리고, 그 결정에 의해 변한 상태를 $s'$, 그로인해 받는 reward를 $r$라고 했을 때, Markov decision process의 **state transition probability**는 다음과 같다.
$$
p(s',r|s,a)
$$



### Episodic Tasks vs Continuous Tasks

- **Episodic Tasks**

  바둑, 스타크래프트와 같은 게임처럼, "한번의 판(한 판), stage"이 존재하는 problem을 가리킨다. 따라서, terminal state 라는 것이 존재하며, 하나의 stage를 시작해서 끝난 후 최종 reward까지 받을 때 까지를 하나의 episode라고 부른다. Agent는 여러 episode를 체험해보면서 학습하게 된다. 한 episode에서 이런 선택을 했다면 다음 episode에서 다른 선택을 하면서 다른 결말 및 reward를 획득하면서 학습하게 되는 task이다.

  Episode는 이전의 모든 episode와 독립적이다. 즉, 이전 episode가 어떻게 끝났던 간에, 현재 episode는 이전 episode에 의해 영향을 받지 않는다. 매 게임이 독립이라는 이야기이다.

- **Continuous Tasks**

  일반적인 로봇이 수행하는 작업들이 보통 continuous task이다. 이 경우, terminal state가 없으며, 그냥 life를 살아가면서 마주치는 state에서 action을 수행하면서 학습을 진행하게 된다.



## Goals of MDP

MDP의 목적은 당장 action을 선택했을 때의 reward를 최대화 하는 것이 아닌, 현재 어떤 action을 선택한 후, 미래의 모든 reward들 합의 기댓값을 최대하하도록 하는 action을 선택하는 것이다. 즉, 다음과 같은 action $a_t^*$를 선택한다.
$$
a_t^* = \underset{a}{\text{argmax}} ~ \mathbb{E}[G_t] = \underset{a}{\text{argmax}} ~ \mathbb{E}[R_{t+1} + \cdots + R_T]
$$
이때, $T$는 final state에서의 time 이다. 즉, 한 episode의 끝일때의 time이다.

$G_t$는 random variable인데, $R_t$들이 random variable이고, random variable의 합이기 때문이다. 따라서, random variable $G_t$의 기댓값을 최대화하는 action $a$를 선택하도록 한다.



### Goals of MDP for Continuous Tasks

위에서 소개한 action 선택법은 episodic task에만 적용이 가능하다. 미래의 모든 reward의 합의 기댓값이므로, terminal state가 존재해야 $\mathbb{E}[G_t]$가 finite($\infty$가 아님)하다. continuous task의 경우에는, $R_T$가 없고 무한히 더해지기 때문에, $\mathbb{E}[G_t] \approx \infty$가 된다. 따라서 **discounting**이라는 것을 통해 액션을 선택한다.
$$
a^*_t = \underset{a}{\text{argmax}} ~ G_t = \underset{a}{\text{argmax}} ~ [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots] = \underset{a}{\text{argmax}} ~ [R_{t+1} + \gamma G_{t+1}]
$$
Discounting을 하는 이유는 $G_t$를 finite하게 만들기 위함이며, 다음과 같기 때문에 finite하다. 이때, $0 \leq \gamma < 1$이어야 한다. $R_{max}$를 agent가 한 액션을 취했을때 얻을 수 있는 액션의 최대치라고 하자.
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots \\
G_t \leq R_{max} + \gamma R_{max} + \gamma^2 R_{max} + \cdots \\
G_t \leq R_{max}(1 + \gamma + \gamma^2 + \cdots) \\
G_t \leq R_{max} \cdot \frac{1}{1 - \gamma} ~~~ \text{iff } 0 \leq \gamma < 1
$$
따라서, $0 \leq \gamma < 1$을 만족하면, $G_t$는 $R_{max} \cdot \frac{1}{1 - \gamma}$보다 작다. 그리고, finite하다($\infty$가 아니다).

굳이 episodic task라고 해서 discounting을 사용하지 말라는 법은 없다. discount rate $$\gamma$$를 통해 미래 reward 지향적일지, 즉각적인 reward 지향적일지 정할 수있기 때문에 discounting 방법은 episodic task에서도 많이 이용된다.



## Summary

MDP란, 현재 상태만을 바탕으로 action을 취하고 reward를 얻는 환경에서의 reinforcement learning 방법 또는 decision process중 하나이다. 액션은 다음과 같이 취한다.
$$
a^*(t) = \underset{a(t)}{\text{argmax}} ~ \mathbb{E}[G_t] = \underset{a(t)}{\text{argmax}} ~ \mathbb{E}[R_{t+1} + \gamma \cdot G_{t+1}]
$$
