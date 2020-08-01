---
title: 01. Introduction
toc: true
date: 2020-06-15 10:00:00
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Introduction



참고: Coursera Reinforcement Learning (Alberta Univ.)

**강화학습(Reinforcement Learning)**

Reinforcement Learning이란, 환경(environment)가 있고, 그 environment 속에서 agent가 자신이 처한 상태(state)에서 가장 최적의 행동(action)을 취할 수 있도록 하는 알고리즘들을 말한다.

Agent는 어떤 행위를 취하는 객체로, 자신에게 주어진 상태에서 액션을 취할 수 있고, 그 액션에 의해 environment가 영향을 받는다. 그에 따라 agent에게 주어지는 상태가 변화하고, environment로부터 reward 또는 피드백을 받게 된다. Reinforcement learning을 바탕으로 학습된 agent는 주어진 environment와 state를 바탕으로 최적의 액션을 취할 수 있다.

![강화학습 개요](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200615082355478.png)

Reinforcement learning은 현재 주어진 state에서 어떤 action을 취한다면, 어떤 reward를 얻을 것인지를 예측한다. 그리고 예측한 reward가 최대가 되는 action을 선택하게 된다.



## Elements of Reinforcement Learning

Reinforcement learning에는 여러가지 중요한 요소들이 존재한다. 우선, reinforcement learning은 어떤 행위를 하는 agent를 학습하는 것이다. 즉, **agent**가 존재한다. 또한, agent는 **action**을 취할 수 있다. Agent는 아무 조건없이 action을 취할 수는 없다. Agent는 자신에게 주어진 상태(**state or environmnet**)가 있고, 그 environment를 바탕으로 action을 취하고 **reward**(또는 피드백)을 얻는다.

- Agent

  어떤 주어진 environment 속에서 environment와 상호작용하며 action을 취하는 행위 당사자

- Environment

  Agent가 직접 상호작용하는 주위 환경. 보통 agent는 environment의 일부분을 관찰하게 되며, 전체 environment는 모른다고 가정한다.

- State

  Agent가 environment와 상호작용하면서 자신이 처한 상태. Agent는 environment 전체를 볼 수는 없고, environment의 일부분인 state를 볼 수 있다. 그 state를 바탕으로 action을 취할 수 있다.

- Action

  Agent가 자신에게 주어진 state를 바탕으로 취할 수 있는 행동

- Reward

  Agent가 주어진 state에서 action을 취했을 때 environment로부터 얻은 피드백. (+)일수도 있고 (-)일수도 있다.



Agent가 action을 선택하는 방법은, 주어진 state에서 어떤 action을 취했을 때 얻을 수 있는 모든 미래의 reward의 기댓값을 계산하고 그 미래 reward 기댓값이 최대가 되는 action을 선택하는 것이다. 이때, agent가 미래에서 얻을 수 있는 모든 reward들의 기댓값을 **value**라고 정의한다. Agent는 계산한 value를 바탕으로 어떤 action을 취할지에 대한 **policy**를 가진다. Agent는 value와 policy를 바탕으로 action을 취할 수 있다.



### Policy

어떤 주어진 환경/상황에서 어떤 동작을 취해야 하는지에 대한 규칙 또는 정책, 또는 매핑 함수이다. 강화학습에서 핵심 역할을 하며, 단순한 매핑 테이블일수도, 아주 복잡한 함수나 확률적인 모델일 수도 있다.

현재의 시간을 $t$, 현재 시간에서 agent에게 주어진 상태를 $s_t$라고 한다. 그리고, 취할 수 있는 액션을 $a_t$라고 해 보자. 그럼, policy $\pi$는 다음처럼 정의될 수 있다.

**Input**: $s_t$ (current state), $a_t$ (possible action at current time $t$)

**Output**: $\text{preference}$ (preference of action $a_t$)

$$
\text{preference} = \pi(s_t, a_t)
$$

$\text{preference}$는 현재 상태 $s_t$에서 어떤 action $a_t$를 취할지에 대한 선호도를 나타낸다. $s_t$에서 가능한 action $a_t$가 여러개가 존재할 수 있는데, $\pi(s_t, a_t)$가 가장 높은 액션 $a_t'$를 취할 수 있을 것이다.



### Value & Value Function

Reward가 매 action마다 얻을 수 있는 것이라면, value는 미래에 가능한 action들을 취해봤을 때 얻을 수 있는 reward들의 누적 합의 기댓값을 의미한다. 현재 시간을 $t$라고 해 보자, 현재 상태 이후부터 얻을 수 있는 reward의 합 $Q_t$는 다음처럼 말할 수 있다.

$$
Q_t = \sum_{i=t+1}^{\infty} r_i = r_{t+1} + r_{t+2} + \cdots
$$

$$
Q_t = r_{t+1} + Q_{t+1}
$$

Value $V_t$란, 현재 상태 $s_t$가 주어지고, agent가 action을 취할 수 있는 policy $\pi$가 주어졌을 때, 이 미래 reward의 누적값인 $Q_t$의 기댓값이다.

**Input**: $s_t$ (current state), $\pi$

**Output**: $V_t$ (value)

$$
V_t = v(s_t, \pi) = \mathbb{E}[Q_t]
$$

이때, value $V_t$를 계산해 주는 함수 $v$를 **value function**이라고 부른다. Reinforcement learning은 당장의 reward $r_{t+1}$보다는 value $V_t$를 최대화하는 action을 선택하도록 해야 한다.

Value function은 environment의 특성에 맞게 두 가지의 구현방법이 있다.

- Tabular solution method

  가능한 state의 수와 가능한 action 수가 셀수 있는 경우에 사용한다. 이 경우, value function을 단순한 mapping table로도 표현이 가능하다(그래서 이름도 tabular method라고 부른다).

- Approximate solution method

  가능한 state의 수가 무수히 많은 경우에 사용할 수 있다. 이때, value function의 입력 도메인(state)이 무한하므로 table로는 value function을 표현할 수 없고, function approximation을 사용한다. 보통 neural network를 approximator로 사용한다.



Reinforcement learning에서 가장 중요한 개념은 value function과 policy라고 볼 수 있다. Reinforcement learning 알고리즘들은 value function을 계산하고 그로부터 최적의 policy를 이끌어내는 과정으로 이루어지기 때문이다.



## Difference with Supervised Learning

Reinforcement learning은 결국, value를 예측하는 문제라고 볼 수도 있지 않을까? 주어진 상태에서 supervised learning 알고리즘을 사용하여 모든 액션에 대한 value를 예측하고 value값이 가장 큰 action을 취하면 되지 않을까?

하지만, reinforcement learning은 다음과 같은 차이점이 있다.

- 정답 라벨이 바로 주어지지 않는다. (delayed reward)

  Reinforcement learning에서 정답 라벨은 주어질 수 있다. 그러나, 바로 주어지지 않는다. 일단 action을 취해봐야 비로소 environment로부터 reward를 얻을 수 있다.

- 시간이라는 개념이 존재한다.

  Supervised learning의 경우, 시간이라는 개념이 보통 존재하지 않는다(항상 그렇지는 않지만). 존재한다고 하더라도, 일정 시간동안의 데이터는 미리 주어지며, 실시간으로 학습하지 않는다. 반면, reinforcement learning은 실시간으로 학습하면서 액션을 취한다.

- 데이터가 매우 비정형적이다.

  Reinforcement learning에서의 데이터는 dataframe이나 이미지로 끝나지 않고, 여러가지 센서(사람으로 치면 시각, 청각 등등)가 있을 수 있다.

- 데이터가 고정되지 않는다.

  Supervised learning은 미리 데이터가 주어지고, 정답 라벨이 주어진다. 그리고 그 데이터들로 학습을 하는데, reinforcement learning은 실시간으로 environment와 상호작용하면서 얻은 정보들로 학습해야 한다.

- Trials and errors

  Reinforcement learning에서는 시행착오를 통해 학습한다. 데이터가 주어지지 않은 채로 agent가 액션을 취하며, 그 액션을 통해 얻은 reward로 학습한다.

- 가능한 state의 수와 가능한 action 수가 너무 많다.

  사실상 supervised learning으로는 계산이 불가능할 수도 있다.



## Types of Reinforcement Learning

Reinforcement learning은 두 가지 타입으로 나눌 수 있다.

- Episodic task
- Continuous task

Episodic task란, 흔히 게임 같은 환경을 말한다. 시작과 끝이 존재한다. 게임으로 치면 게임 시작이 있고, 게임의 끝이 있다. 이때, 한 판의 게임을 episode라고 부른다.

Continuous task란, 시작과 끝이 없는 task를 의미한다. 현실 세계에서의 로봇이 그렇다. 한번 동작하면 끊임없이 environment와 통신한다.



## Unusual & Unexpected Stretagy in RL

Reinforcement learning은 최종 value를 최대화하면서 학습한다. 그리고, 최종 value를 가장 높게 하는 방법을 알아서 찾아나가는데, 이때, 그 방법이 소위 말해서 수단과 방법을 가리지 않는 방법일 수  있다. 또한, agent가 취하는 액션은 나중에 보면 최대 reward를 받는 방법이었다는 것이 드러나지만, 액션 하나하나를 보면 인간이 전혀 이해하지 못하는 방향의 액션일 수도 있다.

**그러므로 매우 안전하지 않을 수 있고 agent가 비도덕적으로 행동할 수 있다.**



## Exploitation-Exploration Dilema

Reinforcement learning에서의 agent는 최대한 많은 reward를 얻는 action을 선택해가며 문제를 해결해야 한다. 이미 알고 있는 문제 해결 방법중에서 가장 큰 reward를 얻을 수 있는 action을 선택해서 문제를 해결하는 것이 합리적일 것이다(**exploitation**). 그러나, agent는 새로운 길을 탐색해 나가면서 더 나은 길을 찾을 필요가 있다(**exploration**). 하지만, exploration과정은 많은 비용이 들 수도 있고 탐험 결과가 좋은 reward를 주는 경로가 아닐 수도 있다. 그럼에도 exploration은 필요하다.

- Exploitation

  이미 찾은 문제 해결 방법중에서 가장 나은 방법을 선택하는 것. 즉, 최대 reward를 찾아가는 것.

- Exploration

  새로운 길을 탐색하는 것. 많은 비용이 들지만, 새로운 길이 지금까지 가지고 있었던 해결 방법들 보다 더 나은 reward를 줄 수도 있다.

Exploration은 많은 비용이 들기 때문에, 당장은 reward를 얻지 못할 수도 있다. Exploitation을 하면 당장은 많은 reward를 얻을 수 있다. 이처럼 이 두가지는 서로 상반되고 동시에 좋아질 수는 없는데, 이를 exploitaiton-exploration dilema라고 부른다.

