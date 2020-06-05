---
title: 01. Introduction
toc: true
date: 2020-03-03 10:00:00
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Introduction



Reinforcement Learning이란, 무엇을 할지에 대해 학습하는 것이다. 다르게 말하면, 어떤 상황이 입력으로 들어가서 어떤 액션이 출력되는 함수를 학습하는 것이다.

Reinforcement learning에선, 두 가지 중요한 특징이 있는데, 다음과 같다.

- Trails & erros search
- Delayed rewards

물론 그 전에 environment를 인지할 수 있는 센서가 있어서 상황을 바탕으로 위 두 가지 특징이 발현된다.



## Overview of Reinforcement Learning

Supervised learning과 unsupervised learning은 다음과 같은 특성을 지닌다.

- 숫자로된 테이블 형태의 데이터가 존재한다. ($X: N × D$, $Y: N × 1$) 이 두가지 데이터를 모두 넣고 모델을 학습하게 된다.
- Static한 학습만 가능하다. 데이터를 추가하려면, 기존 데이터도 모두 넣고 fine-tuning 해야 하는 경우가 많다. 따라서, online-learning에 매우 불리하다.
- 시간이라는 개념이 없다(Sequence라는 개념은 있어도...). 그저 $X$를 받으면 $Y$를 줄 뿐. 
- Supervised learning의 경우, $Y$가 $X$ 동시에 주어지기에, 즉각적인 피드백이 있다.



반면, reinforcement learning은 다음과 같은 차이점이 있다.

-  데이터가 매우 비정형적이다. 로봇의 경우, environment로부터 받은 카메라 정보와 여라가지 센서 정보나 environment representation 정보 등이 있을 수 있다.
- Dynamic하게 학습한다. Reinforcement learning은 데이터를 모아서 데이터셋을 만들어서 학습하는 형태가 아니라, environment에서 action을 취한 결과 피드백을 얻고 학습하는 형태이다. 즉, 그 자체가 그냥 online learning이다.
- 어떤 액션을 취하면 즉각적인 피드백이 없을 수 있고, 게임이 끝날 때 까지 피드백을 얻지 못할 수도 있다. 따라서, 상대적으로 시간이라는 개념이 존재한다. 즉, 액션과 리워드가 동시에 주어지지 않고 중간에 일정 시간이 있을 수 있다.



### Unusual & Unexpected Stretagy in RL

Reinforcement learning은 최종 value를 최대화하면서 학습한다. 그리고, 최종 value를 가장 높게 하는 방법을 알아서 찾아나가는데, 이때, 그 방법이 소위 말해서 수단과 방법을 가리지 않는 방법일 수  있다. 또한, agent가 취하는 액션은 나중에 보면 최대 reward를 받는 방법이었다는 것이 드러나지만, 액션 하나하나를 보면 인간이 전혀 이해하지 못하는 방향의 액션일 수도 있다.



### Supervised Learning as Reinforcement Learning?

액션을 취하고 리워드를 얻는다는 것은 어떻게 보면 supervised learning과 연관지을 수도 있을 것이다. Environment가 $X$가 되고 그에 적절한 optimal action이 $Y$가 되는 것이다.

하지만, supervised learning을 쓰지 않고 reinforcement learning을 쓰는 이유가 있다.

- 계산 불가능할 정도로 많은 environment/state 경우의 수
- Supervised learning은 $X$와 $Y$를 동시에 필요로 하지만, $Y$가 있긴 한데, $X$ 동시에 주지 못하는 경우가 있다. 이때는 supervised learning을 할 수 없다.



## Exploitation-Exploration Dilema

Reinforcement learning에서의 agent는 최대한 많은 reward를 얻으면서 문제를 해결해야 한다. 이미 알고 있는 문제 해결 방법중에서 가장 큰 reward를 얻을 수 있는 방법을 선택해서 문제를 해결하는 것이 합리적일 것이다(exploitation). 그러나, agent는 새로운 길을 탐색해 나가면서 더 나은 길을 찾을 필요가 있다(exploration). 하지만, exploration과정은 많은 비용이 들 수도 있고 탐험 결과가 좋은 reward를 주는 경로가 아닐 수도 있다. 그럼에도 exploration은 필요하다.

- Exploitation

  이미 찾은 문제 해결 방법중에서 가장 나은 방법을 선택하는 것. 즉, 최대 reward를 찾아가는 것.

- Exploration

  새로운 길을 탐색하는 것. 많은 비용이 들지만, 새로운 길이 지금까지 가지고 있었던 해결 방법들 보다 더 나은 reward를 줄 수도 있다.

Exploration은 많은 비용이 들기 때문에, 당장은 reward를 얻지 못할 수도 있다. Exploitation을 하면 당장은 많은 reward를 얻을 수 있다. 이를 exploitaiton-exploration dilema라고 부른다.



## Elements of Reinforcement Learning

크게 4가지로 나눌 수 있다.

- Policy

  어떤 주어진 환경/상황에서 어떤 동작을 취해야 하는지에 대한 규칙 또는 정책, 또는 매핑 함수이다. RL에서 핵심 역할을 하며, 단순한 매핑 테이블일수도, 아주 복잡한 함수나 확률적인 모델일 수도 있다.

- Reward signal

  액션에 대한 결과적인 상황에 따라 agent가 어떤 reward를 받을지에 대한, 즉, 시스템의 목표를 어떻게 할 것인지에 대한 것이다.

- Value

  Reward는 액션마다 주어질 수 있는 것으로, 이것만 있으면 greedy하게 갈 수 있다. 이를 방지하기 위해 reward를 누적하고 시스템 전체의 reward를 바라볼 수 있게 하는 것이 value이다. 즉, 어떤 state $s_{i}$에 대한 value $value(s_i)$는 그 상태 이후, 미래의 상태들 $s_{i+1}, s_{i+2},...$로부터 얻을 수 있는 reward 기댓값이다. 즉, $\text{argmax} ~ value(s)$라 함은, 당장 greedy한 선택이 아니라, 미래에 총 reward가 높은 방향으로 액션을 선택할 수 있게 해 준다.

  value-function은 당장 앞에 놓인 action에 대해 value를 계산해주는 함수?

  Reinforcement learning의 주요 task는 이 value function을 추정하는 것이다. agent는 value function이 가장 높은 value를 리턴해주는 액션을 선택하면 되니까.

- Model of environment

  Agent가 상호작용하는 환경을 정의한 것.

이외에, episode라는 것이 있다. episode란, agent가 게임을 시작하고 끝날 때 까지의 기간, 즉, 한 게임을 의미한다. 다만, 게임같이 "한 게임"이라는 개념이 존재하는 episodic task가 있는 반면(바둑, 스타크래프트), "한 게임"을 정의할 수 없는, continuous task도 존재한다(로봇은 수명이 다할 때 까지 끊임없이 환경과 통신함). 이 경우에는 episode가 없다.

Reinforcement learning은 궁극적으로 reward를 최대화하는 것이지만, agent는 어떤 상황에서 높은 reward를 고르는게 아니라 value가 높은 쪽을 골라야 한다.

이 value function을 정의하는 방법은 두 가지가 있을 수 있다.

- Tabular solution method

  Value function은 deterministic하다. 보통 deterministic한 함수들은 입력과 출력 매핑을 테이블 형태로 표현가능하다. 따라서, deterministic한 방법을 tabular method라고 부른다.

  - Markov Decision Process

- Approximate solution method

  Value function은 확률적이다.

