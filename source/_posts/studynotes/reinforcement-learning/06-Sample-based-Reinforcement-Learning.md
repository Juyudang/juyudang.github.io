---
title: 06. Sample-based Reinforcement Learning
toc: true
date: 2020-03-03 10:00:05
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning

---



# Sample-based Reinforcement Learning



참고: Coursera Reinforcement Learning (Alberta Univ.)

### Motivations

- 지금까지 K-arm bandit problem으로 reinforcement learning의 기초를 보았고, 그를 이용해서 exploration-exploitation dilemma를 보았다.
- 또한, Exploration, exploitation이 충분히 이루어져서 environment dynamic에 해당하는, transition probability $p(s',r|s,a)$가 이미 모델링되었다고 가정했을 때, MDP 환경에 한정해서 value function과 policy를 어떻게 정의하고 계산하는지 보았다.
- Bellman equation을 이용해서 dynamic programming 방식으로 policy iteration, value iteration 등, optimal value function과 optimal policy를 계산하는 방법을 보았다.

하지만, 여기서, 현실은 environment는 너무나도 복합적인 것이라, 모델링하기가 쉽지 않다. 그러나, 지금까지 해 왔던 방법들은 environment가 필수적으로 모델링되어 있어야 한다.

이렇게, environment를 모델링하는 데 있어서 어려움이 있다는 문제를 해결하는 방법 중 하나가 **Sample-based Reinforcement Learning**이다.



### Introduction

이름에서 알 수 있다시피, 이 방법의 주요 특징은, environment dynamic에 해당하는 transition probability function $p(s',r|s,a)$를 모델링하는 대신, sample-based로 추정해보겠다는 의미이다. 특히 **Monte Carlo Estimation**이 이용된다.



## Monte Carlo Estimation

Monte-Carlo Estimation이란, 파라미터의 기댓값을 계산하고자 할 때, 그 파라미터를 모델링하는 과정을 거치지 않고, 파라미터에 대한 샘플을 많이 얻어서 그 샘플들의 평균값을 기댓값으로 추정하는 방법을 의미한다.

샘플 개수가 많을수록, Central Limit Theorem에 의해, 샘플들의 평균은 실제 평균과 매우 가깝다고 확신할 수 있다.

여기서 직감할 수 있다시피, 정확한 기댓값을 추정하기 위해서는 상당히 많은 샘플이 필요하다.



### Implementation Overview

Episodic task로 예를 들려고 한다.

1. 일단 하나의 episode를 완주한다. 즉, 게임이 끝날 때 까지 일단 플레이를 한다. 지나온(또는 처해있었던) state, 취했던 action들, 받았던 reward들의 History는 기록해 둔다.

2. 위 history를 $S_0, A_0, S_1, R_1, A_1, S_2, ..., S_{T-1}, R_{T-1}, A_{t-1}, S_T, R_T$라는 sequence로 표현했을 떄, $T$에서 backward방향으로 reward 기댓값, 즉, value를 계산한다.

   (Final state는 정의에 따라 value가 0이다)
   $$
   G_T = 0
   $$

   $$
   G_{T-1} = R_{T} + \gamma G_T
   $$

   $$
   G_{T-2} = R_{T-1} + \gamma G_{T-1}
   $$

   $$
   \cdots
   $$

3. $G_t$를 $t$일때의 state였던 놈, $S_t$에 대한 value의샘플이라고 간주한다. 하나의 episode에 그 state를 지난 횟수만큼 샘플이 생긴다

4. 여러 episode를 플레이해본다.

5. 모아진 value sample을 이용해서 state-value function을 추정한다(평균내기).

하지만, 하나의 episode가 모든 state를 골고루 방문하지는 않으므로, 여러 episode를 시행해도, state마다 샘플 수는 다르다. 따라서 어떤 state는 state-value의 정확한 추정이 어려울 수 있다.



### Exploring Starts

Action value function도 state value function을 추정하는 방법과 똑같이 추정할 수 있다. Action value function을 굳이 쓰는 이유는 한 state에서 어떤 액션을 선택할지에 대한, 즉 policy를 찾는데 도움을 줄 수 있기 때문이다.

하지만, 만약, deterministic policy를 따르고 있으면, 각 state에서 특정 action만 수행한다. 따라서, 하나의 액션만 exploitation하게 되는데, 이러면, policy를 비교, 조정할 수 없다. 이에 대한 해결책 중 하나로, episode의 시작은 무조건 random state에서 random action을 취하도록 시작하는 것이다. 첫 번째 액션을 취한 이후로는 policy를 따르게 된다.

이 문제는 결국, exploration에 관한 문제이다.



### Monte Carlo Prediction

Reinforcement learning의 context에서, Monte Carlo prediction이란, 주어진 episode를 플레이한 history $S_0, A_0, R_1, S_1, A_1, ..., R_T, S_T$를 이용해서 value function을 추정하는 것을 의미한다.



## Monte Carlo for Policy Control

Monte Carlo를 통해 generalized policy iteration을 구현할 수 있다.

![image-20200125175225811](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200125175225811.png)

- Policy Evaluation: 1번의 Episode 수행 및 Monte Carlo Estimation
- Policy Improvement: 계산된 value function을 통한 policy의 greedify

이전과 같이 policy evaluation과 improvement를 반복하게 되며, evaluation 단계에서 Monte Carlo prediction을 적용하여 value function을 추정하게 된다. Improvement 단계에서는 추정된 value function을 바탕으로 greedy한 policy를 생성해 낸다.

한 번의 반복(evaluation-improvement) 동안, 단 1번의 episode를 플레이하기 때문에 evaluation 단계에서 value function을 완전히 추정하지 않는다. Evaluation을 완성하려면 수많은 episode를 플레이하고 value function을 제대로 추정하고 improvement로 넘어가야 할 것이다. 하지만, 엄청 오래 걸릴 것이다.

Pseudo-code는 다음과 같다.

```python
def monte_carlo_gpi(states, actions, gamma):
    pi = initialize_policies(states)
    action_values = initialize_action_values(states, actions)
    returns = initialize_rewards(states, actions)
    
    while True:
        s0, a0 = exploring_starts(states, actions)
        estimated_action_values, history, T = play_one_episode(pi)
        G = 0
        
        for t in range(T-1, 0, by=-1):
            G = history[t+1]["reward"] + gamma * G
            
            state = history[t+1]["state"]
            action = history[t+1]["action"]
            
            returns[state][action].append(G)
            
            action_values[state][action] = mean(returns[state][action])
            pi[state] = argmax(action_values[state][action])
```



### Epsilon-soft Policy

Exploring starts 방식은 deterministic policy 환경에서 출발점에서나마 랜덤으로 state와 action을 선택하게 함으로써, 모든 state들이 그래도 한번씩은 다 방문되도록 하게끔 하는 것이다. 하지만, 다음 문제점이 있다.

- State 개수가 너무 많을 경우, 첫 시작을 임의로 시작한다고 한들, 모든 state를 다 방문하기엔 역부족이다. 계산 불가능할 정도로 많은 시도횟수를 요구할 것이다.

하지만, exploring 방법은 여전히 필요하며, exploitation만 할 수는 없다. exploring starts의 대안으로 나온 것이 바로 $\epsilon$-soft 방식이다. 이것은 $\epsilon$-greedy을 포함하는 상위 개념으로, optimal action에는 좀 높은 확률을 두고, 나머지 액션은 확률 0이 아니라 작은 확률을 설정해 두는 것이다.

구현 방법은 value evaluation에서 계산된 value function으로부터 가장 좋은 action을 뽑아내고, 그 액션에 좀 높은 확률을 주고, 나머지 액션은 작은 확률을 준다.

![image-20200126105656485](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200126105656485.png)

$$\epsilon$$-soft는 stochastic policy이다. 즉, optimal policy보다는 value 기댓값이 적다. 하지만, 적절히 greedy한 액션도 취해가면서 확률적으로 많은 state를 방문할 수 있게 해 준다.