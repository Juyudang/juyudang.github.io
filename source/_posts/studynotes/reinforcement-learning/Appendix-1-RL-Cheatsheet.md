---
title: Appendix 01. Which Algorithm Should be selected
toc: true
date: 2020-03-24 09:12:08
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Which Algorithm Should be Selected



많은 RL알고리즘 중에서 어떤 알고리즘을 선택해야 하는지 난감하다. 이럴때는 다음과 같은 과정을 거쳐서 적절한 알고리즘을 선택해야 한다.

1. 풀고자 하는 문제를 정의한다.
2. 문제에서, environment의 구성, agent의 행동 범위 등을 정의한다.
3. Environment를 기반으로 **continuous task인지, episodic task인지 구분한다.**
   1. Continuous task라면, $G_t$를 discounting, average reward중 하나를 사용해야만 한다.
   2. Episodic task라면, $G_t$에서 discounting을 사용하거나 사용하지 않아도 된다.
4. Function approximation을 사용할 것인지 결정한다.
   1. Function approximation을 사용할 것이면, 적절한 모델링이 필요하다.
      1. Value function만 function approximation할 수도 있다. (Continuous action이면 좀 힘들 수 있다.)
      2. Actor-Critic처럼 value function, policy모두 function approximation할 수도 있다.
5. Action이 discrete인지 확인해야 한다.
   1. Discrete하다면, softmax policy 또는 greedy policy, $\epsilon$-greedy policy로 갈 수도 있다.
   2. Contiunous하다면, policy function approximation을 통한 gaussian policy같은 것을 선택할 수도 있다.
6. Planning 여부를 결정한다.
   1. 방문하는 state의 개수가 한정적이라면 Dyna-Q 또는 Dyna-Q+를 사용할 수 있다.
   2. 방문하는 state가 많고, 각 state가 중복되기 힘든 상황이라면, planning은 매우 주의해서 사용해야 한다. 아직, open research area이다.
