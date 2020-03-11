---
title: 09. Models and Planning
toc: true
date: 2020-03-11 10:00:08
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Models and Planning



참고: Coursera Reinforcement Learning (Alberta Univ.)

지금까지, 두 가지 경우를 보았다.

1. Agent가 environment를 알고 있어서 실제 action을 취하지 않고, value function을 계산하고 optimal policy를 찾을 수 있다. (**Dynamic programming**)
2. Agent가 environment를 알 수 없어서, 오직 action을 취해서 얻은 sample들로만 value function을 추정했다. (**Sample-based**)



하지만, 이번에는 environment도 같이 추정해서 value function을 더 정확히 추정하고 더 나아가 더 나은 policy를  추정하는 방법을 찾고자 한다.

(즉, 두 가지 방법을 합친 것)



## Models

Environment를 모델링한 것이라고 보면 된다.

Environment를 모델링함으로써,

- Experiment를 통한 sampling이 가능하다.
- 각 event(action, reward가 생기는)의 확률을 알고, likelihood를 계산할 수 있다. 또는 marginalization 등 확률적인 계산들이 가능하다.



Model에는 **sample model**과 **distribution model** 두 가지가 있을 수 있다.

모델은 반드시 정확히 environment를 향해 가도록 모델링해야 한다. Bias되면 답이 없다.



### Sample Models

이 모델은 experiment를 통한 샘플링이 가능하도록 모델링한 것이다. 다음의 특징이 있다.

- 모델링하기 간단하다. 크기가 작다. Joint 확률까지는 필요없다.

  "주사위 5개의 결과"를 예로 들어보면, 주사위 1개의 확률분포만 모델링하면 된다.



### Distribution Models

이 모델은 각 state가 될 확률, reward의 확률분포를 알 수 있도록 모델링한 것이다.

- 모델링하기 복잡하고 크기가 크다. 가능한 경우의 수에 대해서 (joint)확률을 매겨야 한다.

  "주사위 5개의 결과"를 예로 들어보면, distribution model은 $6^5$개의 확률을 계산할 수 있어야 한다.



## Planning

Model을 통해 experiment를 시행해서 episode를 만들고(샘플링), 그것을 이용해서 value function을 업데이트하는 과정을 의미한다.

Model은 sample model에 해당한다.



### Random-sample One-step Tabular Q-Learning

이 방법은 planning중 하나의 방법으로, state transition dynamic을 sample model로 알고 있다고 가정한다. 또한, (시뮬레이션 중에)action을 어떻게 선택할지에 대한 전략도 있을 것이라고 가정한다.

다음과 같은 과정으로 이루어진다.

1. 첫 state와 action을 선택

2. Model로부터 다음 state를 샘플링하고(given current state, action), action 선택 전략에 따라 action을 뽑음

3. Q-learning 알고리즘에 따라 value function을 업데이트
   $$
   Q(S_t, A_t) = Q(S_t, A_t) + \alpha \cdot (R_{t+1} + \gamma \cdot \text{max} ~ Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
   $$

4. 2,3번 반복

5. Value function이 수렴했으면, policy update
   $$
   \pi_* = \underset{a}{ \text{max} } ~ Q(s,a)
   $$
   
   $$
   \pi \leftarrow \pi_*
   $$
   
   

### Advantage of Planning

Planning의 1-step 속도는 당연히 현실에서 1-step 가는것보다 빠르다. 따라서, 현실에서 action을 취하고 다음 action을 취하는 그 사이 간격에서 planning을 수행해줌으로써, value function의 효율적인 수렴이 가능할 수도 있다.

![image-20200217150229429](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200217150229429.png)



## Dyna Architecture

- Direct RL: Environment와 상호작용하면서 얻은 샘플로 value function을 업데이트하는 것.
- Planning: Environment를 모델링한 모델을 통해 얻은 샘플로 value function을 업데이트하는 것.

이라고 정의해보면(공식적인 정의는 아닌 듯 하다), 결국, 이 두 가지를 적절히 섞어서 value function을 업데이트하면, environment와 상호작용한 샘플이 적어도 효율적인 업데이트가 가능할 것이다.

이것을 구현한 것 중 하나로, **Dyna architecture**가 있다.

![image-20200217204140527](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200217204140527.png)

Dyna architecture는 environment와 상호작용하면서 얻은 진짜 샘플로 value function을 업데이트함과 동시에 model도 학습해야 한다. 모델은 진짜 샘플로 학습된 이후, 시뮬레이션 sequence를 생성해서 value function 학습에 이용되게 한다.

결국 이 모든건, environment를 추정하는 과정이라고 볼 수 있다. 어찌됬든, environment를 경험해서 배우고 그 environment에서 각 state의 최적 액션을 배우고자 함이니까.

다만, 첫 번째 episode에서는 큰 효과를 발휘하지 못할 수도 있다. 모든 state가 처음이고, value function이 0으로 초기화되어있기 때문.

![image-20200217204928616](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200217204928616.png)

하지만, 첫 번째 episode를 마치고나면, reward로 인해 value function이 조금 업데이트되고, model이 학습되면서 planning에 의해 value function이 업데이트가 시작된다. 첫 번째 episode에서 방문했던 모든 state가 사실상 업데이트가 될 수 있으며, 이에 따라, 매우 소수의 episode만을 가지고도 상당히 정확한 policy를 얻을 수 있다.



### Tabular Dyna-Q Algorithm

Environment가 deterministic하다고 가정한 Dyna architecture.

Dyna-Q란, Dyna architecture에서 Q-learning을 채용한 알고리즘을 말한다.

다음은 Dyna 알고리즘의 일종인 **tabular Dyna-Q 알고리즘**의 pseudo code.

![image-20200217205755986](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200217205755986.png)

이때, $Model(S,A) \leftarrow R, S'$의 의미는, model에다가 $S$상태일때, $A$를 취하면 무조건 $R$의 리워드를 얻고 $S'$상태로 간다고 매핑해 두라는 의미이다. (Tabular Dyna-Q 알고리즘은 deterministic environment라고 가정한다. 물론, 알고리즘을 수정하면 non-deterministic하게도 할 수 있겠지)

한번의 step을 할 때 마다 여러번(여기서는 $$n$$번)의 planning이 일어날 수 있다. Planning은 많이많이 해야 한다.

Planning은 랜덤으로 start state를 고르고 1 step만 간다.



### Random Search in Dyna-Q

문제가 있다. 위에서 본 Dyna-Q에서는 planning을 랜덤으로 고른 state에서 진행하고 있다. 만약, 미로찾기에서, 모든 step의 reward는 0이고, final state에서만 +1일 경우, 1회의 episode를 마친 후면, final state의 바로 앞의 state의 value만 업데이트된다(TD종류의 알고리즘을 사용했다면). 그리고, planning은 반드시 아래 사진의 위치를 골라야 value가 업데이트될 것이다.

![image-20200217211340623](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200217211340623.png)

업데이트 가능 위치가 저기 하나뿐이다.

이것은 매우 많은 planning이 업데이트를 실패하게끔 만들며, environment 크기가 클수록 상황은 더 심각해진다.

다음의 개선책이 있겠다.

- Model이 value의 변화를 관찰하고 있다가, value가 변하는 state와 그  action을 기록한다. 그 state로 오게 만들었던 과거 state와 action을 업데이트하기 위해 backward로 간다.
- Update되고 있는 state, action pair가 있는 queue를 관리한다. (위 사진에선 state, action pair가 1개만 queue에 있겠지). 이 queue는 priority queue로, 어느 한쪽의 state만 업데이트되게 하는 것을 막아주는 목적이다. Priority가 높은 놈이 선택되어 그 놈부터 backward로 planning이 이루어지게 된다. 만약, 한 state가 일정 이상 업데이트가 이루어지면 priority를 낮춘다.



### Inaccurate Models

모델이 environment를 반영못하는 경우는 두 가지가 있을 수 있다. 알고리즘은 제대로 되있다고 가정한다.

- 학습 초기. Environment와의 interaction 횟수가 매우 적을 때
- Environment가 변할때

이때는 문제가 될 수 있는게, planning이 잘못된 모델로부터 얻은 transition으로 value를 학습하게 된다.

첫번째 문제는 어쩔 수 없다. 그러나, 두번째 문제는 대처해야할 필요가 있다.



### Exploration-Exploitation Trade-off in Chaning Environment

Environment가 변하는 것에 대처하기 위해서는 모델도 exploration을 해야 한다. 그저 옛 모델에 머물 수는 없다. 따라서, 모델이 모델링하고 있는 가상의 environment또한 exploration-exploitation dilema가 생기게 된다. Environment가 언제 변할지 알수 없고, 변하지 않는다면, exploration은 손해를 낳는다.

Exploration을 모델에서 직접 할 수는 없고, policy 학습에 영향을 주어 environment에서 exploration을 하도록 유도하는 방식으로 이루어진다. 모델을 수정하지 않는 이상 모델의 가상 environment가 고정되어 있기 때문에 거기서 exploration해봤자 아무 의미가 없다.

즉, Planning 또한 exploration을 유도해야 한다. 아예 environment에서 bahavior policy의 exploration에만 의존하면 모델로 인한 planning이 너무 잘못된 방향으로 policy를 유도한다.



### Dyna-Q+ Algorithm

Environment가 변할때를 반영하기 위해 Dyna-Q를 변형한 알고리즘이다. Dyna architecture에서 planning은 과거에 방문한 state만 업데이트하게 되는데, 문제는 environment가 변함으로써, 방문한 놈들의 reward, next state 분포가 변할 수 있다는 것. 따라서, 방문한 지 오래된 state에 대해서는 보너스 reward를 **모델에서** 할당한다. 그리고 그 보너스 reward를 value를 계산할 때 반영하게 된다.
$$
R \leftarrow R + \kappa \sqrt{\tau}
$$
이때, $\kappa$는 작은 상수이며, $\tau$는 방문한지 얼마나 됬는지에 대한 time step이다. 즉, 방문한 지 오래된 놈이면 reward를 증가시킨다.

(Value에다가 더하지 말고, action에 따른 결과 reward에다가 더하자.)

그럼 취한 지 오래된 action으로 인한 transition이 장려될 것이다.