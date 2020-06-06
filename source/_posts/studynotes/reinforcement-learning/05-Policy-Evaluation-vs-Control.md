---
title: 05. Policy Evaluation & Control
toc: true
date: 2020-03-03 10:00:04
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Policy Evaluation & Control



참고: Coursera Reinforcement Learning (Alberta Univ.)

현재 가지고있는 policy가 좋은지 평가하고(evaluation) 더 좋은 policy로 향상시키는 작업(control)을 의미한다. 현재 가지고 있는 policy $\pi$와 dynamic environment를 표현하는 $p(s',r|s,a)$분포가 있으면, dynamic programming을 통해 value function을 계산해낼 수 있고, 그 value function을 이용해서 policy를 평가(evaluation)할 수 있다. 또한, dynamic programming을 통해 더 나은 policy를 찾을 수 있다(control).




## Policy Evaluation

어떤 policy가 좋은지 평가하는 방법은 각 policy에 대해 value function을 계산하고, value function의 값을 비교하는 것이다. 이때, 주어진 policy에 대해 value function을 계산하는 과정을 **policy evaluation**이라고 한다.



### Iterative Policy Evaluation

주어진 policy를 이용하여 value function을 계산하는 한 가지 방법으로, dynamic programming을 통한 iterative 방법이다. 처음에 모든 state의 value를 0으로(또는 임의의 아무 숫자) 초기화시킨 후, state Bellman equation을 통해 모든 state의 value가 수렴할때까지 반복적으로 value function을 업데이트하게 된다.

방법은 다음과 같다.

1. Define threshold $\theta$
2. initialize $V$, initialize uniform policy $\pi$
3. while True:
   1. $V' \leftarrow v_{\pi}(s|V,\pi)$
   2. compute $\epsilon = \text{max}|V - V'|^2$
   3. if $\epsilon < \theta$:
      1. break
   4. $V \leftarrow V'$

이 루프를 계산하고나면, 해당 policy에 대해 최적에 가까운 value function을 계산할 수 있다. 이때, policy는 바뀌지 않는다.



## Policy Control

Policy control이란, 주어진 policy와 그것으로부터 계산해낸 value function을 가지고, 그 value function에서의 새로운 optimal policy를 찾는 과정을 말한다.



### Policy Improvement Theorem

Action value function $q(s, a)$가 존재하고, 두 policy $\pi_1, \pi_2$가 있고  각 policy에서 상태 $s$에서 취한 액션을 $a_1, a_2$라고 할 때, 다음을 만족하는 policy $\pi'$는 항상 $\pi_1, \pi_2$보다 항상 같거나 좋은 policy이다.
$$
\pi'(s) \leftarrow \underset{a}{\text{max }}(q(s, a_1), q(s, a_2))
$$
위 이론에 따라, 현재 policy보다 좀 더 좋은 policy를 찾는 방법은, 현재 policy를 바탕으로 계산한 value function에서 다시 greedy한 policy를 계산하는 것이다.
$$
\pi' = \underset{a}{\text{argmax} } \sum_{s',r} p(s',r|s,a)[r + \gamma \cdot v_{\pi}(s')] ~ \text{(for all state } s\text{)}
$$

한 가지 짚고 넘어가야 할 점은, 어떤 value function을 바탕으로 greedy한 policy를 선택했다고 해서, 그 policy로 다시 value function을 계산해보면, 그 value function에 대해서는 현재 policy가 greedy하지 않을 수 있다.

Policy를 찾은 후, 다시 value function을 계산하면 이전의 value function과 같지 않을 수 있다. 그리고, 새로 계산한 value function에 대해 greedy한 policy를 다시 찾으면 그 policy는 이전 policy와 다를 수 있다.

그래서, value function 계산과 policy 구하는 과정을 반복해서 수행하고 더 이상 달라지지 않으면, 수렴했다고 간주하고, 마지막 policy를 우리가 가지고 있는 environment에 대한 최종 optimal policy로 삼는다. 이렇게 value function과 policy 계산을 반복적으로 수행하는 것을 policy iteration이라고 부른다.



## Policy Iteration - Dynamic Programming

**Optimal policy를 찾는 알고리즘**으로, 다음과 같은 과정으로 이루어진다. 일단 어떤 state에서 어떤 액션을 취하면 어떤 immediate reward를 받는지는 이미 알고 있다고 가정한다. Immediate reward의 분포 $p(r|s)$은 exploration & exploitation 으로 추정해야 하거나 개발자가 이미 정해놓거나?

1. Initialize $\pi_0$. 즉, 최초의 policy를 만들고 이를 현재의 policy $\pi$로 삼는다. 최초의 policy는 아무거나로 한다. 모든 액션을 uniform distribution에 따라 선택하는 policy로 해도 된다.
2. Evaluate $\pi$. 즉, 주어진 policy에 대해 value function을 계산한다.
3. Control $\pi$. 즉, 계산된 value function을 바탕으로 모든 상태에서 greedy한 action을 선택하는 새로운 policy $\pi'$를 만든다.
4. $\pi' = \pi$라면, $\pi$를 반환하고 끝낸다. 아니라면, $\pi$에  $\pi'$를 대입하고 2번으로 간다.



이 과정을 통틀어서 policy iteration 방법이라고 부른다. 다음은 전체 pseudo code.

![image-20200121183345364](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200121183345364.png)



일단, 가장 처음 policy를 제외하고 이후 control에 의해 만들어진 모든 policy는 greedy하고 deterministic한 policy이다. 하지만, 이것은 새로 evaluate로 생성된 value function에서 greedy하지 않게 된다.

즉, evaluate과정을 거처서 만들어낸, 현 policy를 따르는 value function에서 현재 policy가 greedy하지 않게 되고,

control하는 과정을 거치면 greedy한 policy를 만들 수 있지만, 이건 또 다시 policy evalutation을 통해 더 좋은 policy가 있다는 것이 밝혀진다.

이 과정을 통해, 더 이상 좋은 policy가 없을 때 까지 수렴하게 된다.

![image-20200121183703146](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200121183703146.png)



## Generalized Policy Iteration

Policy Iteration의 일반적인 형태.

앞서 나온 policy iteration은 policy evaluation과 control를 번갈아가면서 수행했다. 그리고, evaluation에서는 value function이 수렴할때까지 loop를 돌렸고, 수렴한 다음에야 policy control을 시행했다. Policy control 또한 완전한 greedy한 deterministic policy를 선택했다. 하지만, generalized policy iteration은 다음처럼 작동한다.

![image-20200122120728463](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200122120728463.png)

Policy evaluation은 loop를 돌지 않고 한번만 회전하고 policy control또한 조금 완화된 greedy action을 선택하게끔 한다.



### Value Iteration

Generalized policy iteration의 한 방법으로, policy evaluation과 control를 번갈아서 수행하지 않고 evaluation을 policy와 관계없이 value값에 대해서만 수행해서 수렴시키고 control를 최종적으로 수행한다.

![image-20200122121014540](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200122121014540.png)

알고리즘은 위와 같은데, value funciton을 계산할 때, "어떤 state에서는 무조건 이 액션을 선택해"라고 말하는 policy를 넣지 않고, 그냥 value를 최대로 하는 액션을 선택하도록 한다. 그리고 그 최대 value로 업데이트한다. 이 과정을 반복하면 value function 혼자서 optimal에 수렴하게 되고 optimal value function을 이용해서 policy를 뽑아낸다.



### Asynchronous Dynamic Programming

한번 value function을 업데이트할때, 모든 state를 순차적으로 다 돌지 말고, 필요한 state에 대한 value만, 순서관계없이 업데이트하자는 것이라고 한다.

또한, 모든 state를 다 업데이트하는것이 아니라 관계있는 state들만 업데이트한다.



### Monte Carlo Methods

지금까지 dynamic programming을 통해 value function과 policy를 계산 및 추정했는데, dynamic programming을 통한 방법 외에도 여러가지 방법이 존재한다.

Monte carlo method는 하나의 state에 대해 각 액션을 많이 취해보고 Monte carlo estimation을 통해 value 추정값을 계산하자는 방법이다. 즉, 그 state에서 각각 액션을 많이 취해보고 얻은 reward들을 단순 평균내자는 이야기이다. 이 방법은 optimal policy를 매우 정확하게 찾을 것을 보장해준다.(단, action을 해서 reward를 한 trial이 많아야 한다.)

Monte carlo estimation의 단점은 모든 state에서 모든 액션을 많이 취해봐야 정확한 value function을 추정할 수 있는데, 그게 현실적으로 불가능하다.



### Brute-Force Estimation

Brute-force 방법은 간단하다. 가능한 모든 deterministic policy 조합을 나열하고 그중에서 optimal policy를 찾는 것을 말한다. 이 방법 역시 optimal policy를 반드시 찾을 것을 보장해준다. 하지만, action수에 따라 가능한 policy 조합이 exponential하게 증가한다. 그래서 사실상 적용이 불가능하다.