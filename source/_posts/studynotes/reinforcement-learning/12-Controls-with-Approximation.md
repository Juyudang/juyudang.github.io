---
title: 12. Controls with Approximation
toc: true
date: 2020-03-22 09:22:08
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Controls with Approximation



Value function을 function approximation을 통해 모델링했다면, 이제, 주어진 value function으로 policy를 계산해 낼 차례이다.



## Representation of Actions

Sarsa, expected Sarsa, Q-learning등을 하려면, state value function보단, action value function이 더 유용한데, 앞에서 봤던 function approximation방법은 state value function을 추정하는 것이었다.

Action을 추가한 function approximation은 두 가지 방법이 있을 수 있다.

1. Action별로 따로 function을 모델링한다.

   ![image-20200307093934585](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307093934585.png)

   즉, 어떤 action 1개당 1개의 function이 있는 셈이며, 위 그림처럼 stacking해서 하나의 linear 형태로 표현이 가능하다. 하지만, 이 경우, action끼리의 generalization이 일어나지 않는다. 즉, $a_0$의 function과 $a_1$의 function은 서로 다른 weight를 사용하기 때문에 서로 영향을 미치지 못한다.

   신경망으로 치자면 다음처럼 될 것이다.

   ![image-20200307094147927](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307094147927.png)

   마지막 hidden layer의 output이 state의 representation이며, 그것과 최종 output layer사이의 weight는 action끼리 서로 공유되지 않는다.

2. Action도 function의 입력으로 넣는다.

   ![image-20200307094258479](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307094258479.png)

   신경망의 output은 하나로 통일해버리고, action은 input으로 넣는 것이다. 이렇게되면, state뿐 아니라 action 사이에서도 weight가 공유되므로, action generalization도 수행될 것이다.



## Controls with Function Approximation

### Sarsa with Function Approximation

방법은 tabular Sarsa와 semi-gradient TD와 상당히 유사하다. Value function을 weight로 모델링한 후, weight와 policy를 initialization한다. 그리고, 다음 코드를 구현한다.

![image-20200307094518046](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307094518046.png)

Semi-gradient 방법을 사용한다.



### Expected Sarsa with Function Approximation

Sarsa와 Expected Sarsa는 Bellman equation 형태만 조금 다르므로, 그것만 수정해주면 된다. 다음은 Sarsa의 업데이트 방식이다.
$$
w \leftarrow w + \alpha(R_{t+1} + \gamma \cdot \hat{q}(s_{t+1}, a_{t+1}, w) - \hat{q}(s_t, a_t, w)) \nabla \hat{q}(s_t, a_t, w)
$$
다음은 expected Sarsa의 업데이트 방식이다.
$$
w \leftarrow w + \alpha(R_{t+1} + \gamma \cdot \sum_{a_{t+1}} \pi(a_{t+1}|s_{t+1}) \hat{q}(s_{t+1}, a_{t+1}, w) - \hat{q}(s_t, a_t, w)) \nabla \hat{q}(s_t, a_t, w)
$$
다만, action방향으로 expectation을 계산할 수 있어야 한다. (여기서는 action set이 finite하다고 가정)



### Q-learning with Function Approximation

Q-learning은 expected Sarsa의 특수한 형태이다. 즉, expectation값을 구하는 대신 maximum action value를 취한다.
$$
w \leftarrow w + \alpha (R_{t+1} + \gamma \cdot \underset{a_{t+1}}{ \text{argmax} } ~ \hat{q}(s_{t+1}, a_{t+1}, w) - \hat{q}(s_t, a_t, w)) \nabla \hat{q}(s_t, a_t, w)
$$


## Exploration with Function Approximation

Function approximation에서도 exploration-exploitation dilema가 발생한다. 따라서, 이를 완화시켜야 하는데, function approxmiation은 각 state 사이에 value generalization이 이뤄지기 때문에 tabular settings보다 exploration에서 제한적이다.

예를들어, optimistic initial value를 들어본다. Optimistic initial value setting에서는 처음에 value function을 매우 높은 값으로 초기화하고, 어떤 state를 방문할수록 방문한 state의 value function이 낮아지며, 자연스럽게 아직 방문하지 않은 state에 방문하게 된다. 학습이 진행될수록, 낮은 value를 가진 state의 value funciton은 낮은 값이 되어 더 이상 방문하지 않게 될 것이다.

하지만, function approximation setting에서는 이것이 유효하지 않은데, generalization이 이뤄지기 때문에 방문하지 않은 state의 value function도 같이 낮아진다는 것이다. 따라서, optimistic initial value의 의도와는 다르게 흘러가며, exploration이 제대로 이뤄지지 않는다.



### $\epsilon$-greedy with Function Approximation

하지만, $\epsilon$-greedy 방식은 어떤 function approximation 방법과도 융합될 수 있다.

- $1 - \epsilon$확률에 따라 exploitation
  $$
  a \leftarrow \underset{a}{ \text{argmax} } ~ \hat{q}(s, a, w)
  $$

- $\epsilon$확률에 따라 exploration
  $$
  a \leftarrow \text{random}(a)
  $$

이외에 function approximation setting에서의 exploration-exploitation 조화는 아직 연구중인 분야라고 한다.



## Average Rewards

지금까지, 어떤 state에서 어떤 action을 취했을 때의 value는 discounting을 이용한 future reward의 합으로 정의했다. 하지만, 이것은 discount라는 hyperparameter가 존재하며, 이것을 정하는 것은 어떤 문제를 푸느냐에 따라 크게 달라질 수 있다. 때로는 discount rate가 알고리즘을 잘못된 방향으로 학습시킬 수 있다(큰 reward를 받는게 너무 먼 미래인 경우 discount가 너무 많이 된다). 이것은 continuous task일때도, 마찬가지로, 당장은 작은 reward, 먼 미래에 다소 큰 reward를 받는 액션 중 택하는 문제에서, discount는 큰 영향을 준다.

Continuous task를 위한 RL알고리즘들은 보통 discounting대신 average reward방식을 사용한다고 한다.

Average reward는 이것을 해결하기 위해서 나왔으며, 어떤 policy를 따를 때, 앞으로 받을 reward의 평균을 말한다.
$$
r(\pi) = \lim_{h \rightarrow \infty} \frac{1}{h} \sum_{t=1}^h E[R_t|S_t,A_{0:t-1} \sim \pi]
$$
Value의 평균이 아니라, reward의 평균이다. 이는 다음처럼 generalize할 수 있다.
$$
r(\pi) = \sum_s \mu_{\pi}(s) \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)r
$$
$\mu_{\pi}(s)$는 $s$가 해당 policy $\pi$에 따른 방문 횟수 비율 분포이다.

예를들어, 다음 environment가 있다고 했을 때,

![image-20200307112210435](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307112210435.png)

Left로만 가라고 하는 deterministic policy $\pi_L$일때, state $s$에서의 average reward $r(\pi_L)$은 0.2이다. 왼쪽으로만 가라고 하고, 왼쪽으로 갔을 때, 총 5개의 state를 지나며, 총 +1의 reward를 얻을 수 있으니, 왼쪽으로 가는 action을 선택했을 때 평균 reward는 0.2이다. 오른쪽으로는 갈 수가 없으므로, 고려하지 않는다.

반면, 오른쪽으로만 가라고 하는 deterministic poliyc $\pi_R$의 경우, average reward $r(\pi_R)$은 0.4이다.

따라서, 두 개의 policy를 average reward로 비교했을 때, $\pi_R$이 더 좋다고 할 수 있다. **즉, average reward를 통해 어떤 policy가 더 좋은지 판단할 수 있다.**

Policy 하나당 하나의 average reward를 계산할 수 있다.

하나의 policy에서 average reward를 계산했다면, 그 policy내에서 어느 action이 좋은지에 대해서도 판단할 수 있다. 즉, value function을 새롭게 계산할 수 있다는 것이다. 이때, value는 다음처럼 정의한다.
$$
G_t = (R_{t+1} - r(\pi)) + (R_{t+2} - r(\pi)) + \cdots
$$
이때, $R - r(\pi)$를 differential return이라고 부른다. 즉, value를 미래 reward의 총합이 아닌, differential reward의 총합으로 differential return을 정의한다.

따라서, action value Bellman equation은 다음처럼 변형된다.
$$
q(s, a) = \sum_{s', r} p(s', r|s, a) \sum_{a'} [r - r(\pi) + \sum_{a'}\pi(a'|s') q(s',a')]
$$


### Differential Sarsa

Differential return을 이용해서 Sarsa를 변형한 것을 말한다.

![image-20200307113711291](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307113711291.png)

그런데, average reward를 계산할 때, TD error로 계산하는게 더 효과적이라고 한다.

![image-20200307113828452](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200307113828452.png)

(이렇게 되면 average reward가 아니라 average value로 differential return을 계산?)