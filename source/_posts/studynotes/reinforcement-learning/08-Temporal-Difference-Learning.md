---
title: 08. Temporal Difference Learning
toc: true
date: 2020-03-03 10:00:07
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Temporal Difference Learning



참고: Coursera Reinforcement Learning (Alberta Univ.)

지금까지 봐 왔던 RL 학습법은, 일단 한 episode를 다 플레이하고 value function을 업데이트했으나, TD(Temporal difference) learning 방식은, episode를 플레이함과 동시에 value function을 업데이트한다. $t-1$일때의 state $s_{t-1}$에서, 액션 $a_{t-1}$을 취해서 reward $R_{t}$를 얻었다면, $v(s_{t-1})$은 즉시 업데이트 가능하다. 결국, value function은 Bellman equation에서 보다시피 next state에서의 reward와  value에 의해서 계산될 수 있기 때문.

$$
v(s_{t}) = E[G_t|s=s_t]
$$

$$
v(s_t) = E[R_t + \gamma G_{t+1}|s=s_t]
$$

또한, Monte Carlo estimation을 통해 기댓값을 추정한다면,

$$
v(s_t) \approx \frac{1}{n}\sum_{i} [R_t + \gamma G_t]
$$

일 것인데, 이러면, 모든 샘플을 저장하고 있어야 한다. 따라서, 샘플 하나 모으고 반영하고 하나 모으고 반영하기 위해, incremental 하게 구현하려면 다음과 같이 value function 식을 수정할 수 있다.

$$
v(s_t) = v(s_t) + \alpha(G_t - v(s_t))
$$

$$
v(s_t) = v(s_t) + \alpha(R_{t+1} + \gamma G_{t+1} - v(s_t))
$$

여기서 $\alpha$는 step size이며, $\frac{1}{n}$이 들어간다고 보면 된다.



### Temporal Difference Error

$R_{t+1} + \gamma G_{t+1} - v(s_t)$부분을 말한다. 원래의 평균치인 $v(s_t)$로부터 얼만큼 업데이트 될 것인지를 나타내기도 하며, 원래 기댓값와 새로운 기댓값의 오차 정도로 이해하면 될듯.



## TD(0) Algorithm

Monte Carlo prediction을 위해서 한 episode의 history를 저장해놓고 있어야 했지만, TD(0)에서는 바로 앞 전 previous time에서의 정보만 저장해두고, time step마다 업데이트를 incremental하게 시행한다.

(과거 1스텝만 본다고 해서 TD(0)이다)

![image-20200131163518632](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200131163518632.png)

TD는 MC방법보다는 low variance라고 한다. 즉, 적어도 일관된 대답은 내놓을 수 있게 학습된다(low noise).



## TD vs MC

Temporal difference방법은 episode가 끝나지 않아도 value function의 업데이트가 이루어지지만, 일반적인 Monte Carlo 방식은 episode가 완전히 끝나야 value function의 업데이트가 이루어진다.

수렴속도가 TD가 더 빠르다는 실험 결과가 있다.

![image-20200207132946501](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200207132946501.png)

y축은 실제 value와 예측된 value의 RMS(Root mean squared) error이고, x축은 episode 횟수이다.



## Batch TD(0)

Episode를 무한히 만들 수 없고, 한정된 episode만 이용할 수 있을 때, (즉, 데이터셋이 한정되어 있음) batch TD(0)를  사용하기도 한다.

먼저 episode가 100개가 있다고 가정하면, 각 episode를 돌면서 다음을 계산한다.

$$
\text{increment} = \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$

episode 100개를 모두 한번씩 보는 것을 1 batch라고 하면, 1 batch를 모두 볼때까지 value function $V(S)$를 업데이트 하지 않는다. 100개를 모두 보고 난 후, 각 episode마다 계산된 $\text{increment}$를 state마다 모두 합해서 $V(S)$를 업데이트하게 된다. 그리고, 다시, 업데이트된 $V(S)$를 이용해서 episode 100개를 다시 반복해서 본다.

일반적인 TD(0)는 1 episode 를 돌때도 즉시 value function을 업데이트하지만, batch TD(0)는 모든 episode를 본 후, 각각 $\text{increment}$를 계산하고 이들의 합으로 value function을 업데이트한다. 즉, 모든 episode를 본 후, 비로소 한 번의 업데이트가 이루어진다.



## Temporal Difference Learning for Control

Control이라 함은, policy control을 의미한다.TD(0) 알고리즘은 policy evaluation 또는 prediction으로 이용해서 한 episode가 끝나지 않더라도, value function을 업데이트가 가능하게 해 주었다. 하지만, value function을 한번 업데이트했다면, policy또한 업데이트가 가능할 것이다.

Policy control을 하려면, state value보단, action value가 편하다. State가 주어졌을 때, 큰 action value를 가지는 action을 선택하면 되기 때문.

Action value function을 TD(0)알고리즘에 이용하기 위해, increment하게 바꾼 형태는 다음과 같다.

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

(State value의 incremental 식이랑 똑같다)



### Sarsa Algorithm

한 episode 내에서 한 time step마다 value function을 업데이트하고, policy를 업데이트하는, generalized policy iteration 과정을 거치게 할 수도 있다. TD(0)를 generalized policy control에도 응용한 것으로, 이것을 **Sarsa algorithm**이라고 부른다.

일반 Monte Carlo 방식에서는, 한 episode가 모두 마무리되어야 value function을 업데이트하고 policy를 학습한다. 그러나, 이 경우, 초기 policy를 어떻게 initialize했느냐에 따라 학습 초창기에 episode가 지나치게 길어질 수 있다. 물론, 몇 episode가 끝나면 policy가 업데이트되어 어느정도 빨라지겠지만 말이다.

이것은 한 episode가 모두 끝나고나서 업데이트하는 방식의 단점이라고 할 수 있으며, Sarsa algorithm은 이를 피하게 해 준다. 가는 도중에도 policy가 업데이트되어 수렴도 빠르다.

Value function 업데이트 수식은 다음과 같다.(action value 이용)

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

Sarsa 알고리즘은 다음과 같은 특징이 있다.

- **Action value Bellman equation**와 Monte Carlo estimation을 이용해서 policy evaluation을 수행한다.
- Policy evaluation과 improvement를 번갈아 수행한다. 즉, **policy iteration**의 방식을 따른다.
- 기본적으로 behavior policy와 target policy가 같은 **on-policy learning이다.** 즉, $A_{t+1} \sim \pi$이다.



### Q-Learning

Sarsa 알고리즘은 기본적으로 on-policy를 기반으로 한다. (물론, off-policy로 구성할 수도 있을 것 같다.) Q-learning은 off policy TD control 방법으로, action value function은 다음처럼 업데이트한다.

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \cdot \underset{a}{ \text{max} }~Q(S_{t+1}, a) - Q(S_t, A_t))
$$

Q-learning은 Sarsa와는 달리, 다음의 특징이 있다.

- **Action value Bellman *optimality* equation**과  Monte Carlo estimation을 통해 policy evaluation을 수행한다.
- Policy evaluation만 계속 하다가 마지막에 policy improvement를 한다. 즉, **value iteration**의 방식이다.
- Behavior  policy대로 action을 취하고 episode가 만들어지지만, value function의 업데이트는 greedy policy, 즉, target policy를 따르도록 업데이트된다. 위 식에서 $a \sim \pi^*$가 되며, 이는 **Q-learning이 off-policy learning이라는 것을 알려준다.**



### Q-Learning with/without Importance Ratio

**Q-learning은 importance ratio를 곱할 필요가 없다.** 왜냐하면, behavior policy에서 뽑은 액션대로 업데이트를 바로 하지 않기 때문이다. Sarsa 알고리즘에서 target policy를 따로 둔다면, importance ratio를 곱해주어야 겠지만, Q-learning은 value function 업데이트는 behavior policy의 action을 사용하지 않는다. 무조건, maximum value를 가지는 액션에 따라 업데이트하게 되며, 이는, target policy에 따라 업데이트하는 것이다. (어차피 target policy는 maximum action만 확률이 1.0이고 나머진 0이다)



### Sarsa vs Q-Learning

Sarsa는 policy iteration 방법이고, Q-learning은 value iteration 방법이다. Sarsa는 아주 optimal인 path를 찾는게 느리며, 못찾을 수도 있다. 그러나, 매우 reliable한 path를 찾아서 간다.

Q-learning은 value iteration을 통해 optimal value를 찾고 바로 optimal policy를 계산해낸다. 이것은 Q-learning이 reliablity와 상관없이 가장 빠른 길을 찾게 한다.

Reliable path란, path 중간에 매우 부정적인 reward를 주는 위험요소가 없는 path이다. Sarsa와 Q-learning이 모두 epsilon-greedy behavior policy를 사용한다는 가정 하에, 다음 그림을 생각해보면,

![image-20200213152009871](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200213152009871.png)

Cliff(절벽)에 빠지면 -100 reward, 그냥 1스탭은 -1 reward를 얻는다. $S$에서 $G$로 가야 한다.

Sarsa는 파란색 길을 찾을 가능성이 더 높다. 반면, Q-learning은 빨간색 길을 찾을 확률이 더 높다. 그러나, 이때의 단점은, epsilon-greedy의 특성상, exploration이 발동될 수 있고, cliff에 빠질 확률이 있다.

Sarsa는 episode의 모든 액션을 샘플로 포함시키면서 value function을 계산하고 policy를 업데이트하기에, epsilon의 확률로 인한 다음 액션도 고려하게 된다.

반면, Q-learning은 episode에서 취한 액션이 최적 액션이 아닐 경우, 그 액션 결과는 보지 않는다. 다음 액션이 cliff로 빠지는 액션이라면, 그 액션은 당연히 최적 액션이 아니고, cliff에 빠졌다는 결과는 보지 않게 된다. 그 결과, 아주 optimal value와 optimal policy는 빠르게 찾지만, 위험할 수 있다.

Sarsa는 학습하는 policy와 액션을 취하는 policy가 같으니까 exploration을 고려하면서 업데이트하게 되고, Q-learning은 target policy를 업데이트하지만, exploration하는 behavior policy를 그다지 고려하지 않는다. 그저 최적 루트만 찾게 된다.



### Expected Sarsa Algorithm

Sarsa의 value function 업데이트 식은 다음과 같다.

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \cdot Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

즉, 샘플링한 action을 기반으로 value function을 업데이트하게 된다.

Expected Sarsa는 다음과 같이 식을 수정한다.

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha (R_{t+1} + \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))
$$

즉, action 샘플 방향으로만 업데이트하지 말고, 가능한 모든 액션 방향을 고려하자는 것이다.

$$
Q(S_{t+1}, A_{t+1}) \rightarrow \sum_{a'} \pi(a'|S_{t+1})Q(S_{t+1}, A_{t+1})
$$

Action value를 업데이트할때, 다음 state의 action value를 고려해야 하는데, Sarsa에서는 다음 state에서 episode에서 취한 action만을 이용한다. 즉, 다른 액션을 통한 action value는 이용하지 않는다.

Expected Sarsa에서는 episode에서 취한 action도 고려하면서 아예 policy에 따른 기댓값으로 action value를 계산한다.

Expected Sarsa도 Monte Carlo 방식이다. 다만, action value update식만 조금 다를 뿐.

**다만, 한 state에서의 액션 개수가 많으면 계산 속도가 급감한다.**

Sarsa 보다 low variance라고 한다.



### Expected Sarsa vs Sarsa

Expected Sarsa는 value function을 좀 더 일반적으로 고려하므로(episode 방향만 고려하는게 아니니까) 좀 더 안정적인 업데이트가 가능하다고 한다. Sarsa는 어찌됬든 많은 episode를 수행하다보면 value function의 추정이 정확해진다. 하지만, 각 episode에서 잘못된 액션이 있다할지라도, 그 방향으로 업데이트를 수행한다. 하지만, expected Sarsa는 잘못된 액션이든, 올바른 액션이든, 무조건 기댓값을 취하므로, 항상 안정적인 업데이트가 가능하다.

Expected Sarsa에서는 큰 step size $\alpha$를 사용하기 쉽다. Sarsa에서는 $\alpha$가 크면 잘못된 방향으로도 큰 업데이트를 수행하겠지만, expected Sarsa에서는 그 정도가 작다. 기댓값으로 업데이트하기 때문. 이것은 심지어 optimal value function으로의 수렴 속도까지 expected Sarsa가 뛰어나게 만들기도 한다. (운이 좋다면, Sarsa가 빠를수도 있다)

또한, value function이 거의 다 수렴한 상태에서도, 큰 $\alpha$를 가진다면, Sarsa는 샘플링하는 대로 업데이트를 똑같이 큰 step으로 지속하게 된다. 어쩌다보면 발산 방향으로 갈 수도 있다. 반면, expected Sarsa에서는 샘플링이 계속되도, 기댓값은 크게 변하지 않으므로, value function또한 안정적으로 유지된다.



### Expected Sarsa vs Q-Learning

**Expected Sarsa는 Sarsa와 다르게 on-policy와 off-policy learning 둘 다에 해당한다.**

Expected Sarsa의 식은 다음과 같다.

$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha (R_{t+1} + \gamma \cdot \sum_{a'} \pi(a'|S_{t+1})Q(S_{t+1}, a') - Q(S_t, A_t))
$$

이때, 미래의 action value를 계산할 때, target policy에 대한 기댓값을 계산하게 되는데, 이는, 액션 $a'$이 behavior policy에서 나온 것이라도 동일하다. 즉, 자연스럽게 위 value function은 target policy를 따르는 기댓값을 이용하게 되며, importance sampling 없이 off-policy learning을 달성한다.

**Expected Sarsa는 Q-learning의 일반화 버전이다.**

만약, target policy가 greedy한 policy라면, 위 식은 Q-learning과 똑같아진다. 즉, expected Sarsa 방식은 greedy한 target policy이던, greedy하지 않은 target policy이던 적용할 수 있는 off-policy learning 알고리즘이다.