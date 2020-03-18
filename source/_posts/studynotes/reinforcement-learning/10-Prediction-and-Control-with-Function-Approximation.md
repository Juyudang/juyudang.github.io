---
title: 10. Prediction and Control with Function Approximation
toc: true
date: 2020-03-18 09:00:08
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Prediction and Control with Function Approximation



참고: Coursera Reinforcement Learning (Alberta Univ.)

지금까지, state와 action이 discrete하며, (state, action) pair의 value가 deterministic한, tabular method를 보았다. 하지만, 이건 실생활에서 매우 한정적일 수 밖에 없다.

Value function을 table로 표현하지 말고, function으로 추정하자는게 지금부터 다룰 내용이다.
$$
V(s) = f_W(s)
$$


Function은 어떤 파라미터 $W$로 parameterized되어 있으며, function은 linear 형태의 function이나, 인공신경망과 같이 non-linear한 형태도 가능하다.



## Tabular Method is a Linear Function

Tabular method는 linear function approximation의 한 방법이다.

![image-20200224100627139](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200224100627139.png)

state의 개수만큼 feature가 있고, feature는 각 state를 나타내는 indicator가 된다. 그러면, 각 state의 value는 그 state의 weight가 된다.
$$
V(s_i) = \begin{pmatrix}
0 \\
\cdots \\
0 \\
1 \\
0 \\
\cdots \\
0
\end{pmatrix} \cdot
\begin{pmatrix}
w_1 \\
\cdots \\
w_{i-1} \\
w_i \\
w_{i+1} \\
\cdots \\
w_{16}
\end{pmatrix} = w_i
$$
즉, tabular method는 위와 같이 indicator feature를 이용해서 linear function으로 표현이 가능하다. 따라서, tabular method는 linear function approximation의 한 instance이다.



## Generalization and Discrimination

Generalization과 discrminiation은 reinforcement learning에서 상당히 중요하다.

- Generalization

  Generalizaiton은 어떤 state를 학습(value를 추정)했다면, 비슷한 다른 state까지 영향을 미쳐서 학습되는 것을 의미한다.

- Discrimination

  Discrimination은 서로 다른 state끼리는 학습 또는 추정시, 영향을 미치지 않아야 함을 의미한다. 즉, 어떤 state끼리는 독립적으로 value가 추정되어야 한다는 것이다.



Tabular method는 generalization을 전혀 하지 못하고, discrimination을 100% 수행하는 방법이다. 반면, 모든 state를 똑같은 value를 두도록 설정하면, discrimination을 전혀 수행하지 못하고 generalization을 100% 수행하게 된다. RL에서는 generalization과 discrimination을 동시에 높여야 하며, trade-off관계라서, 적절히 조정하는게 필요하다.



## Value Estimation as Supervised Learning

True reward 리턴값이 있다면, reward를 target label로 삼아서 $f_W(s)$를 학습할 수 있지 않을까.

![image-20200225104841408](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200225104841408.png)

하지만, reinforcement learning에서는 각 데이터 샘플(하나의 state-action pair)이 서로 correlate되어 iid에 위반되므로, 모든 supervised learning technique이 다 잘 적용될 수 있는 것은 아니다. 또한, TD의 경우, target 자체가 estimation값(다음 state의 value는 또 다른 estimation 값임)이므로, target이 학습이 지속되면서 변한다. 이는, supervised learning이랑 상당히 다른 환경이다.



## The Objective for On-policy Prediction

Function approximation을 하기 위해서는, target value와 얼만큼 가까운지 판단할 수 있는 objective function이 필요하다. 



### The Value Error Objective

다음은 Mean squared error를 이용한 objective function이다.
$$
\text{VE} = \sum_s \mu(s)[V_{\pi}(s) - \hat{v}(s, W)]^2
$$
이때, 각 state마다 서로 다른 가중치 $\mu(s)$를 주어서, 상대적으로 중요한 state에게는 높은 가중치를, 덜 중요한 state에게는 낮은 가중치를 주도록 한다. Policy에 의해 자주 방문하는 state에 대해서는 높은 가중치를 줄 수도 있겠다.



### Gradient Monte Carlo for Policy Evaluation

Gradient descent를 Monte Carlo RL에 맞게 수정한 것. Stochastic gradient descent.

Value error식은 때때로 state개수가 너무 많아서 계산이 불가능하다. 대신, gradient를 approximation한다. 원래 Gradient descent식은 다음과 같다. 이때, $x(s)$는 state $s$의 feature vector이다.
$$
W \leftarrow W + \alpha \sum_s \mu(s)[V_{\pi}(s) - \hat{v}(s,W)] \nabla \hat{v}(s, W)
$$
그런데, 이 식을 쓰지 말고, 다음처럼 gradient를 approximate해서 쓰자는 것이 된다.
$$
W \leftarrow W + \alpha [V_{\pi}(S_t) - \hat{v}(S_t, W)] \nabla \hat{v}(S_t, W)
$$
왜냐하면 다음이 성립하기 떄문. 즉, 위 gradient는 원래 gradient의 추정치라고 볼 수 있다.
$$
E[V_{\pi}(S_t) - \hat{v}(S_t,W)] = \sum_s \mu(s)[V_{\pi}(s) - \hat{v}(s,W)]
$$
이는, 샘플 하나 (state-action pair 1개)씩 보면서 한 번 업데이트하는 stochastic gradient descent 방식이다.

하지만, 이 경우는 target value function인 $V_{\pi}(s)$를 알아야 한다. 얘네도 $G_t$를 이용해 approximation한다.
$$
W \leftarrow W + \alpha [G_t - \hat{v}(S_t,W)] \nabla \hat{v}(S_t,W)
$$
역시 다음을 추정한 것이다.
$$
E[V(S_t)] = E[G_t|S_t]
$$
다음은 pseudo code.

![image-20200225135449599](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200225135449599.png)



### State Aggregation

State개수가 너무 많아서 모든 state를 따로 업데이트하기 힘들 때, 비슷한 state끼리는 묶어서 하나의 state로 보는 것을 말한다. 따라서, 하나의 state가 업데이트되면 같은 그룹의 다른 state도 같은 값으로 업데이트된다.



## The Objective for TD

On-policy learning인 Gradient Monte Carlo의 objective로 squared error를 사용했었다. 하지만, TD는 $G_t$가 다음의 특성을 가진다.
$$
v(S_t, W) = R_{t+1} + \gamma v(S_{t+1}, W)
$$
($G_t = \hat{v}(S_t, A_t)$이다. ) 즉, 다음의 gradient 수식에서,
$$
(G_t - \hat{v}(S_{t}, W)) \nabla \hat{v}(S_{t}, W)
$$
$G_t$부분은 실제 value의 추정값이므로, TD learning에서는 $v(S_t,W)$와 대체해야 한다. 그런데, 이놈은 $v(S_{t+1}, W)$를 참조하고 있으며, 이 $v(S_{t+1}, W)$는 true value의 estimation이기 보단 현재의 value estimation이다. 따라서, biased된 추정값이며, $v(S_t, W)$를 미분해도 이 식이 $W$를 가지고 있으므로, Gradient Monte Carlo의 gradient 수식과 같게 나오지 않는다. 하지만, 그냥 $v(S_t, W)$는 상수처럼 간주해버리고 쓰게 되는데(즉, $W$에 대한 함수가 아니라고 간주), 이를 semi-gradient 방법이라고 부른다.

최종적으로 $W$의 업데이트 식은 다음과 같이 쓴다.
$$
W \leftarrow W + \alpha(R_{t+1} + \gamma v(S_{t+1}, W) - \hat{v}(S_t,W)) \nabla \hat{v}(S_t, W)
$$



## TD vs Monte Carlo

Function approximation에서, TD와 Monte Carlo 방식의 차이점은 다음과 같다.



**TD**

- 장점
  - 에피소드가 끝나기 전에 바로바로 학습하므로 빠른 학습 속도(loss가 빠르게 줄어듬)
- 단점
  - Estimation이 최종 reward를 반영하지 않은, 현재의 value estimation을 true estimation으로 삼기 때문에 biased된 학습. 즉, 부정확할 수 있다. Local minimum의 근처까지밖에 못갈 수 있다.



**Monte Carlo**

- 장점
  - 에피소드의 최종 reward를 반영한 true value의 estimation을 사용하기에 TD보단 unbiased된 학습. 즉, local minimum을 다소 정확하게 찾는다.
- 단점
  - 느리다. step size를 작게 줘야 한다.



## Linear TD

Value function을 linear하게 모델링한 것을말하며, 간단하고 쉽지만, 잘 정제된 feature가 있다면, 매우 강력한 성능을 발휘한다.

Tabular TD(0)는 linear TD의 한 종류인데, 다음처럼 feature가 생겼다고 가정한다면, 완벽히 linear TD이다.
$$
w = \begin{pmatrix}
w_0 \newline 
w_1 \newline 
w_2 \newline 
w_3 \newline 
\cdots \newline 
w_d
\end{pmatrix},
x(s_i) = \begin{pmatrix}
0 \newline 
0 \newline 
1 \newline 
0 \newline 
\cdots \newline 
0
\end{pmatrix},
\hat{v}(s,w) = w \cdot x
$$
feature는 어떤 state인지 나타내는 indicator이고, weight가 각각 상응하는 state들의 value가 되는 셈. Feature $x$를 어떤 aggregation인지를 나타낸다고 하면, aggregation tabular TD(0)역시 linear TD의 모양이 되므로, aggregation tabular TD(0)역시, linear TD의 한 종류이다.

만약, squared error를 사용하는 linear TD라면, 다음 식으로 $$w$$가 업데이트된다.
$$
w \leftarrow w + \alpha (R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)) X(S_t)
$$
