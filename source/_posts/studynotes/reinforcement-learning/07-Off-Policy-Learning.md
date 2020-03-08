---
title: 07. Off-policy Learning
toc: true
date: 2020-03-03 10:00:06
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Off Policy Learning



참고: Coursera Reinforcement Learning (Alberta Univ.)

학습 데이터를 수집할때, agent가 action을 취할 policy와 학습과정에서 업데이트할 policy를 따로 구분해 놓은 경우를 off-policy learning이라고 부른다. 반대로, 그냥 하나의 policy를 두고, agent는 그 policy에 따른 action을 취하고 업데이트도 그 policy를 업데이트하는 방식을 on-policy learning이라고 부른다.



### On-policy Learning

- Policy

  오직 하나의 policy만 존재하며, agent가 학습 데이터를 모으기 위한 action을 취할 때, 이 policy에 따라 액션을 선택한다. 또한, policy control에서도 이 policy를 직접 수정하게 되고, 다음 학습 데이터를 모을 때 영향을받게 된다.



### Off-policy Learning

- Target Policy

  학습 데이터를 모을 때는 이 policy에 따라 action을 취하지 않는다. 다만, 업데이트 할때는 이 target policy를 업데이트한다. 테스트에서는 target policy에 따라 agent가 action을 취하게 된다.

- Behavior Policy

  Agent는 학습 데이터를 모을 때는 이 policy에 따라 action을 취한다. 업데이트되지 않아서, 학습이 진행되도 최적 action만 따라가는 현상을 없앨 수 있다.

왜 policy를 분리했냐면, 결국 exploration-exploitation dilema때문이다. Policy를 하나만 두면, 학습이 진행될수록 다소 안좋은(?) state는 방문횟수가 급격히 줄고, agent는 평소 하던 action만 취한다. 하지만, 학습 데이터를 수집할 때는, 적절히 exploration을 해야 할 필요가 있다. Behavior policy는 random uniform policy와 같은, 다소 불확실하지만, 모든 state가 고르게 방문될 수 있도록 설정하게 된다.

Off-policy의 장점은, target policy를 stochastic하게 둘 필요가 없이 deterministic하게 둬도 된다는 것이다. Target policy가 $\epsilon$-soft일필요가 없다.



## Importance Sampling

그런데, Off-policy learning에 문제가 있다. 만약, Monte Carlo estimation을 시행한다고 하면, 많은 샘플을 모으고 그 샘플들의 단순 평균을 계산하여 기댓값을 계산한다. 샘플 $(S_t, A_t, R_t)$을 모을 때, agent는 behavior policy에 따라 action을 취하고 샘플을 얻게 된다. 따라서, 그 샘플들의 평균은 behavior policy $b$에 대한 기댓값 $E_b[G]$가 되지, $E_{\pi}[G]$가 되지 않는다.

따라서, policy evaluation을 할 때, $b$에서 샘플링한 샘플로 $\pi$에 대한 기댓값을 추정하도록 해야 한다.

**Importance sampling**이란, Monte Carlo estimation을 할 때, **다른 분포**에서 샘플링한 샘플을 이용해서 분포의 기댓값을 추정하는 것을 말한다. 분포 $b$를 따르는 random variable $X$와 그 샘플 $x$에 대해, 분포 $b$에 대한 $X$의 기댓값은 다음처럼 계산할 수 있다.
$$
x_i \sim b
$$
$$
E_b[X] \approx \sum_{i=1}^n x_i \cdot b(x_i)
$$

그리고, 여기서 약간의 수정을 가해서 다른 분포 $\pi$에 대한 $X$의 기댓값을 계산하도록 할 수 있다.
$$
E_{\pi}[X] = \sum_X X \cdot \pi(X)
$$
$$
= \sum_X X \cdot \pi(X) \cdot \frac{b(X)}{b(X)}
$$

$$
= \sum_X X \cdot b(X) \cdot \frac{\pi(X)}{b(X)}
$$

$$
= \sum_X X \cdot \rho(X) \cdot b(x), ~ (\rho(X) = \frac{\pi(X)}{b(X)})
$$

$$
= E_b[X\rho(X)] \approx \frac{1}{n}\sum_{i=1}^n x_i \cdot \rho(x_i)
$$

따라서,
$$
E_{\pi}[X] \approx \frac{1}{n} \sum_{i=1}^n x_i \cdot \rho(x_i)
$$
이때, $\rho(x)$를 importance sampling ratio라고 하며, 한 샘플에 대해, 두 분포간의 확률 비율을 말한다.

Policy iteration에서, 특정 시점 $t$에서의 policy $\pi_t$가 있고, 그 policy $\pi_t$를 이용해 value function을 계산해야 하는데, 이 과정에서 Monte Carlo estimation이 사용된다. 여기서, behavior policy를 통해 얻은 샘플들을 이용해서 $\pi_t$에 대한 기댓값을 계산하게 된다.



## Implementation of Off-Policy Learning

Behavior policy로 episode를 만들고, 각 타임에서의 state와 action을 얻고, 다음 식을 통해 value-function을 Monte Carlo estimation하게 되면, policy $b$에 대한 기댓값 추정이 된다.
$$
G_{t} \leftarrow \gamma \cdot G_{t+1} + R_t
$$
$$
G_t \text{ appends to } Returns(S_t)
$$

$$
V(S_t) \leftarrow \text{mean}(Returns(S_t))
$$

여기서, $R_t$는 environment dynamic distribution으로부터 나왔으니, importance sampling에서 예외로 하고, $G_{t+1}$은 behavior policy $b$에 대한 value 기댓값일 것이다. 따라서, $G_{t+1}$에 importance ratio를 곱해주어야 한다.
$$
G_t \leftarrow \gamma \cdot W \cdot  G_{t+1} + R_{t}
$$
$$
G_t \text{ appends to } Returns(S_t)
$$

$$
V(S_t) \leftarrow \text{means}(Returns(S_t))
$$

그럼 이제 importance ratio를 계산해야 하는데, $G_{t+1}$은 바로 다음 시점 $t+1$로부터, policy $b$에 대한 기댓값이므로, importance ratio는 $t+1$시점부터 episode의 끝 $T$까지 분포 $b$로부터 $\pi$로 바꿔주는 역할을 해 주어야 한다. 즉, $\rho_t$ 대신, $\rho_{t+1:T}$를 계산해야 한다는 의미이다.
$$
\rho_{t+1:T} = \prod_{i=t+1}^T \frac{\pi(A_i|S_i)p(S_{i+1},R_{i+1}|S_i,A_i)}{b(A_i|S_i)p(S_{i+1}, R_{i+1}|S_i, A_i)}
$$
$$
= \prod_{i=t+1}^T \frac{\pi(A_i|S_i)}{b(A_i|S_i)}
$$

$$
= \prod_{i=t+1}^T \rho_i
$$



그리고 이것은 보다시피 incremental implementation이 가능하다.
$$
\rho_{T:T} = \rho_T
$$
$$
\rho_{T-1:T} = \rho_{T-1}\rho_T  =  \rho_{T-1} \cdot \rho_{T:T}
$$

$$
\rho_{T-2:T} = \rho_{T-2}\rho_{T-1}\rho_T = \rho_{T-2} \cdot \rho_{T-1:T}
$$



Monte Carlo prediction은 $T$로부터 backward 방식으로 이루어지므로, 위와 같이 incremental하게 구현이 가능하다. 이는 다음 pseudo code에서 $W$를 업데이트 하는 방식에서 드러난다.

다음은 pseudo-code.

![image-20200129145429486](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200129145429486.png)

On-policy Monte Carlo로부터, episode를 하나 플레이할때, policy $\pi$를 따르지 않고, policy $b$를 따른다는게 특징. 또한, importance sampling을 위해 $G \leftarrow \gamma G + R_{t+1}$에서 $G \leftarrow \gamma W G + R_{t+1}$로 변경되었다. $W$는 importance ratio를 계산한 것이다.