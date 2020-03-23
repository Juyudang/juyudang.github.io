---
title: 13. Policy Gradient
toc: true
date: 2020-03-23 09:12:08
tags:
	- StudyNotes
	- ReinforcementLearning
categories:
	- Study Notes
	- Reinforcement Learning
---



# Policy Gradient



Policy gradient 방법은, policy를 parameterized function으로 모델링해서, state, action feature vector로부터 바로 policy를 추론하는 방법이다.

즉, state, action feature로부터 action value function을 추론하고 그로부터 policy를 계산하는 이전 방법에서, action value function을 거치지 않고 바로 policy를 추론하는 것이다. 

차이점은, 이전 방법에서는 action value function을 parameterize했지만, policy gradient에서는 policy를 parameterize한다. 이때, policy funciton의 parameter는 $\theta$로 표기한다.

![image-20200309091753523](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200309091753523.png)



## Softmax Policies

Policy는 반드시 확률분포여야 한다. 즉, 다음이 성립해야 한다.
$$
\sum_a \pi(a|s, \theta)=1 ~ \text{ for all } s
$$
즉, 각 state에서, action들의 확률들의 합은 1이어야 한다. 또한, 각 확률은 [0, 1] 범위에 있어야 한다.

Action value function에서는 linear function을 이용했지만, policy function은 위와 같은 이유로 linear function의 결과값을 그대로 이용할 수 없다. 그 대신, linear function의 결과값에 softmax 함수를 적용해 준다.
$$
\pi(a|s, \theta) = \text{softmax}(h(s, a, \theta)) = \frac{e^{h(s, a, \theta)}}{\sum_{a'} e^{h(s, a', \theta)}}
$$
이때, $h(s, a, \theta)$는 state $s$와 action $a$를 입력으로 받고 action preference를 출력하는 linear 함수(마지막 레이어가 linear이면 linear하다고 하자)이다.

이제, action preference는 linear해도 된다. action preference가 심지어 음수가 나와도 exponential에 의해 양수임이 보장되며, 분모의 normalization으로 인해 [0, 1]사이 값으로 유지됨이 보장되며, 합이 1임이 보장된다.



### Action Preference vs Action Value

Action preference는 action value와는 다르다. Action value는 미래 expected return의 합으로 이루어져 있으나, action preference는 그러한 것을 고려하지 않는다.

또한, action value는 가장 높은 값을 가지는 action만이 높은 확률을 갖고($1 - \epsilon$), 나머지 action은 $\epsilon/N_a$값을 가진다($N_a$는 액션 개수). 즉, 높은 action value 값을 가지는 action이외의 action은 모두 작은 확률로 같은 확률을 가진다.

![image-20200309092901785](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200309092901785.png)

하지만, action preference는 그 값이 크고 작음의 순서가 유지가 된다. Softmax 함수는 preference가 클수록 큰 확률을 가지게 하며, 작을수록 작은 확률을 가지게 만들어준다.

![image-20200309092952785](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200309092952785.png)



### Stochastic Policies

때로는 deterministic policy를 가지면 풀기가 불가능한 문제도 있다고 한다. Epsilon greedy방법(action value approximation에서의)은 deterministic policy를 결과로 내놓기에, 이때 매우 성능이 좋지 않다. Policy gradient는 반면, softmax함수로 인해 stochastic policy를 출력한다.

(Action value에 따라서 policy를 정할때도 epsilon greedy가 아닌 softmax를 씌우면 어떻게 될까)



## Selecting Function Approximation

때로는 action value를 parameterized function으로 모델링하는 것 보다 policy를 parameterized function으로 모델링하는게, 훨씬 간단할 때도 있고 반대의 경우도 존재한다. 따라서, 적절히 덜 복잡한 방향을 선택하면 될 듯 하다.



## Policy Gradient for Continuous Tasks

Episodic task에서는 다음을 최대화하는 액션을 선택하게 하는 policy를 찾는 것이다.
$$
G_t = \sum_{t=0}^T R_t
$$
Continuous task에서는 다음을 최대화하는 액션을 선택하게 하는 policy를 찾는 것이다.
$$
G_t = \sum_{t=0}^T \gamma^t R_t ~ (0 \leq \gamma < 1)
$$
(Continuous이기 때문에 discounting이 필요. 안그러면 $\infty$로 간다.)

Continuous task에서는 다음을 최대화하는 액션을 선택하게 할 수도 있다.
$$
G_t = \sum_{t=1}^T (R_t - r(\pi))
$$
이때, $R_t - r(\pi)$는 발산하지 않고 수렴하기에 discounting이 필요하지 않다.



어떤 policy가 최대 reward를 획득하게 한다는 것은 모든 state의 평균 reward를 의미하는 $r(\pi)$를 최대화하는 것과 같다. 그 policy가 궁극적으로 총 reward가 최대가 되도록 학습한다면, $r(\pi)$도 최대이기 때문.

$r(\pi)$는 다음과 같았다.
$$
r(\pi) = \sum_s \mu_{\pi}(s) \sum_a \pi(a|s, \theta) \sum_{s',r} p(s', r|s, a)r
$$

이 $r(\pi)$를 최대화하기 위해서, policy $\pi$에 대해 미분해보면,
$$
\nabla r(\pi) = \nabla \sum_s \mu_{\pi}(s) \sum_a \pi(a|s, \theta) \sum_{s', r} p(s', r|s, a)r
$$
인데, $\mu_{\pi}(s)$는 각 state의 방문 횟수로, policy에 의해 영향을 받는다. 하지만, $\mu_{\pi}(s)$는 stationary distribution으로, 추정하기 쉽지 않다.



### The Policy Gradient Theorem

The Policy Gradient Theorem이라는 이름으로, $\nabla r(\pi)$는 다음으로 추정할 수 있다고 증명되어 있다.
$$
\nabla r(\pi) = \sum_s \mu_{\pi}(s) \sum_a \nabla \pi(a|s,\theta) \cdot q_{\pi}(s,a)
$$
$\nabla \pi(a|s,\theta)$는 액션 $a$의 확률값이 높아지는 방향의 gradient이다. 이것을 상응하는 action value와 weighted sum하게 된다. 그러면 전체 average reward가 상승하는 방향일 것이라는 것이고, 이 과정을 모든 state에서 계산하고 모두 더해준다.

예를들어, 어떤 state에서 액션이 상, 하, 좌, 우 네가지가 있고, 상, 좌 방향으로는 action value가 낮다고 하자. 하지만, 하, 우 방향으로는 높다고 할 때, 상대적으로, 하, 우 방향의 action value가 높으므로, 하, 우 방향이 gradient와 action value의 곱이 상대적으로 비중을 차지하는 비율이 증가할 것이다. 그리고, policy는 이 gradient 방향으로 업데이트하게 되면, policy는 하, 우 의 확률 비중을 약간 늘릴 수 있을 것이다. 이런 과정을 전체 state에 반복하면서 전체 average gradient를 높일 수 있다는 것이다.



### Estimation of Policy Gradient

$\nabla r(\pi)$를 계산하는 대신 추정하고자 하는데, $\nabla r(\pi)$는 다음과 같았다.
$$
\nabla r(\pi) = \sum_s \mu_{\pi}(s) \sum_a \nabla \pi(a|s,\theta) q(s,a)
$$
그런데, 이것은 다음처럼 $\mu$에 대한 기댓값으로 표현이 가능하다.
$$
\nabla r(\pi) = E_{\mu}[\sum_a \nabla \pi(a|s,\theta) q(s, a)]
$$
참고로, $E[X]$는 stochastic sample인 $x_i \sim X$ 여러개로 추정이 가능하다. 따라서, $\nabla r(\pi)$는 다음처럼 추정할 수 있다.
$$
\nabla r(\pi) = \sum_a \nabla \pi(a|S_t,\theta) q(S_t, a)
$$
이때, $S_t$는 agent가 environment와 상호작용하면서 얻은 샘플 또는 경험이다. 하지만, action에 대한 summation이 남아 있다. 이것은 계산이 가능하지만, 이것 또한 stochastic sample을 이용해서 추정이 가능하다.

다음처럼 $\pi$에 대한 기댓값이 되도록 식을 수정한다.
$$
\nabla r(\pi) = \sum_a \pi(a|S_t, \theta) \frac{1}{\pi(a|S_t,\theta)} \nabla \pi(a|S_t, \theta) q(S_t, a)
$$

$$
\nabla r(\pi) = E_{\pi} [\frac{\nabla \pi(a|S_t, \theta)}{\pi(a|S_t,\theta)}q(S_t, a)]
$$

이것은 다음으로 추정이 가능할 것이다(stochastic sample).
$$
\nabla r(\pi) = \frac{\nabla \pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)} q(S_t, A_t)
$$
최종적으로, policy gradient ascent는 다음과 같다.
$$
\theta_{t+1} \leftarrow \theta_t + \alpha \cdot \nabla r(\pi)
$$

$$
\theta_{t+1} \leftarrow \theta_t + \alpha \cdot \frac{\nabla \pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)} q(S_t, A_t)
$$

($\nabla r(\pi)$는 $r(\pi)$가 증가하는 방향이므로 +를 한다.)

이것은 또 다음처럼 간단하게 쓸 수 있다.
$$
\theta_{t+1} \leftarrow \theta_t + \alpha \cdot \nabla ~ \text{ln}[\pi(A_t|S_t, \theta) ]q(S_t, A_t)
$$
$\pi$는 parameterized function이라 계산이 가능하고, $q(S_t,A_t)$는 TD같은 것으로 추정할 수 있다(Value function도 여전히 추정해야 한다!).
$$
\theta_{t+1} \leftarrow \theta_t + \alpha \cdot \nabla \text{ln}[\pi(A_t|S_t,\theta)](R_{t+1} - r(\pi) + \hat{v}(S_{t+1}))
$$
Action value는 액션 $A_t$을 취했을 때 얻은 reward $R_{t+1}$와 그 후의 value, 즉 state value $v(S_{t+1})$을 합한 것과 같기 때문에 위 처럼 된다. 이 경우, action value function은 단순히 reward의 합이 아닌, differential reward의 합을 추정한 놈이므로 $q(S_t, A_t) = R_{t+1} - r(\pi) + v(S_{t+1},W)$이 된다.



## Actor-Critic Algorithm

Actor가 critic의 피드백을 받고 본인의 policy를 수정하면서 발전해가는 형식이라고 한다.

- Actor

  Parameterized policy function을 말하며, 어떤 행동을 하는 객체라고 해서 이렇게 이름을 붙인 듯.

- Critic

  Parameterized state value function을 말하며, actor가 다음에 action을 취할지, 즉, policy를 어떻게 업데이트할지에 대해, action value $q(S_t, A_t)$값으로 피드백을 주고 actor의 다음 행동에 영향을 미치게 한다.

Agent가 environment와 상호작용하면서 얻은 샘플(또는 경험)들을 이용해서 actor와 critic을 동시에 업데이트시키게 된다.

그 전에, 업데이트의 편의를 위해 policy의 파라미터인 $\theta$의 업데이트식에 action value baseline을 추가한다.
$$
\theta_{t+1} \leftarrow \theta_t + \alpha \cdot \nabla \text{ln}[\pi(A_t|S_t,\theta)](R_{t+1} - r(\pi) + \hat{v}(S_{t+1},W)- \hat{v}(S_t,W))
$$
이때, $\hat{v}(S_t,W)$가 action value의 baseline이며, 이것을 추가하는 것은 $\theta$의 업데이트 방향에 영향을 전혀 미치지 않는다. 왜냐하면, 위 식을 다시 기댓값 식으로 바꿔보면,
$$
\nabla r(\pi) = \nabla \text{ln}[\pi(A_t|S_t, \theta)](R_{t+1} - r(\pi) + \hat{v}(S_{t+1},W))
$$

$$
\nabla r(\pi) = E_{\pi} [\nabla \text{ln}[\pi(a|S_t, \theta)](R_{t+1} - r(\pi) + \hat{v}(S_{t+1},W))]
$$

그런데 여기서, 다음을 더해주는데, 아래 기댓값은 0이기 때문에 $\nabla r(\pi)$에 영향을 미치지 않는다.
$$
E_{\pi}[-\nabla \text{ln} [\pi(a|S_t,\theta)]\hat{v}(S_t,W)]
$$
따라서, 다음과 같이 된다.
$$
\nabla r(\pi) = \nabla \text{ln}[\pi(A_t|S_t, \theta)](R_{t+1} - r(\pi) + \hat{v}(S_{t+1},W) - \hat{v}(S_t,W))
$$
이 식이 Actor-Critic 알고리즘에서 사용될 policy gradient 식인데, baseline의 기댓값이 0인 이유는 다음과 같다.
$$
E_{\pi}[-\nabla \text{ln} [\pi(a|S_t,\theta)] \hat{v}(S_t, W)]
$$

$$
= -\sum_a \pi(a|S_t,\theta)\nabla \text{ln} [\pi(a|S_t,\theta)] \hat{v}(S_t, W)
$$

$$
= -\sum_a \nabla \pi(a|S_t,\theta)\nabla \hat{v}(S_t, W)
$$

이때, $S_t$는 이미 주어져 있다($S_t$가 주어진 후, action/policy에 대한 기댓값이니까). 따라서, $\hat{v}$은 action에 의해 영향을 받지 않는 값이며, 밖으로 나올 수 있다.
$$
= -\hat{v}(S_t, W) \sum_a \nabla \pi(a|S_t,\theta)
$$
그리고, gradient의 합은 합의 gradient이므로 다음과 같다.
$$
= -\hat{v}(S_t, W) \nabla \sum_a \pi(a|S_t,\theta)
$$
근데, 이때, $\pi$는 확률분포이고, 그들의 합은 1이다. 따라서,
$$
= -\hat{v}(S_t,W) \nabla 1 = 0
$$
어쨌든 최종적으로, policy gradient식은 다음과 같다.
$$
\nabla r(\pi) = \nabla \text{ln}[\pi(A_t|S_t, \theta)](R_{t+1} - r(\pi) + \hat{v}(S_{t+1},W) - \hat{v}(S_t,W))
$$

$$
\nabla r(\pi) = \nabla \text{ln}[\pi(A_t|S_t, \theta)] \delta_t
$$

이때, $\delta$는 TD error이다(TD error로 만들어주기 위해 baseline을 넣은 것이다).

최종적으로, Actor-Critic 알고리즘의 pseudo code는 다음과 같다.

![image-20200310110319075](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200310110319075.png)

(Learning rate는 actor, critic, 그리고 average reward 평균비율 전부 따로 줄 수 있다)



State value function의 feature는 state로만 구성되지만, policy의 feature는 state와 action 모두로 구성된다. Policy의 feature는 state feature를 action 개수만큼 stack한 feature라고 가정해보자.

다음 그림에서 state feature는 $x_0(s), x_1(s), x_2(s), x_3(s)$ 4개로 구성되어 있고, state-action pair feature 개수는 state feature 4개를 3번 복사한, 총 12개로 구성되는 것이다.

![image-20200311104559800](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200311104559800.png)

만약, state value function의 구현으로 linear parameterized function을 사용한다면, 그의 파라미터 $w$의 업데이트 규칙은 다음처럼 될 것이다.
$$
w \leftarrow w + \alpha \cdot \delta \cdot x(s)
$$
또한, policy function을 softmax로 구현했다면, 그의 파라미터 $\theta$의 업데이트는 다음처럼 될 것이다.
$$
\theta \leftarrow \theta + \alpha \cdot \delta \cdot (x_h(s,a) - \sum_b \pi(b|s,\theta) x_h(s,b))
$$

이때, $x(s)$는 4개로 구성된 state feature이고, $x_h(s,a)$는 12개로 구성된 state-action pair feature인데, action $a$에 해당하는 4개의 feature만 가저온 것이다.

($\nabla \text{ln} \pi(a|s,\theta) = x_h(s,a) - \sum_b \pi(b|s,\theta) x_h(s,b)$이기 때문; $\nabla h(s,a,\theta) = x_h(s,a)$)



## Actor-Critic for Continuous Actions

Action 개수만큼 state feature를 stacking하는 것도 가능한 action 집합이 discrete할때만 유효한 것이다. 만약, action 집합이 continuous하다면, critic을 softmax로 모델링 할 수 없다.

이런 경우에는, critic을 각 state에 따른 action을 distribution으로 모델링하는 것이다. 예를 들어, state에 따른 action 분포를 gaussian distribution으로 모델링한다고 가정한다.

![image-20200311114613408](https://raw.githubusercontent.com/wayexists02/my-study-note/image/typora/image/image-20200311114613408.png)

이때, critic은 이 gaussian distribution의 파라미터 $\mu, \sigma$를 모델링하게 된다. 즉, $\theta = \{ \theta_{\mu}, \theta_{\sigma} \}$가 된다.

때로는 discrete한 액션 집합이더라도, 그 수가 많고 촘촘하다면, continuous하게 취급하는 것도 도움이 된다. Distribution으로 모델링할 때의 장점은 distribution이기 때문에, 하나의 action에 대한 업데이트도 범위의 액션에 영향을 미치기 때문에 action generalization이 실현된다.