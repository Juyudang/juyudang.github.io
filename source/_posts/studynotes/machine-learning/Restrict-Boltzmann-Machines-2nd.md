---
title: Restrict Boltzmann Machines 2
toc: true
date: 2020-03-03 22:28:53
tags:
	- StudyNotes
	- MachineLearning
categories:
	- Study Notes
	- Machine Learning

---



# Restrict Boltzmann Machine



이번에는 RBM의 학습, 즉, weight를 수정하는 방법을 알아보고자 한다. RBM의 학습 역시 MLE 방식을 이용하게 된다. 즉, 데이터에 대한 likelihood를 최대화하게 된다.
$$
\hat{W}, \hat{b}, \hat{c} = \underset{W,b,c}{ \text{argmax} } ~ p(v)
$$
데이터는 visible unit에 들어가게 되므로, $$p(v)$$를 최대화하게 된다.

그런데 왜 $$p(v)$$를 최대화하는게 RBM의 학습일까. 그 이유는, 현재 데이터가 얻어진 이유는 상대적으로 높은 확률을 가지기 때문에 얻어진 것일 것이고, 왠만하면 불안정 상태보다 안정 상태의 데이터일 가능성이 높다.

그런데, $$p(v)$$는 intractable하다.
$$
p(v) = \sum_h p(v,h) = \sum_h \frac{1}{Z} \text{exp} \{ \sum_i \sum_j W_{i,j} v_i h_j + \sum_i b_i v_i + \sum_j c_j h_j \}
$$
여기서, normalizaiton constant는 모델 크기가 매우 작지 않는 이상 intractable하다.



## Free Energy

일단 미분을 위해 다음을 정의한다.
$$
F(v) = -\text{log} \sum_h e^{-E(v,h)}
$$
$F(v)$를 Free energy라고 정의한다. 특별한 의미가 있는것은 아니라고 한다. 편의상 정의하는 것 같다.

Free energy식을 조금만 변형한다.
$$
e^{-F(v)} = \sum_h e^{-E(v, h)}
$$
Free energy를 이용해서 $p(v)$를 $F(v)$에 대한 식으로 정리해보면,
$$
p(v) = \sum_h p(v,h) = \sum_h \frac{1}{Z} e^{-E(v,h)} \\
p(v) = \frac{1}{Z} e^{-F(v)}
$$
따라서, normalization constant는 다음과 같다고도 할 수 있다.
$$
Z = \sum_{v'} e^{-F(v')}
$$
이제, $\text{log} ~ p(v)$를 미분해보면,
$$
\frac{d \text{log} ~p(v)}{d\theta} = \frac{d}{d\theta}[\text{log} \frac{1}{Z}e^{-F(v)}]
$$
$$
= \frac{d}{d\theta} [\text{log} ~e^{-F(v)} - \text{log}~Z]
$$

$$
= \frac{d}{d\theta}[-F(v)] - \frac{d}{d\theta}[\text{log} Z]
$$

$$
= -\frac{dF(v)}{d\theta} - \frac{1}{Z} \frac{dZ}{d\theta}
$$

$$
= -\frac{dF(v)}{d\theta} - \frac{1}{Z} \frac{d}{d\theta}[\sum_{v'} e^{-F(v')}]
$$

$$
= -\frac{dF(v)}{d\theta} + \sum_{v'} \frac{1}{Z} e^{-F(v')} \frac{dF(v')}{d\theta}
$$

$$
= -\frac{dF(v)}{d\theta} + \sum_{v'} p(v') \frac{dF(v')}{d\theta}
$$

이때, 양변에 - 부호를 곱해주면서,
$$
-\frac{d \text{log} ~p(v)}{d\theta} = \frac{dF(v)}{d\theta} - \sum_{v'}p(v')\frac{dF(v')}{d\theta}
$$
그런데, 이때, 두번째 항은 $E[\frac{dF(v')}{d\theta}]$와 같다.
$$
-\frac{d\text{log} ~ p(v)}{d\theta} = \frac{dF(v)}{d\theta} - E[\frac{dF(v')}{d\theta}]
$$
이제 이것을 적분해보면, negative log likelihood를 얻을 수 있다. 이는 곧 cost와 같다.
$$
\mathbb{L} = F(v) - E[F(v')]
$$
이 loss는 두 개의 term으로 나뉘는데, 첫번째는 positive term이고, 두번째는 negative term인데, 다음과 같은 역할을 한다.

- Positive term

  Visible unit에 입력으로 들어간 데이터에 대한 에너지는 낮게 유도하는 효과가 있다.

- Negative term

  Visible unit에 입력으로 들어간 데이터 이외에 모든 조합들에 대해 에너지를 높게 유도한다.

사실상 대부분 EBM은 위와 같은 cost를 가진다고 한다. (또는 가저야만 한다고 한다)



## Contrastive Divergence

EBM에서, positive data와 negative data를 가지고, positive data의 에너지는 상대적으로 낮게, negative data의 에너지는 상대적으로 높게 학습시키는 방법을 말한다.

위 cost function을 보면, negative term은 intractable하다.  가능한 모든 visible unit 조합이 필요하기 때문. 따라서 기댓값을 추정해야 하는데, 기댓값은 Monte Carlo estimation처럼 샘플들의 평균으로 추정할 수 있다. 즉, negative term은 다음처럼 표현이 가능하다.
$$
E[F(v')] = \frac{1}{n} \sum_{i=1}^n F(v'_i)
$$
Visible unit을 샘플링해야 위 수식처럼 negative term을 추정할 수 있는데, visible unit을 샘플링해야 한다는 이야기가 된다. 이것은 RBM으로 샘플링이 가능하다.



### Gibbs Sampling

Gibbs sampling이라고 하면, Markov chain Monte Carlo의 일종이다. Markov chain Monte Carlo 방법은 어떤 random variable에 대해 모델링한 후, Markov chain을 통해 샘플을 생성하는데, 모델이 여러 random variable을 포함하는 경우, Gibbs sampling을 이용한다. Gibbs sampling은 하나의 random variable 이외에 다른 random variable이 모두 주어졌다(given)고 가정하고 샘플링하는 방식이다.

RBM은 다음과 같은 과정으로 샘플을 생성한다.

1. 주어진 데이터로 $p(h=1|v)$를 계산한다.
2. $p(h=1|v)$를 이용해서 $h$를 샘플링한다.
3. 샘플링한 $h$를 이용해서 $p(v'=1|h)$를 계산한다.
4. $v'$을 샘플링한다.

이렇게 하면, visible unit하나를 새로 샘플링한 것이다.

RBM의 구조는 Markov chain의 구조이며(visible unit은 바로 앞의 hidden unit들에게 의해서만 영향을 받음), 위 처럼 샘플링하는 것은 MCMC의 일종이다. 특히, Gibbs sampling의 일종이라고 하는데, 왜 Gibbs sampling인지는 잘 모르겠다. 여러 블로그들은 그저 "이것은 Gibbs sampling이다" 라고만 소개되어 있고, 이유를 소개한 블로그는 찾기 힘들다.



### CD-k (Contrastive Divergence with k Samples)

그럼 visible unit을 어떻게 샘플링해야 하는지는 결정되었고(RBM을 통한 MCMC 샘플링), 과연 몇 개의 샘플이 필요한가가 논의될 필요가 있다. MCMC를 통한 기댓값 추정은 MCMC의 이름에서 알수 있다시피, Monte Carlo 추정법을 이용하게 된다. 그리고, Monte Carlo 추정은 샘플이 많을수록 좋고 정확하다.

하지만, RBM에서는 단 하나의 샘플로만 해도 괜찮다고 한다. 즉, CD-1 방식을 이용한다. RBM의 학습은, 데이터셋에서 관찰되는 visible unit조합들에 대해서의 에너지는 높게, 그 이외의 조합들에 대해서는 에너지가 낮게 하는게 목적이다. CD-1 방식은 CD-k 중에서는 그 효과가 가장 떨어진다고 할지라도, 크게 문제가 되지 않는다고 한다.



### Fake Loss

RBM의 loss는 직접 계산할 수 없다. Negative term이 intractable하기 때문이다. 따라서, 이를 Monte Carlo 추정법으로 추정했다(1개의 샘플로). 즉, 다음처럼 loss가 수정될 수 있다.
$$
\text{fake-loss} = F(v) - F(v')
$$
**Fake loss는 공식적으로 붙은 이름은 아니다.**

진짜 loss는 아니지만, loss를 추정한 것이라서 이렇게 부르기로 한다.

진짜 loss는 다음 식이다.
$$
\text{loss} = F(v) - \mathbb{E}_{v'}[F(v')]
$$


### Intractability of Free Energy

그런데, free energy의 식을 보면 가능한 모든 hidden unit 조합에 대한 summation이 있다.
$$
F(v) = -\text{log} \sum_h e^{-E(v,h)}
$$
Hidden unit 조합을 모두 구한다는 것도 사실상 intractable하다. Free energy를 tractable하게 변환할 수 있다면, loss 함수를 계산할 수 있을 것이다.

Free energy 식에서, $E(v,h)$를 원래 에너지 식으로 대체한다.
$$
F(v) = - \text{log} \sum_h \text{exp}(\sum_i \sum_j W_{i,j}v_ih_j + \sum_i b_iv_i + \sum_j c_j h_j)
$$
그리고 다음처럼 $$h$$와 관계없는 항은 제일 밖으로 뺄 수 있다.
$$
F(v) = -\text{log} ~\text{exp}(\sum_i b_i v_i) \sum_h \text{exp}(\sum_i \sum_j W_{i,j}v_i h_j + \sum_j c_jh_j)
$$
$$
= -\sum_i b_iv_i - \text{log} \sum_h \text{exp} (\sum_i \sum_j W_{i,j} v_i h_j + \sum_j c_j h_j)
$$

$\text{exp}$안에 $\sum_j$가 공통으로 포함되어 있으므로 밖으로 뺀다.
$$
F(v) = -\sum_i b_i v_i - \text{log} \sum_h \prod_j \text{exp} (\sum_i W_{i,j} v_i h_j + c_j h_j)
$$
$h_j$도 빼보자.
$$
F(v) = - \sum_i b_i v_i - \text{log} \sum_h \prod_j \text{exp} \{h_j(\sum_iW_{i,j} v_i + c_j)\}
$$
그리고, $u_{j} = \sum_i W_{i,j} v_i + c_j$라고 해 보자(어차피 $i$방향으로는 summation이므로 변수가 아니다. 따라서 $j$로만 인덱싱한다).
$$
F(v) = -\sum_i b_i v_i - \text{log}\sum_h \prod_j \text{exp} \{h_j u_j\}
$$
$h_j$는 베르누이 변수이므로, $\{0, 1\}$중 하나의 값을 가진다. $\text{log}$안의 term은 가능한 모든 $h$의 조합을 더한 것을 의미하게 된다. 즉, hidden unit 수가 $M$개라면, $2^M$개의 경우를 모두 더하는 것이다.

근데, 만약, hidden unit 개수가 2개라고 해 보자(즉 $j=[0,1]$). 그럼 $\text{log}$안은 다음처럼 된다.
$$
e^{0u_0}e^{0u_1} + e^{0u_0}e^{1u_1} + e^{1u_0}e^{0u_1} + e^{1u_0}e^{1u_1}
$$
$$
= (e^{0u_0} + e^{1u_0})(e^{0u_1} + e^{1u_1})
$$

$j$번째 hidden unit $h_j$는 0과 1밖에 가지지 못하기에 위 4가지 경우가 전부.

그렇다면, hidden unit 개수가 $M=3$이라고 해 보자. 정리해보면 다음처럼 나올 것이다.
$$
(e^{0u_0} + e^{1u_0})(e^{0u_1} + e^{1u_1})(e^{0u_2} + e^{1u_2})
$$
즉,
$$
\sum_h \prod_j \text{exp} \{ h_j u_j \} = \prod_{j} \sum_{h_j=\{0,1\}} \text{exp} \{h_j u_j\}
$$
이며, 이것의 time complexity는 $M$이다. (인수분해를 활용한 계산 최적화!)

즉, free energy는 다음처럼 정리가 가능하다.
$$
F(v) = -\sum_i b_i v_i - \text{log} \prod_j \sum_{h_j=\{0,1\}} \text{exp}\{ h_j(\sum_i W_{i,j} v_i + c_j) \}
$$
$$
= -\sum_i b_i v_i - \sum_j \text{log} (1+ \text{exp} \{\sum_i W_{i,j} v_i + c_j\})
$$

이는, tractable하다.

