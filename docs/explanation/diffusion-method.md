# 확산 방법론 — 비평형 열역학

> **범위:** 확산모델의 개념, 비평형 열역학 틀, 순방향/역방향 과정과 변분 목적함수의 수식, 그리고 두 구현의 차이. 코드 구조는 [architecture.md](architecture.md), 결함은 [known-issues.md](known-issues.md).
> **대상:** 방법론을 이해하려는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17. 근거: Sohl-Dickstein et al. 2015(arXiv:1503.03585), Ho et al. 2020(arXiv:2006.11239).

## 1. 핵심 아이디어

샘플링·정규화가 쉬우면서 유연한 확률모델을 직접 쓰기는 어렵다. 이 방법은 모델을 마르코프 체인의 종점으로 **암묵적으로** 정의한다:

```
 x⁽⁰⁾ ──순방향 q (고정, 잡음 추가)──►  x⁽ᵀ⁾ ≈ N(0, I)
 x⁽⁰⁾ ◄──역방향 pθ (학습, 디노이즈)──  x⁽ᵀ⁾
```

스텝당 잡음이 작으면 각 역방향 조건부가 가우시안으로 잘 근사돼, 어려운 전역 모델링이 쉬운 국소 디노이징 여러 개로 분해된다. 이 2015 논문은 DDPM과 현대 확산 생성기의 직접 전신이다.

## 2. 비평형 열역학 틀

- 순방향 체인은 구조를 잡음으로 파괴해 시스템을 평형(가우시안 prior)으로 모는 **확산 과정**에 대응한다.
- 각 스텝이 분포를 조금만 바꾸면(작은 $\beta_t$) 느린 **준정적** 과정처럼 역방향이 다루기 쉬워진다.
- 학습량은 로그우도의 **변분 하한**이며 그 slack 항이 궤적을 따른 KL 합 형태 — 엔트로피 생성/자유에너지 한계와 구조적으로 닮았다.

## 3. 순방향 확산 과정

스텝별 커널(고정, 학습 파라미터 없음):
$$q(x^{(t)}\mid x^{(t-1)}) = \mathcal{N}\!\big(x^{(t)};\, \sqrt{1-\beta_t}\,x^{(t-1)},\, \beta_t I\big).$$

스케줄([networks/dpm.py](../../networks/dpm.py) `_initialize_diffusion_params()`): $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s\le t}\alpha_s$. 임의 $t$로의 닫힌 형식 점프([networks/dpm.py](../../networks/dpm.py) `forward_process()`):
$$x^{(t)} = \sqrt{\bar\alpha_t}\,x^{(0)} + \sqrt{1-\bar\alpha_t}\,\varepsilon,\quad \varepsilon\sim\mathcal N(0,I).$$
✅ 구현된 순방향 수식은 논문/DDPM과 일치한다.

## 4. 역방향 디노이징 과정

학습할 역방향 커널:
$$p_\theta(x^{(t-1)}\mid x^{(t)}) = \mathcal{N}\!\big(x^{(t-1)};\, \mu_\theta(x^{(t)},t),\, \sigma_\theta^2 I\big).$$

[networks/dpm.py](../../networks/dpm.py) `get_mu_sigma()`의 파라미터화:
$$\mu = \Big(\textstyle\sum_j c^{\mu}_j g_j(t)\Big)\sqrt{1/\alpha_t},\qquad
  \sigma = \mathrm{clamp}\!\big(\exp(\textstyle\sum_j c^{\sigma}_j g_j(t)),\, \sqrt{\beta_t},\, 1\big).$$

샘플러 `DiffusionModel.sample()`:

```
 x⁽ᵀ⁾ ~ N(0,I)
 for t in T-1 … min_t:          # 🔴 t=0 까지 가지 않음
     μ,σ = get_mu_sigma(x, t)
     x = μ + σ·ε
 return x                        # t=min_t 에서 멈춤 → 잔여 잡음
```

🔴 절단(`min_t`)과 `else: x=μ` 죽은 분기는 결함이다 — [known-issues.md](known-issues.md) §1 D1.

## 5. 변분 목적함수 & 학습 손실

논문의 하한(스텝별 KL 합):
$$\mathcal{L} = \mathbb{E}_q\!\Big[D_{\mathrm{KL}}\!\big(q(x^{(T)}\!\mid\! x^{0})\|p(x^{(T)})\big)
  + \sum_{t>1} D_{\mathrm{KL}}\!\big(q(x^{(t-1)}\!\mid\! x^{t},x^{0})\|p_\theta(x^{(t-1)}\!\mid\! x^{t})\big)
  - \log p_\theta(x^{0}\!\mid\! x^{1})\Big].$$

**MNIST가 실제 최적화하는 것**([networks/dpm.py](../../networks/dpm.py) `cost_single_t()`):
$$\mathcal{L}_\text{MNIST} = -\log\mathcal N\big(x^{(0)};\,\mu_\theta(x^{(t)},t),\,\sigma_\theta^2\big),\quad t\sim\mathcal U[min\_t,T).$$
🟠 즉 임의의 한 $t$에서 **$x^{(0)}$를 직접 예측**하는 NLL이며, 스텝별 KL도 DDPM의 ε-MSE도 아니다. 게다가 $\mu$는 $x^{(0)}$-예측기로 학습되는데 샘플러는 이를 역방향 평균으로 써서 **학습량과 샘플링량이 불일치**한다([known-issues.md](known-issues.md) §1 D2).

## 6. 두 구현의 차이 (MNIST vs Swiss Roll)

Swiss Roll([scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py))은 `forward_diffusion()`이 **해석적 posterior** $q(x^{(t-1)}\mid x^{(t)},x^{(0)})$를 반환하고, `train_diffusion_model()`이 닫힌 형식 가우시안 KL을 최소화한다:
$$\mathrm{KL} = \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q-\mu_p)^2}{2\sigma_p^2} - \tfrac12.$$
✅ 이는 논문의 스텝별 하한 항 그 자체다. 따라서 Swiss Roll 데모가 이 repo에서 논문에 가장 충실한 부분이고, MNIST 경로는 단순화/불일치를 가진다.

## 관련 문서

- [architecture.md](architecture.md) · [known-issues.md](known-issues.md) · [reference/data-model.md](../reference/data-model.md)
