# Swiss Roll 데모 실행

> **범위:** 2D Swiss Roll 토이 데모([scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py)) 실행·평가지표·시각화. 이 데모는 MNIST 모델과 **무관한 독립 구현**이며 논문에 더 충실하다(이유는 [explanation/diffusion-method.md](../explanation/diffusion-method.md) §6).
> **대상:** 방법을 작은 데이터로 검증·시각화하려는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 실행

```bash
chmod +x run_swiss_roll.sh
./run_swiss_roll.sh
```

[run_swiss_roll.sh](../../run_swiss_roll.sh)가 [scripts/test_swiss_roll.py](../../scripts/test_swiss_roll.py) `main()`을 실행한다. 🟠 이 스크립트도 `CUDA_VISIBLE_DEVICES=2`를 하드코딩하므로 필요 시 덮어쓴다.

> 🟠 **기본 설정이 매우 무겁다:** `num_epochs=300000`, `batch_size=128000`(매 스텝 Swiss Roll 점을 새로 생성). 빠른 확인은 인자로 줄인다:
>
> ```bash
> python scripts/test_swiss_roll.py --num_epochs 5000 --eval_interval 1000 --device cpu
> ```

## 2. 핵심 구성요소

| 요소 | 심볼 | 역할 |
|---|---|---|
| 설정 | `DiffusionConfig` | dataclass, CLI 인자 매핑 |
| 네트워크 | `DiffusionEncoder` | 공유 head + 타임스텝별 분리 head(μ, log σ²) |
| 모델 | `DiffusionModel`(test_swiss_roll 내부) | sigmoid β 스케줄, 해석적 posterior, 역방향 |
| 손실 | `train_diffusion_model()` | posterior vs 예측의 **닫힌 형식 가우시안 KL** |

순방향 `forward_diffusion()`은 해석적 posterior $(\mu_q,\sigma_q)$를 반환하고, `reverse_diffusion()`은 예측 $(\mu_p,\sigma_p)$를 낸 뒤 KL을 최소화한다 — 즉 논문의 스텝별 변분 하한 항을 그대로 구현한다([explanation/diffusion-method.md](../explanation/diffusion-method.md) §5).

## 3. 평가지표

`evaluate_model_metrics()`가 `eval_interval` 스텝마다, 그리고 마지막에 세 지표를 계산해 TensorBoard `metrics/*`에 로깅한다:

| 지표 | 심볼 | 의미 |
|---|---|---|
| MMD | `calculate_mmd_distance()` | 실제 vs 생성 분포 차이(가우시안 커널 폭 0.1). **MMD 최저로 best 선택** |
| Manifold coverage | `calculate_manifold_coverage()` | 실제 점 중 가장 가까운 생성점이 0.1 이내인 비율 |
| Wasserstein | `scipy.stats.wasserstein_distance` | 평탄화 좌표 기반 1D 근사 거리 |

## 4. 시각화 & 산출물

`visualize_diffusion_steps()`가 순방향(t=0, T/2, T)·역방향 산점도를 그려 저장한다.

| 경로 | 내용 |
|---|---|
| `{save_dir}/diffusion_process_{step}.png` | 순/역방향 시각화 |
| `{save_dir}/{exp_name}_best.pth` | MMD 최저 체크포인트 |
| `{save_dir}/{exp_name}_final.pth` | 최종 체크포인트 |
| `{log_dir}/{exp_name}/` | TensorBoard 이벤트 |

> 🟠 `scipy`는 import되지만 `environment.yml`에 선언돼 있지 않다 — [explanation/known-issues.md](../explanation/known-issues.md) §2 D5.

## 관련 문서

- [explanation/diffusion-method.md](../explanation/diffusion-method.md) · [reference/data-model.md](../reference/data-model.md)
