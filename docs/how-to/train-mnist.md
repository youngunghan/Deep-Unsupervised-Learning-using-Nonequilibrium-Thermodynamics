# MNIST 학습하기

> **범위:** MNIST 이미지 모델([networks/dpm.py](../../networks/dpm.py) `DiffusionModel`)의 학습 실행·재개·하이퍼파라미터 변경. 방법론 설명은 [explanation/diffusion-method.md](../explanation/diffusion-method.md), 인자 명세는 [reference/configuration.md](../reference/configuration.md).
> **대상:** 모델을 학습/튜닝하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 기본 학습

```bash
./train.sh
```

[train.sh](../../train.sh)는 [scripts/train.py](../../scripts/train.py) `main()`에 인자를 넘긴다. 핵심 흐름:

```
train.sh ──> scripts/train.py main()
                ├─ batch_size = min(arg, 8)        # 메모리 상한(하드코딩)
                ├─ MNIST 로더 + Normalize([-1,1])
                ├─ DiffusionModel(...).to(device)
                ├─ Adam(lr=learning_rate)
                └─ scripts/trainer.py train_epoch()
                      └─ 배치마다 loss = model.cost_single_t(x); backward; step
                         val_interval 배치마다 검증·샘플·체크포인트
```

> 🟠 [scripts/train.py](../../scripts/train.py) `main()`은 `batch_size`를 **8로 강제 상한**한다(200-layer 백본의 메모리 때문). CLI/셸에서 더 큰 값을 줘도 8로 깎인다.

## 2. 하이퍼파라미터 변경

`train.sh`에 인자를 덧붙이면 그대로 전달된다(전체 표: [reference/configuration.md](../reference/configuration.md) §1):

```bash
./train.sh --batch_size 64 --learning_rate 1e-4 --epochs 20 --exp_name my_run
```

자주 바꾸는 값:

| 인자 | 기본 | 메모 |
|---|---|---|
| `--learning_rate` | `1e-5` | 기본값이 매우 낮다 — 5 epoch로는 수렴이 부족할 수 있음 |
| `--epochs` | `5` | |
| `--trajectory_length` | `1000` | 확산 스텝 수 $T$ |
| `--min_t` | `100` | 학습/샘플링 하한(아래 §4) |
| `--val_interval` | `10` | **배치 단위**(epoch 아님) |

## 3. 체크포인트에서 재개

```bash
./train.sh --continue_train --checkpoint_path checkpoints/my_run_epoch1_batch710.pth
```

[scripts/train.py](../../scripts/train.py) `main()`이 `model_state_dict`·`optimizer_state_dict`·`epoch`·`batch`를 복원하고 다음 배치부터 재개한다([scripts/trainer.py](../../scripts/trainer.py) `train_epoch()`의 `start_batch` 스킵 로직).

## 4. 알아둘 동작

- **결정성 없음(🟠).** 시드를 설정하지 않아 실행마다 결과가 다르다. 재현이 필요하면 `main()` 시작부에 `torch.manual_seed(...)` 등을 직접 추가한다 — [explanation/known-issues.md](../explanation/known-issues.md) §2 D4.
- **`min_t` 결합(🟠).** `forward_process()`는 `t∈[min_t, T)`만 샘플링하므로 모델은 `t<min_t` 영역을 학습하지 않는다. 이는 샘플러 절단과 연결된다 — [explanation/known-issues.md](../explanation/known-issues.md) §1 D1.
- **검증/체크포인트 주기.** `val_interval` **배치**마다 [scripts/validate.py](../../scripts/validate.py) `validate()`가 검증 손실을 계산하고, 개선 시 `{exp_name}_best.pth`를 저장한다.

## 5. 산출물 위치

| 경로 | 내용 |
|---|---|
| `checkpoints/{exp_name}_epoch{e}_batch{b}.pth` | 주기적 체크포인트 |
| `checkpoints/{exp_name}_best.pth` | 검증 손실 최저 |
| `logs/{exp_name}_{time}.log` | 텍스트 로그 |
| `runs/{exp_name}_{time}/` | TensorBoard 이벤트 |

## 관련 문서

- [tutorials/quickstart.md](../tutorials/quickstart.md) · [reference/configuration.md](../reference/configuration.md) · [reference/api-reference.md](../reference/api-reference.md)
