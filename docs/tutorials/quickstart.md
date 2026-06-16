# 빠른 시작 (Quickstart)

> **범위:** 설치부터 MNIST 학습 실행, TensorBoard로 생성 샘플을 확인하는 happy path. 개별 과업(재개·튜닝·샘플링)은 [how-to/](../how-to/) 참고.
> **대상:** 처음 이 repo를 실행하는 개발자.
> **상태:** 구현 반영 — 기준일 2026-06-17.

## 1. 사전 준비

- conda(또는 mamba)와, GPU 사용 시 NVIDIA 드라이버 + CUDA 11.8 호환 환경.
- 디스크: MNIST 자동 다운로드(~12MB) + 체크포인트 공간.
- 🟢 CPU만 있어도 동작은 하지만 모델이 무거워 매우 느리다(아래 §4 참고).

## 2. 환경 생성

```bash
conda env create -f environment.yml      # 환경 이름: deep_thermo
conda activate deep_thermo
```

설정·의존성 상세는 [reference/configuration.md](../reference/configuration.md) 참고. `scipy`는 Swiss Roll 데모에서 쓰이지만 `environment.yml`에 선언돼 있지 않다(보통 scikit-learn으로 함께 설치됨).

## 3. MNIST 학습 실행

```bash
chmod +x train.sh
./train.sh
```

- [train.sh](../../train.sh)가 [scripts/train.py](../../scripts/train.py) `main()`을 기본 하이퍼파라미터로 실행한다(MNIST는 `download=True`로 자동 다운로드).
- 🟠 `train.sh`는 `CUDA_VISIBLE_DEVICES=2`를 하드코딩한다. GPU가 1장이거나 인덱스가 다르면 다음처럼 덮어쓴다:

  ```bash
  CUDA_VISIBLE_DEVICES=0 ./train.sh
  ```

- 진행 로그는 콘솔과 `logs/`에, 체크포인트는 `checkpoints/`에, TensorBoard 이벤트는 `runs/`에 쌓인다.

## 4. 동작 확인

학습 루프([scripts/trainer.py](../../scripts/trainer.py) `train_epoch()`)는 `val_interval` 배치마다 검증 손실을 찍고 `DiffusionModel.sample()`로 미리보기 샘플을 생성해 TensorBoard에 로깅한다.

```bash
tensorboard --logdir runs
```

확인할 항목:

| 패널 | 의미 |
|---|---|
| `train/loss`, `eval/loss` | 학습/검증 음의 로그우도(낮을수록 좋음, 글로벌 배치 스텝) |
| `train/epoch_loss` | 에폭 평균 학습 손실(에폭 단위 스텝) |
| `samples/generated` | 역방향 과정으로 생성한 샘플 |
| `samples/real`, `samples/noisy` | 원본·순방향으로 잡음화한 이미지 |

> 🟠 **샘플이 흐릿/잡음이 남아 보일 수 있다.** 기본 샘플러(`DiffusionModel.sample()`)는 `t=min_t`(기본 100)에서 멈추고 `t=0`까지 내려가지 않는다. 의도된 한계가 아니라 알려진 이슈다 — [explanation/known-issues.md](../explanation/known-issues.md) §1 D1.

## 5. 다음 단계

- 하이퍼파라미터 변경·체크포인트 재개 → [how-to/train-mnist.md](../how-to/train-mnist.md)
- 학습된 모델로 샘플/inpainting → [how-to/sample-and-inpaint.md](../how-to/sample-and-inpaint.md)
- 2D 토이(Swiss Roll) 데모 → [how-to/run-swiss-roll.md](../how-to/run-swiss-roll.md)
- 방법론 이해 → [explanation/diffusion-method.md](../explanation/diffusion-method.md)
