class Config:
    def __init__(self):
        # 기존 파라미터
        self.HASH_DIM = 64
        self.FEATURE_DIM = 2048
        self.BOTTLENECK_DIM = 256
        self.BATCH_SIZE = 128    # 증가 (더 빠른 학습)
        #self.NUM_EPOCHS = 150    # 증가 (더 충분한 학습)
        self.NUM_EPOCHS = 10    # 증가 (더 충분한 학습)
        self.LEARNING_RATE = 0.0001  # 감소 (더 안정적인 학습)
        self.WEIGHT_DECAY = 1e-4    # 조정 (더 강한 정규화)
        self.WARMUP_EPOCHS = 10     # 증가 (더 안정적인 시작)
        self.TEMPERATURE = 0.1
        self.GRAD_CLIP = 1.0
        self.DATASET_NAME = 'cifar10'

        # 추가할 파라미터들
        # 데이터 관련
        self.NUM_WORKERS = 4
        self.INPUT_SIZE = 224
        
        # 모델 관련
        self.DROPOUT = 0.5
        self.PRETRAINED = True
        
        # 학습 관련
        self.STAGE1_EPOCHS = 75   # 전체 epoch의 절반
        self.PRINT_FREQ = 50     # 더 자주 출력
        self.SAVE_FREQ = 15      # 적절한 간격으로 조정
        self.VAL_FREQ = 5        # 현재 값 유지
        self.VAL_BATCH_FREQ = 200  # 200 배치마다 validation 수행
        
        # Loss 가중치 (논문의 값으로 조정)
        self.ALPHA = 1.0
        self.BETA = 0.1
        self.GAMMA = 0.1
        
        # 평가 관련
        self.TOP_K = [1, 100, 1000]  # Precision@K 계산을 위한 K값들
        
        # 경로 관련
        self.DATA_DIR = './data'
        self.RESULT_DIR = './result'
        self.MODEL_DIR = 'model'
        self.LOG_DIR = 'log'
        self.CODE_DIR = 'code'