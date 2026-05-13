# Punching-shear Project

이 프로젝트는 뚫림 전단(Punching-shear) 분석을 위한 데이터 처리 및 모델링 코드를 포함하고 있습니다.

## 가상환경 설정 가이드 (Anaconda)

다음에 한 번에 가상환경을 구축하려면 아래 단계를 따르세요.

### 1. 가상환경 생성
```bash
conda create -n punching-shear python=3.10 -y
```

### 2. 가상환경 활성화
```bash
conda activate punching-shear
```

### 3. 패키지 설치
`requirements.txt` 파일을 사용하여 필요한 패키지를 설치합니다.

```bash
# 주요 패키지 우선 설치 (추천)
conda install pandas numpy matplotlib scikit-learn xgboost shap seaborn jupyter -c conda-forge -y

# 나머지 패키지 설치
pip install -r requirements.txt
```

*참고: 만약 특정 버전 설치 시 에러가 발생한다면, `requirements.txt`의 버전 정보를 지우고 패키지 이름만 남겨서 설치를 시도해 보세요.*

## 프로젝트 구조
- `코드/`: Jupyter Notebook 및 파이썬 스크립트
- `데이터 베이스/`: 분석에 사용되는 데이터 파일
- `requirements.txt`: 프로젝트 의존성 목록
