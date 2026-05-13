import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # GUI 없는 환경에서도 안전하게 실행
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import sys

# 1. 스타일 및 폰트 설정
plt.style.use("classic")
try:
    plt.rcParams["font.family"] = "Times New Roman"
except:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 14
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 2. 경로 설정 (프로젝트 루트 기준)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "데이터 베이스", "3. 12개 입력값 머신러닝 전단파괴만_b100이하 제거.xlsx")
    save_dir = os.path.join(base_dir, "데이터 분포 그래프")
    
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다.\n경로: {file_path}")
        return

    # 3. 데이터 불러오기
    print(f"데이터 로딩 중: {os.path.basename(file_path)}")
    data = pd.read_excel(file_path)
    
    if "d" not in data.columns:
        print("오류: 데이터에 'd' 컬럼이 존재하지 않습니다.")
        return
        
    reinforcement_data = pd.to_numeric(data["d"], errors='coerce').dropna()

    # 4. 히스토그램 가중치 및 구간 설정
    weights = np.ones_like(reinforcement_data) / len(reinforcement_data)
    bin_min = reinforcement_data.min()
    bin_max = reinforcement_data.max()
    bins = np.linspace(bin_min, bin_max, 20)

    # 5. x축 범위 및 y축 범위 설정
    margin = (bin_max - bin_min) * 0.05
    x_min = bin_min - margin
    x_max = bin_max + margin

    counts, _ = np.histogram(reinforcement_data, bins=bins, weights=weights)
    y1_max = counts.max() * 1.2

    # 6. 그래프 생성
    print("그래프 생성 중 (d 분포)...")
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 6-1. 왼쪽 y축: 히스토그램
    ax1.hist(
        reinforcement_data,
        bins=bins,
        weights=weights,
        color="red",
        alpha=0.6,
        edgecolor="black",
        label="Relative Frequencies",
    )
    ax1.set_xlabel("d", fontsize=16)
    ax1.set_ylabel("Relative Frequency", fontsize=16)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, y1_max)
    ax1.tick_params(axis="both", labelsize=14)

    # 6-2. 오른쪽 y축: CDF 곡선
    grid_x = np.linspace(x_min, x_max, 1000)
    try:
        kde = gaussian_kde(reinforcement_data)
        density = kde(grid_x)
        cum = np.cumsum(density) * (grid_x[1] - grid_x[0])
        cum = cum / cum[-1]
        
        ax2 = ax1.twinx()
        ax2.plot(grid_x, cum, color="blue", lw=2, label="CDF")
        ax2.set_ylabel("CDF", color="blue", fontsize=16)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="blue", labelsize=14)
        ax2.set_yticks(np.arange(0, 1.01, 0.5))
    except Exception as e:
        print(f"CDF 계산 오류: {e}")

    plt.title("Distribution of d", fontsize=18)

    # 7. 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "1. d.pdf")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
    
    print(f"\n[성공] 파일이 저장되었습니다!\n위치: {save_path}")

if __name__ == "__main__":
    main()
