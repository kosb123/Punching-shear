import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import sys

# 출력 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# 폰트 및 스타일 설정
plt.style.use("classic")
try:
    plt.rcParams["font.family"] = "Times New Roman"
except:
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 12
plt.rcParams['axes.unicode_minus'] = False

def generate_distribution_plot(data_series, column_name, save_dir):
    """특정 컬럼에 대한 히스토그램 및 CDF 그래프 생성"""
    print(f"Processing {column_name}...", flush=True)
    try:
        # 데이터 전처리
        data = pd.to_numeric(data_series, errors='coerce').dropna()
        if len(data) == 0:
            print(f"Warning: {column_name}에 유효한 데이터가 없습니다.")
            return

        # 가중치 및 구간 설정
        weights = np.ones_like(data) / len(data)
        bin_min, bin_max = data.min(), data.max()
        bins = np.linspace(bin_min, bin_max, 20)
        
        # x축 범위 설정
        margin = (bin_max - bin_min) * 0.05 if bin_max > bin_min else 1.0
        x_min, x_max = bin_min - margin, bin_max + margin

        # 히스토그램 데이터 계산 (y축 범위용)
        counts, _ = np.histogram(data, bins=bins, weights=weights)
        y1_max = counts.max() * 1.3 if len(counts) > 0 else 1.0

        # 그래프 생성
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        # 히스토그램 (왼쪽 y축)
        ax1.hist(data, bins=bins, weights=weights, color="red", alpha=0.6, 
                 edgecolor="black", label="Relative Frequencies")
        ax1.set_xlabel(column_name, fontsize=14)
        ax1.set_ylabel("Relative Frequency", fontsize=14)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(0, y1_max)

        # CDF (오른쪽 y축)
        try:
            grid_x = np.linspace(x_min, x_max, 1000)
            kde = gaussian_kde(data)
            density = kde(grid_x)
            cum = np.cumsum(density) * (grid_x[1] - grid_x[0])
            if cum[-1] > 0:
                cum = cum / cum[-1]
                ax2 = ax1.twinx()
                ax2.plot(grid_x, cum, color="blue", lw=2, label="CDF")
                ax2.set_ylabel("CDF", color="blue", fontsize=14)
                ax2.set_ylim(0, 1.05)
                ax2.tick_params(axis="y", labelcolor="blue")
                ax2.set_yticks(np.arange(0, 1.1, 0.5))
        except Exception as e:
            print(f"[{column_name}] CDF 계산 실패: {e}", flush=True)

        plt.title(f"Distribution of {column_name}", fontsize=16)
        
        # 저장
        save_path = os.path.join(save_dir, f"{column_name}_distribution.pdf")
        try:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            print(f"Saved: {save_path}", flush=True)
        except Exception as e:
            print(f"Error saving {column_name}: {e}", flush=True)
        
        plt.close()
    except Exception as e:
        import traceback
        traceback.print_exc()

def main():
    print("Main function started...", flush=True)
    # 경로 설정
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        excel_path = os.path.join(base_dir, "데이터 베이스", "3. 12개 입력값 머신러닝 전단파괴만_b100이하 제거.xlsx")
        output_dir = os.path.join(base_dir, "results", "plots")
        
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(excel_path):
            print(f"Error: 파일을 찾을 수 없습니다. ({excel_path})")
            return

        # 데이터 로드
        print(f"Loading data from: {os.path.basename(excel_path)}...", flush=True)
        df = pd.read_excel(excel_path)
        print(f"Data loaded successfully. Rows: {len(df)}", flush=True)
        
        # 분석할 컬럼 리스트
        target_columns = ['d', 'h', 'fc', 'fy', 'Vn']
        
        print(f"Generating plots for: {target_columns}", flush=True)
        for col in target_columns:
            if col in df.columns:
                generate_distribution_plot(df[col], col, output_dir)
            else:
                print(f"Skip: {col} (컬럼이 존재하지 않음)", flush=True)

        print("\n모든 작업이 완료되었습니다. 'results/plots' 폴더를 확인하세요.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
