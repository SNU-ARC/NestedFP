import matplotlib.pyplot as plt

# 데이터 입력
time = list(range(1, 61))
request_rate = [
    9, 8, 9, 5, 6, 5, 8, 7, 7, 2,
    6, 7, 8, 5, 3, 3, 5, 4, 6, 5,
    7, 6, 4, 8, 3, 6, 7, 3, 3, 6,
    5, 4, 3, 4, 2, 4, 4, 3, 4, 5,
    4, 4, 5, 6, 6, 3, 8, 5, 11, 3,
    6, 2, 1, 1, 2, 6, 5, 6, 5, 5
]




# 그래프 설정
fig, ax = plt.subplots(figsize=(10, 5))

# 선 그리기 (범례 없이)
ax.plot(time, request_rate, marker='s', color='navy', linewidth=2.0)

# 평균 request rate 계산 및 점선 추가 (범례 포함)
avg_rate = sum(request_rate) / len(request_rate)
ax.axhline(y=avg_rate, color='gray', linestyle='--', linewidth=1.5, label='Average Request Rate')

# 축 화살표 추가
x_max = max(time) + 3
y_max = max(request_rate) + 1
ax.annotate('', xy=(x_max, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.annotate('', xy=(0, y_max), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# 기존 축 숨기기
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# 축 라벨 및 스타일
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_xlabel("Time (s)", fontsize=20)
ax.set_ylabel("Request Rate", fontsize=20)
ax.grid(True, linestyle='--', alpha=0.6)

# 눈금 폰트 크기
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)

# 범례 (Average만 표시)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize=18)

# 저장 및 출력
plt.tight_layout()
plt.savefig('request_rate_over_time.pdf', dpi=400, bbox_inches='tight')
plt.show()
