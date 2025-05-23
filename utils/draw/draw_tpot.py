import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 시간 (초)
time = list(range(1, 61))

# 주어진 데이터
fp16 = [
    0.01179597, 0.06020945, 0.05454887, 0.0089037, 0.0522357, 0.00799494, 0.01275074, 0.02470552, 0.00879756, 0.00786673,
    0.00868132, 0.01301624, 0.05325762, 0.00831558, 0.00826423, 0.00807513, 0.00817144, 0.03852738, 0.05314716, 0.05464233,
    0.01868088, 0.00828703, 0.00804927, 0.05217837, 0.05756357, 0.01569729, 0.04360542, 0.00826336, 0.00826462, 0.04877432,
    0.03211435, 0.02805118, 0.00857651, 0.00820453, 0.00749885, 0.00761642, 0.00802207, 0.00809145, 0.00797314, 0.00767517,
    0.00787236, 0.00763879, 0.04022409, 0.05243692, 0.05022572, 0.00868716, 0.04806428, 0.00853822, 0.04878281, 0.03985286,
    0.05486794, 0.00785215, 0.00773081, 0.00831645, 0.00810887, 0.01115562, 0.00888029, 0.05217422, 0.03279865, 0.00848653
]

fp8 = [
    0.00792997, 0.04472003, 0.02522685, 0.00608095, 0.03578841, 0.00578472, 0.00608815, 0.00633399, 0.00592461, 0.00564181,
    0.00604673, 0.00616698, 0.02019948, 0.00618409, 0.0059688, 0.00602268, 0.00570374, 0.00850184, 0.03612794, 0.03796672,
    0.00952782, 0.00610202, 0.00602906, 0.03494768, 0.03699266, 0.00646602, 0.01111924, 0.00625187, 0.00595589, 0.00997179,
    0.00810698, 0.00605733, 0.00616013, 0.00620772, 0.00560838, 0.0057967, 0.00575608, 0.00617757, 0.00602203, 0.03266899,
    0.00579379, 0.00588385, 0.00778385, 0.03751155, 0.00650777, 0.00722022, 0.02284008, 0.00625831, 0.0294352, 0.03089899,
    0.01272591, 0.0377558, 0.00568155, 0.00547728, 0.0060461, 0.00638487, 0.00599238, 0.01570663, 0.00651295, 0.00642383
]


#min, max, mean

fp16_min = min(fp16)
fp16_max = max(fp16)
fp16_mean = sum(fp16) / len(fp16)
fp8_min = min(fp8)
fp8_max = max(fp8)
fp8_mean = sum(fp8) / len(fp8)
print(f"FP16 - Min: {fp16_min}, Max: {fp16_max}, Mean: {fp16_mean}")
print(f"FP8 - Min: {fp8_min}, Max: {fp8_max}, Mean: {fp8_mean}")

count_fp16_violations = sum(1 for x in fp16 if x > 0.033)
count_fp8_violations = sum(1 for x in fp8 if x > 0.033)
print(f"FP16 violations: {count_fp16_violations}")
print(f"FP8 violations: {count_fp8_violations}")

## SLO 기준선
threshold = 0.033

# Dual precision 선택
dual = [f16 if f16 <= threshold else f8 for f16, f8 in zip(fp16, fp8)]

# 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 5))

# 기본 라인 플롯
# 기존 라인 플롯 → 점 플롯으로 변경 (선 없이 점만)
# ax.scatter(time, fp16, color='darkblue', label='FP16', s=30, marker='s')
# ax.scatter(time, fp8, color='darkred', label='FP8', s=30, marker='s')

ax.plot(time, fp16, marker='s', linestyle='--', color='darkblue', label='FP16', linewidth=1.2)
ax.plot(time, fp8, marker='s', linestyle='--', color='darkred', label='FP8', linewidth=1.2)

ax.plot(time, dual, color='mediumpurple', linewidth=3.0, label='Dual-Precision')

# Dual 마커 수동 추가
for t, f16_val, f8_val, dual_val in zip(time, fp16, fp8, dual):
    marker_color = 'darkblue' if f16_val <= threshold else 'darkred'
    ax.plot(t, dual_val, marker='s', color=marker_color, markersize=6)

# SLO 수평선
ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.0, label='SLO(33ms)')

# 축 화살표 추가
x_max = max(time) + 3
y_max = max(max(fp16), max(fp8)) + 0.01
ax.annotate('', xy=(x_max, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
ax.annotate('', xy=(0, y_max), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# 기존 축 지우기
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# 라벨 및 격자
ax.set_xlim(0, x_max)
ax.set_ylim(0, y_max)
ax.set_xlabel("Time (s)", fontsize=20)
ax.set_ylabel("p90 TPOT (s)", fontsize=20)
ax.grid(True, linestyle='--', alpha=0.6)


ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)


# 범례: 위쪽 가운데, 프레임 없음
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=18)

# 저장 및 출력
plt.tight_layout()
plt.savefig('p90_tpot_over_time.pdf', dpi=400, bbox_inches='tight')