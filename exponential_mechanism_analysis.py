import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import softmax
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def exponential_mechanism_probabilities(utilities, epsilon):
    """
    计算指数机制的被抽样概率
    
    参数:
    utilities: 候选方案的效用值列表
    epsilon: 隐私预算参数
    
    返回:
    每个候选方案被抽样的概率
    """
    # 指数机制的概率公式: P(i) ∝ exp(ε * u_i / (2 * Δu))
    # 其中Δu是效用函数的敏感度，这里假设为utility的最大可能变化
    # 对于utility在1-10之间的情况，Δu = 10 - 1 = 9
    
    delta_u = 9  # 效用函数的敏感度
    scaled_utilities = epsilon * utilities / (2 * delta_u)
    
    # 使用softmax函数计算概率分布
    probabilities = softmax(scaled_utilities)
    
    return probabilities

def generate_normal_utilities(n_candidates=50, mean=5.5, std=1.5, min_val=1, max_val=10):
    """
    生成正态分布的效用值，并限制在[min_val, max_val]范围内
    """
    # 生成正态分布的随机数
    utilities = np.random.normal(mean, std, n_candidates)
    
    # 截断到指定范围
    utilities = np.clip(utilities, min_val, max_val)
    
    return utilities

def analyze_exponential_mechanism():
    """
    分析指数机制在不同epsilon值下的行为
    """
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    
    # 生成50个候选方案的效用值（正态分布）
    utilities = generate_normal_utilities(n_candidates=50, mean=5.5, std=1.5)
    
    # 不同的epsilon值
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('指数机制在不同ε值下的被抽样概率分布', fontsize=16, fontweight='bold')
    
    # 绘制效用值分布
    ax_util = axes[0, 0]
    ax_util.hist(utilities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax_util.set_title('候选方案效用值分布（正态分布）')
    ax_util.set_xlabel('效用值')
    ax_util.set_ylabel('频次')
    ax_util.grid(True, alpha=0.3)
    
    # 按效用值排序的索引
    sorted_indices = np.argsort(utilities)[::-1]  # 降序排列
    sorted_utilities = utilities[sorted_indices]
    
    # 绘制不同epsilon值下的概率分布
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, epsilon in enumerate(epsilon_values):
        if i == 0:
            ax = axes[0, 1]
        elif i == 1:
            ax = axes[0, 2]
        elif i == 2:
            ax = axes[1, 0]
        elif i == 3:
            ax = axes[1, 1]
        elif i == 4:
            ax = axes[1, 2]
        
        # 计算概率
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_probabilities = probabilities[sorted_indices]
        
        # 绘制概率分布
        bars = ax.bar(range(len(sorted_utilities)), sorted_probabilities, 
                     color=colors[i], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'ε = {epsilon}')
        ax.set_xlabel('候选方案（按效用值降序排列）')
        ax.set_ylabel('被抽样概率')
        ax.grid(True, alpha=0.3)
        
        # 添加效用值标注（显示前5个和后5个）
        if i == 0:  # 只在第一个子图中添加效用值标注
            for j in range(0, len(sorted_utilities), 10):
                ax.text(j, sorted_probabilities[j] + 0.001, 
                       f'{sorted_utilities[j]:.2f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/exponential_mechanism_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 分析不同epsilon值的影响
    print("=" * 60)
    print("指数机制分析结果")
    print("=" * 60)
    
    print(f"候选方案数量: {len(utilities)}")
    print(f"效用值范围: [{utilities.min():.2f}, {utilities.max():.2f}]")
    print(f"效用值均值: {utilities.mean():.2f}")
    print(f"效用值标准差: {utilities.std():.2f}")
    print()
    
    # 分析每个epsilon值下的行为
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # 找到最高效用值的候选方案
        max_utility_idx = np.argmax(utilities)
        max_utility_prob = probabilities[max_utility_idx]
        
        # 计算概率分布的集中程度（熵）
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(utilities))  # 均匀分布的最大熵
        concentration = 1 - entropy / max_entropy  # 集中程度指标
        
        print(f"ε = {epsilon}:")
        print(f"  最高效用值: {utilities[max_utility_idx]:.2f}")
        print(f"  最高效用方案被选中概率: {max_utility_prob:.4f}")
        print(f"  概率分布集中程度: {concentration:.4f} (0=均匀, 1=完全集中)")
        print(f"  前5个高效用方案累计概率: {np.sum(probabilities[sorted_indices[:5]]):.4f}")
        print()
    
    # 创建对比图：效用值 vs 概率
    plt.figure(figsize=(12, 8))
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        plt.scatter(utilities, probabilities, label=f'ε = {epsilon}', 
                   alpha=0.7, s=50)
    
    plt.xlabel('效用值')
    plt.ylabel('被抽样概率')
    plt.title('效用值与被抽样概率的关系（不同ε值）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/yizhang/tech4HSE/utility_vs_probability.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return utilities, epsilon_values

def theoretical_analysis():
    """
    理论分析指数机制的概率分布特性
    """
    print("\n" + "=" * 60)
    print("理论分析")
    print("=" * 60)
    
    print("指数机制的数学公式：")
    print("P(i) = exp(ε * u_i / (2 * Δu)) / Σ_j exp(ε * u_j / (2 * Δu))")
    print()
    
    print("关键特性：")
    print("1. ε越大，概率分布越集中在高效用方案上")
    print("2. ε越小，概率分布越接近均匀分布")
    print("3. 效用值差异被ε放大，影响最终概率分布")
    print("4. 当ε→0时，所有方案被选中概率趋于相等")
    print("5. 当ε→∞时，只有最高效用方案有非零概率")
    print()
    
    print("在正态分布效用值下的表现：")
    print("- 大部分候选方案的效用值接近均值")
    print("- 少数方案有显著高于或低于均值的效用值")
    print("- 指数机制会放大这些差异，特别是在高ε值时")
    print("- 中等ε值时，既保护隐私又保持效用性")

if __name__ == "__main__":
    # 运行分析
    utilities, epsilon_values = analyze_exponential_mechanism()
    
    # 理论分析
    theoretical_analysis()
