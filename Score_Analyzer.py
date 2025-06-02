import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd



def analyze_score_distribution(dataset, sample_size=None, figsize=(15, 12)):
    """
    Analyze and visualize score distribution from a TF dataset of (sample, score) tuples.
    
    Args:
        dataset: tf.data.Dataset containing (sample, score) tuples
        sample_size: Number of samples to extract (None for all data)
        figsize: Figure size for plots
    """
    
    # Extract scores from the dataset
    print("Extracting scores from dataset...")
    scores = []
    
    # Handle batched data
    for batch in dataset:
        if isinstance(batch, tuple) and len(batch) == 2:
            _, batch_scores = batch
            # Convert to numpy and flatten if needed
            batch_scores_np = batch_scores.numpy()
            if batch_scores_np.ndim > 1:
                batch_scores_np = batch_scores_np.flatten()
            scores.extend(batch_scores_np)
        
        if sample_size and len(scores) >= sample_size:
            scores = scores[:sample_size]
            break
    
    scores = np.array(scores)
    print(f"Extracted {len(scores)} scores")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Score Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram with KDE
    ax1 = axes[0, 0]
    ax1.hist(scores, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add KDE curve
    kde = stats.gaussian_kde(scores)
    x_range = np.linspace(scores.min(), scores.max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    ax1.set_title('Score Distribution with KDE')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot with outlier detection
    ax2 = axes[0, 1]
    box_plot = ax2.boxplot(scores, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    ax2.set_title('Box Plot (Outlier Detection)')
    ax2.set_ylabel('Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Log-scale histogram (for tail analysis)
    ax3 = axes[0, 2]
    # Remove zeros and negative values for log scale
    positive_scores = scores[scores > 0] if np.any(scores > 0) else scores + abs(scores.min()) + 1e-10
    ax3.hist(positive_scores, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_yscale('log')
    ax3.set_title('Log-Scale Histogram')
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency (log scale)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Rank-order plot (Zipf-like analysis)
    ax4 = axes[1, 0]
    sorted_scores = np.sort(scores)[::-1]  # Sort in descending order
    ranks = np.arange(1, len(sorted_scores) + 1)
    ax4.loglog(ranks, sorted_scores, 'b-', alpha=0.7, linewidth=1)
    ax4.set_title('Rank-Order Plot (Log-Log)')
    ax4.set_xlabel('Rank (log scale)')
    ax4.set_ylabel('Score (log scale)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Cumulative distribution
    ax5 = axes[1, 1]
    sorted_scores_asc = np.sort(scores)
    cumulative_prob = np.arange(1, len(sorted_scores_asc) + 1) / len(sorted_scores_asc)
    ax5.plot(sorted_scores_asc, cumulative_prob, 'g-', linewidth=2)
    ax5.set_title('Cumulative Distribution Function')
    ax5.set_xlabel('Score')
    ax5.set_ylabel('Cumulative Probability')
    ax5.grid(True, alpha=0.3)
    
    # 6. Q-Q plot for normality check
    ax6 = axes[1, 2]
    stats.probplot(scores, dist="norm", plot=ax6)
    ax6.set_title('Q-Q Plot (Normality Check)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("DETAILED STATISTICS")
    print("="*50)
    
    # Basic statistics
    print(f"Sample size: {len(scores):,}")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")
    print(f"Min: {np.min(scores):.4f}")
    print(f"Max: {np.max(scores):.4f}")
    print(f"Range: {np.ptp(scores):.4f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p:2d}th: {np.percentile(scores, p):.4f}")
    
    # Skewness and kurtosis
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)
    print(f"\nDistribution shape:")
    print(f"Skewness: {skewness:.4f} ({'right-skewed' if skewness > 0 else 'left-skewed' if skewness < 0 else 'symmetric'})")
    print(f"Kurtosis: {kurtosis:.4f} ({'heavy-tailed' if kurtosis > 0 else 'light-tailed' if kurtosis < 0 else 'normal-tailed'})")
    
    # Tail analysis
    print(f"\nTail Analysis:")
    top_1_percent = np.percentile(scores, 99)
    top_5_percent = np.percentile(scores, 95)
    top_10_percent = np.percentile(scores, 90)
    
    tail_1_pct = scores[scores >= top_1_percent]
    tail_5_pct = scores[scores >= top_5_percent]
    tail_10_pct = scores[scores >= top_10_percent]
    
    print(f"Top 1% threshold: {top_1_percent:.4f} ({len(tail_1_pct)} samples)")
    print(f"Top 5% threshold: {top_5_percent:.4f} ({len(tail_5_pct)} samples)")
    print(f"Top 10% threshold: {top_10_percent:.4f} ({len(tail_10_pct)} samples)")
    
    # Concentration analysis
    total_sum = np.sum(scores) if np.sum(scores) > 0 else 1
    top_1_pct_contribution = np.sum(tail_1_pct) / total_sum * 100
    top_5_pct_contribution = np.sum(tail_5_pct) / total_sum * 100
    top_10_pct_contribution = np.sum(tail_10_pct) / total_sum * 100
    
    print(f"\nConcentration (% of total score mass):")
    print(f"Top 1% contributes: {top_1_pct_contribution:.2f}%")
    print(f"Top 5% contributes: {top_5_pct_contribution:.2f}%")
    print(f"Top 10% contributes: {top_10_pct_contribution:.2f}%")
    
    # Power law test (for rank-order plot)
    if len(scores) > 100:
        try:
            # Simple power law test using linear regression on log-log scale
            log_ranks = np.log(ranks[:len(ranks)//2])  # Use first half to avoid noise
            log_scores = np.log(sorted_scores[:len(ranks)//2])
            
            # Remove any infinite values
            valid_idx = np.isfinite(log_ranks) & np.isfinite(log_scores)
            if np.sum(valid_idx) > 10:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    log_ranks[valid_idx], log_scores[valid_idx]
                )
                print(f"\nPower Law Analysis (Rank-Order):")
                print(f"Slope (power law exponent): {slope:.4f}")
                print(f"R-squared: {r_value**2:.4f}")
                print(f"P-value: {p_value:.4e}")
                
                if abs(slope) > 0.5 and r_value**2 > 0.8:
                    print("Strong evidence of power law distribution (potential long tail)")
                elif abs(slope) > 0.3 and r_value**2 > 0.6:
                    print("Moderate evidence of power law distribution")
                else:
                    print("Weak evidence of power law distribution")
        except:
            print("Could not perform power law analysis")
    
    # Normality tests
    print(f"\nNormality Tests:")
    try:
        shapiro_stat, shapiro_p = stats.shapiro(scores[:5000] if len(scores) > 5000 else scores)
        print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")
        
        ks_stat, ks_p = stats.kstest(scores, 'norm', args=(np.mean(scores), np.std(scores)))
        print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.4e}")
        
        if shapiro_p < 0.05 or ks_p < 0.05:
            print("Distribution is NOT normal (p < 0.05)")
        else:
            print("Distribution may be normal (p >= 0.05)")
    except:
        print("Could not perform normality tests")
    
    return scores

# Example usage:
# Assuming your dataset is named 'dataset'
# scores = analyze_score_distribution(dataset, sample_size=10000)

def create_additional_tail_analysis(scores, figsize=(12, 8)):
    """
    Additional focused analysis on tail behavior and score dropoff.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Detailed Tail Analysis', fontsize=14, fontweight='bold')
    
    # 1. Survival function (1 - CDF)
    ax1 = axes[0, 0]
    sorted_scores = np.sort(scores)[::-1]
    survival_prob = np.arange(len(sorted_scores)) / len(sorted_scores)
    ax1.semilogy(sorted_scores, 1 - survival_prob, 'b-', linewidth=2)
    ax1.set_title('Survival Function (Log Scale)')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('P(Score >= x)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Score differences (dropoff rate)
    ax2 = axes[0, 1]
    score_diffs = np.diff(sorted_scores)  # Differences between consecutive scores
    ax2.plot(range(len(score_diffs)), -score_diffs, 'r-', alpha=0.7)
    ax2.set_title('Score Dropoff Rate')
    ax2.set_xlabel('Rank')
    ax2.set_ylabel('Score Difference (Dropoff)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Percentile plot
    ax3 = axes[1, 0]
    percentiles = np.arange(1, 100)
    percentile_values = [np.percentile(scores, p) for p in percentiles]
    ax3.plot(percentiles, percentile_values, 'g-', linewidth=2)
    ax3.set_title('Percentile Plot')
    ax3.set_xlabel('Percentile')
    ax3.set_ylabel('Score Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Tail ratio analysis
    ax4 = axes[1, 1]
    thresholds = np.linspace(np.percentile(scores, 50), np.percentile(scores, 99), 50)
    tail_proportions = [np.mean(scores >= t) for t in thresholds]
    ax4.plot(thresholds, tail_proportions, 'purple', linewidth=2)
    ax4.set_title('Tail Proportion vs Threshold')
    ax4.set_xlabel('Score Threshold')
    ax4.set_ylabel('Proportion Above Threshold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage example:
# scores = analyze_score_distribution(your_dataset)
# create_additional_tail_analysis(scores)