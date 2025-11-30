import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Import both implementations
import DQN
import C51

def plot_comparison(dqn_curves, c51_curves):
    """Plot DQN and C51 performance comparison"""
    
    # Calculate statistics
    dqn_mean = np.mean(dqn_curves, axis=0)
    dqn_std = np.std(dqn_curves, axis=0)
    c51_mean = np.mean(c51_curves, axis=0)
    c51_std = np.std(c51_curves, axis=0)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot DQN
    episodes_dqn = range(len(dqn_mean))
    plt.plot(episodes_dqn, dqn_mean, color='red', label='DQN', linewidth=2)
    plt.fill_between(episodes_dqn, 
                     np.maximum(dqn_mean - dqn_std, 0), 
                     np.minimum(dqn_mean + dqn_std, 500), 
                     color='red', alpha=0.3)
    
    # Plot C51
    episodes_c51 = range(len(c51_mean))
    plt.plot(episodes_c51, c51_mean, color='blue', label='C51', linewidth=2)
    plt.fill_between(episodes_c51, 
                     np.maximum(c51_mean - c51_std, 0), 
                     np.minimum(c51_mean + c51_std, 500), 
                     color='blue', alpha=0.3)
    
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('DQN vs C51 Performance on Noisy CartPole', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('dqn_vs_c51_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"\nDQN:")
    print(f"  Final Average Reward: {dqn_mean[-1]:.2f} ± {dqn_std[-1]:.2f}")
    print(f"  Best Average Reward: {np.max(dqn_mean):.2f}")
    print(f"  Convergence Episode: {np.argmax(dqn_mean > 0.8 * np.max(dqn_mean))}")
    
    print(f"\nC51:")
    print(f"  Final Average Reward: {c51_mean[-1]:.2f} ± {c51_std[-1]:.2f}")
    print(f"  Best Average Reward: {np.max(c51_mean):.2f}")
    print(f"  Convergence Episode: {np.argmax(c51_mean > 0.8 * np.max(c51_mean))}")
    
    improvement = ((c51_mean[-1] - dqn_mean[-1]) / dqn_mean[-1]) * 100
    print(f"\nImprovement: {improvement:.2f}%")
    print("="*60)

if __name__ == "__main__":
    print("Training DQN...")
    print("="*60)
    dqn_curves = []
    for seed in DQN.SEEDS:
        dqn_curves.append(DQN.train(seed))
    
    print("\n\nTraining C51...")
    print("="*60)
    c51_curves = []
    for seed in C51.SEEDS:
        c51_curves.append(C51.train(seed))
    
    print("\n\nGenerating comparison plot...")
    plot_comparison(dqn_curves, c51_curves)