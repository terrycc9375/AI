# built-in packages
import os
import math

# third-party packages
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

def pull_arm(arm: int) -> int:
    """
    input: arm index,
    output: reward
    """
    return REWARD * int(np.random.rand() <= WIN_PROB[arm])

def p1():
    def e_greedy(epsilon: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        input: epsilon,
        output: reward list and win-count list for each arm
        """
        rewards = np.zeros(3)
        counts = np.zeros(3)
        wins = np.zeros(3)

        for _ in range(NUM_TRIALS):
            if np.random.rand() < epsilon:
                arm = np.random.randint(3)  # Explore
            else:
                arm = np.argmax(rewards / (counts + 1e-9))  # Exploit
            reward = pull_arm(arm)
            win = int(reward > 0)
            rewards[arm] += reward
            counts[arm] += 1
            wins[arm] += win

        return rewards, counts, wins

    def ucb(scale: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        input: scale,
        output: reward list and win-count list for each arm
        """
        rewards = np.zeros(3)
        counts = np.zeros(3)
        wins = np.zeros(3)

        for t in range(1, NUM_TRIALS + 1):
            ucb_values = rewards / (counts + 1e-9) + scale * np.sqrt(np.log(t) / (counts + 1e-9))
            arm = np.argmax(ucb_values)
            reward = pull_arm(arm)
            rewards[arm] += reward
            win = int(reward > 0)
            counts[arm] += 1
            wins[arm] += win
        
        return rewards, counts, wins
    
    def thompson_sampling() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        output: reward list, win-count list for each arm, and \alpha, \beta
        """
        rewards = np.zeros(3)
        counts = np.zeros(3)
        wins = np.zeros(3)
        alpha = np.ones(3)
        beta = np.ones(3)

        for _ in range(NUM_TRIALS):
            samples = [scipy.stats.beta(alpha[i], beta[i]).rvs() for i in range(3)]
            arm = np.argmax(samples)
            reward = pull_arm(arm)
            rewards[arm] += reward
            win = int(reward > 0)
            counts[arm] += 1
            wins[arm] += win
            alpha[arm] += win
            beta[arm] += 1 - win
        
        return rewards, counts, wins, alpha, beta

    """
    Run algorithms
    """
    epsilon = 0.5
    confident = 0.5
    eg_rewards, eg_counts, eg_wins = e_greedy(epsilon)
    ucb_rewards, ucb_counts, ucb_wins = ucb(confident)
    ts_rewards, ts_counts, ts_wins, ts_alpha, ts_beta = thompson_sampling()

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    algorithms = [f'Epsilon-Greedy ($\\epsilon={epsilon}$)', f'UCB (C={confident})', 'Thompson Sampling']
    counts = [eg_counts, ucb_counts, ts_counts]
    wins = [eg_wins, ucb_wins, ts_wins]
    alphas = [eg_wins + 1, ucb_wins + 1, ts_alpha]
    betas = [eg_counts - eg_wins + 1, ucb_counts - ucb_wins + 1, ts_beta]
    x = np.linspace(0, 1, 500)
    colors = ['#77d5e6', '#97e677', '#e6a577']
    colors2 = ['#1bb1cc', '#5dd12e', '#e88138']
    for i in range(3):
        ax_bar = axes[i, 0]
        ax_bar.bar(range(3), counts[i], label='Total Played', color=colors[i])
        ax_bar.bar(range(3), wins[i], label='Total Wins', color=colors2[i], alpha=1.0)
        ax_bar.set_title(f'{algorithms[i]} - Counts and Wins')
        ax_bar.set_ylabel('Count')
        ax_bar.set_ylim(0, max(counts[i]) * 1.2)
        ax_bar.legend()

        for j in range(3):
            ax_bar.text(j, counts[i][j] + 15, f"P:{int(counts[i][j])}\nW:{int(wins[i][j])}", ha='center', va='bottom', fontsize=9)

        ax_beta = axes[i, 1]
        for j in range(3):
            y = scipy.stats.beta.pdf(x, alphas[i][j], betas[i][j])
            ax_beta.plot(x, y, label=f'Arm {j} (Alpha={int(alphas[i][j])}, Beta={int(betas[i][j])})', lw=2)
            ax_beta.fill_between(x, 0, y, alpha=0.1)
        for p in WIN_PROB:
            ax_beta.axvline(x=p, color='gray', linestyle='--', alpha=0.5)
        ax_beta.set_title(f'{algorithms[i]} - Beta Distributions')
        ax_beta.set_xlabel('Win Probability')
        ax_beta.set_ylabel('Density')
        ax_beta.legend()
    plt.tight_layout()
    plt.savefig(f'{FOLDER}/part1.png')
    # plt.show()

def run_epsilon_greedy():
    def e_greedy(epsilon: float = 0.5) -> int:
        """
        input: epsilon,
        output: win-count for this epsilon
        """
        counts = np.zeros(3)
        wins = np.zeros(3)

        for _ in range(NUM_TRIALS):
            if np.random.rand() < epsilon:
                arm = np.random.randint(3)  # Explore
            else:
                arm = np.argmax(wins / (counts + 1e-9))  # Exploit
            win = int(pull_arm(arm) > 0)
            counts[arm] += 1
            wins[arm] += win

        return np.sum(wins)
    
    """Part 1: Linearly ascending epsilon"""
    # epsilons = np.arange(0.05, 1.0, 0.05) # for looking 0.05 to 0.95, uncommend this line
    epsilons = np.arange(0.005, 0.100, 0.005) # for looking 0.005 to 0.095, uncommend this line
    uniform_win_counts = [e_greedy(epsilon) for epsilon in epsilons]

    """Part 2: Time-dependent epsilon, t starts from 1"""
    time_dependent_epsilons = [
        lambda t: 1/t, 
        lambda t: 1/np.sqrt(t), 
        lambda t: 1/np.log(t-1 + math.e + 1e-9), # log(e + t-1) >= 1 for t >= 1
        lambda t: np.exp(-(t-1) / 100) # exp((t-1)) >= 1 for t >= 1
    ]
    def td_e_greedy(funcs: list) -> list[int]:
        """
        Input: list of time-dependent epsilon functions,
        Output: list of total win counts for each function
        """
        win_counts = []

        for func in funcs:
            counts = np.zeros(3)
            wins = np.zeros(3)
            for t in range(1, NUM_TRIALS + 1):
                epsilon = func(t)
                if np.random.rand() < epsilon:
                    arm = np.random.randint(3)  # Explore
                else:
                    arm = np.argmax(wins / (counts + 1e-9))  # Exploit
                win = int(pull_arm(arm) > 0)
                counts[arm] += 1
                wins[arm] += win
            win_counts.append(np.sum(wins))

        return win_counts
    
    time_dependent_win_counts = td_e_greedy(time_dependent_epsilons)

    # """Part 2-1: More study on exponential decay epsilon"""
    # scales = np.linspace(1, 1000, 100)
    # functions = [lambda t: np.exp(-(t-1) / s) for s in scales]
    # exp_win_counts = td_e_greedy(functions)

    fig1, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilons, uniform_win_counts, marker='o', label='Uniform Epsilon', color="#9341d6")
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Total Win Count')
    ax.set_title('Epsilon-Greedy with Different Epsilon Values')
    ax.legend()
    ax.grid(True)
    fig1.savefig(f'{FOLDER}/uniform_epsilon_greedy.png')

    fig2, ax = plt.subplots(figsize=(10, 6))
    labels = ['1/t', '1/sqrt(t)', '1/log(t)', 'exp(-t/100)']
    ax.bar(labels, time_dependent_win_counts, color="#db71b2")
    ax.set_ylim(0, max(time_dependent_win_counts) * 1.2)
    for i, count in enumerate(time_dependent_win_counts):
        ax.text(i, count + 15, f"{count}", ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Time-dependent Epsilon Function')
    ax.set_ylabel('Total Win Count')
    ax.set_title('Epsilon-Greedy with Time-dependent Epsilon Functions')
    ax.grid(True)
    fig2.savefig(f'{FOLDER}/time_dependent_epsilon_greedy.png')

    # fig3, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(scales, exp_win_counts, marker='o', color="#71dbdb", label="Exponential Scale")
    # ax.set_ylim(min(exp_win_counts) * 0.9, max(exp_win_counts) * 1.1)
    # ax.set_xlabel('Exponential Decay Epsilon Function exp(-t/s)')
    # ax.set_ylabel('Total Win Count')
    # ax.set_title('Epsilon-Greedy with Exponential Decay Epsilon Functions')
    # ax.legend()
    # ax.grid(True)
    # fig3.savefig(f'{FOLDER}/exponential_decay_epsilon_greedy.png')


def run_ucb():
    def adjusted_ucb(scale: float = 0.5, output_win_only: bool = False) -> tuple[np.ndarray, np.ndarray] | int:
        """
        input: scale,
        output: count and win-count for this scale
        """
        counts = np.zeros(3)
        wins = np.zeros(3)

        for t in range(1, NUM_TRIALS + 1):
            ucb_values = wins / (counts + 1e-9) + scale * np.sqrt(np.log(t) / (counts + 1e-9))
            arm = np.argmax(ucb_values)
            win = int(pull_arm(arm) > 0)
            counts[arm] += 1
            wins[arm] += win
        
        if output_win_only:
            return np.sum(wins)
        return counts, wins
    
    """Part 1: Test adjusted UCB with CL = 0.5"""
    ucb_counts, ucb_wins = adjusted_ucb(0.5, False)
    alphas = ucb_wins + 1
    betas = ucb_counts - ucb_wins + 1
    fig1, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].bar(range(3), ucb_counts, label='Total Played', color='#97e677')
    ax[0].bar(range(3), ucb_wins, label='Total Wins', color='#5dd12e', alpha=1.0)
    ax[0].set_title('UCB with CL=0.5 - Counts and Wins')
    ax[0].set_ylabel('Count')
    ax[0].set_ylim(0, max(ucb_counts) * 1.2)
    ax[0].legend()
    for j in range(3):
        ax[0].text(j, ucb_counts[j] + 15, f"P:{int(ucb_counts[j])}\nW:{int(ucb_wins[j])}", ha='center', va='bottom', fontsize=9)
    
    x = np.linspace(0, 1, 500)
    for j in range(3):
        y = scipy.stats.beta.pdf(x, alphas[j], betas[j])
        ax[1].plot(x, y, label=f'Arm {j} (Alpha={int(alphas[j])}, Beta={int(betas[j])})', lw=2)
        ax[1].fill_between(x, 0, y, alpha=0.1)
    for p in WIN_PROB:
        ax[1].axvline(x=p, color='gray', linestyle='--', alpha=0.5)
    ax[1].set_title('UCB with CL=0.5 - Beta Distributions')
    ax[1].set_xlabel('Win Probability')
    ax[1].set_ylabel('Density')
    ax[1].legend()
    fig1.savefig(f'{FOLDER}/ucb_cl_0.5.png')

    """Part 2: Varying confidence level C"""
    scales = np.linspace(0.1, 2.0, 20)
    ucb_wins = [adjusted_ucb(scale, True) for scale in scales]
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(scales, ucb_wins, marker='o', color="#2dc471")
    ax.set_xlabel('UCB Scale')
    ax.set_ylabel('Total Win Count')
    ax.set_title('UCB with Different Scales')
    ax.grid(True)
    fig2.savefig(f'{FOLDER}/ucb_scales.png')

def run_thompson_sampling():
    def thompson_sampling(alpha: int, beta: int):
        counts = np.zeros(3)
        wins = np.zeros(3)
        alphas = np.ones(3) * alpha
        betas = np.ones(3) * beta
        for _ in range(NUM_TRIALS):
            samples = [scipy.stats.beta(alphas[i], betas[i]).rvs() for i in range(3)]
            arm = np.argmax(samples)
            win = int(pull_arm(arm) > 0)
            counts[arm] += 1
            wins[arm] += win
            alphas[arm] += win
            betas[arm] += 1 - win
        return np.sum(wins)
    
    """Vary Thompson Sampling with Beta(alpha, beta) prior where alpha in [1, 5] and beta in [1, 5]"""
    heatmap_data = np.zeros((10, 10))
    alpha_range = range(1, 11)
    beta_range = range(1, 11)
    results = np.ndarray((100, 10))
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            group = [thompson_sampling(alpha, beta) for _ in range(10)]
            heatmap_data[i, j] = np.mean(group)
            results[i*10 + j] = group

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt='.0f', 
        cmap='YlGnBu',
        xticklabels=[f'β={b}' for b in beta_range],
        yticklabels=[f'α={a}' for a in alpha_range],
        cbar_kws={'label': 'Total Win Count'}
    )
    plt.title('Thompson Sampling Win count Heatmap', fontsize=14, pad=15)
    plt.xlabel('Beta ($\\beta$) prior', fontsize=12)
    plt.ylabel('Alpha ($\\alpha$) prior', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"Thompson_sampling_varing_prior.png")

    f, p = scipy.stats.f_oneway(*results)
    print("Null hypothesis: different priors don't affect the final results")
    print("Significant level = 0.05")
    print(f"F-statistic: {f:.4f}\np-value: {p:.4e}")
    if p < 0.05: print("Reject H0")
    else: print("Failed to reject H0")


if __name__ == "__main__":
    """Global settings"""
    WIN_PROB = (0.3, 0.7, 0.8)
    REWARD = 20
    NUM_TRIALS = 1000
    FOLDER = 'results'
    os.makedirs(FOLDER, exist_ok=True)
    np.random.seed(42)

    """
    Part 1: Multi-armed Bandit Problem with 3 arms, 3 different algorithms
    """
    p1()

    """
    Part 2: Hyperparameter Selection

    In this part, we consider the total reward we get (or equivalently, the total win count) 
    as a criterion of each algorithm. 
    
    I. Epsilon-Greedy: 
        1. Vary epsilon from 0.05 to 0.95 by 0.05.
        2. Time-dependent epsilon: epsilon(t) = 1/t, 1/sqrt(t), 1/log(t), exp(-t/1000), etc.
        3. (deprecated) Randomize epsilon: epsilon(t) = random value in [0.05, 0.95] for each trial.
    II. UCB:
        1. Use win count instead of average reward since the reward may dominate the UCB value.
        2. Vary the confidence level C from 0.1 to 2.0 by 0.1.
    III. Thompson Sampling:
        1. Use different priors, e.g., Beta(2, 2), Beta(5, 5), Beta(1, 3), etc.
        2. Use larger number of trials, e.g., 10,000 or 100,000.
    """
    # run_epsilon_greedy()
    # run_ucb()
    run_thompson_sampling()



