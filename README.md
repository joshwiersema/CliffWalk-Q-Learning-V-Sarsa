### CliffWalking Q-Learning vs SARSA  

**A small reinforcement-learning experiment comparing on-policy vs off-policy learning on a classic gridworld**  

## 🧠 Overview  
This project trains two agents — one using tabular **Q-Learning** (off-policy) and one using **SARSA** (on-policy) — on the `CliffWalking-v0` environment from Gymnasium. The goal is to explore how exploration (ε-greedy), learning rate (α), and discount factor (γ) influence learned behavior, especially risk-taking:  
- Q-Learning tends to learn a **risky, cliff-hugging** but high-reward path.  
- SARSA tends to learn a **safer**, longer but more consistent path that avoids the cliff more often.  

This project offers a concrete demonstration of on-policy vs off-policy learning, and how RL hyperparameters affect performance, stability, and safety.  

## ✅ Key Features  
- Simple tabular implementations of Q-Learning and SARSA in Python  
- Training and evaluation over many episodes  
- Visualization of learning curves (total reward per episode)  
- Statistics tracking: cliff-fall rate, episode length, reward averages  
- Greedy-policy output shown on the grid (ASCII-style map) to illustrate learned behavior  

## 🛠️ Tech Stack  
- Python 3.x  
- gymnasium  
- NumPy  
- Matplotlib  
- (Optional) Jupyter Notebook for interactive runs or plotting  

## 📄 What You’ll Learn / What It Demonstrates  
- Differences between on-policy (SARSA) and off-policy (Q-Learning) RL  
- How hyperparameters (ε, α, γ) affect learning dynamics, risk, and stability  
- Basics of reproducible experiments: logging stats, plotting, comparing algorithms  

---

**Author:** Josh Wiersema  
**Contact:** [josh.wiersema06@gmail.com / [LinkedIn](https://www.linkedin.com/in/josh-wiersema-526452377/) 

