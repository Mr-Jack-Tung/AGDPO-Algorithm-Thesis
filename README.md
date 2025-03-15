# AGDPO-Algorithm-Thesis

The main ideas behind **Adaptive Group-Direct Policy Optimization (AGDPO)** along with its algorithm outline.

## Main Ideas of AGDPO
**Hybrid Approach:**
  AGDPO combines the strengths of direct preference optimization (DPO) and group relative comparisons (GRPO) while incorporating stabilization techniques from PPO (such as clipping). This hybrid approach enables the model to be fine-tuned directly from human preference data while mitigating noise and instability.

**Direct Optimization with Preferences:**
  Rather than building an explicit reward model, AGDPO directly optimizes the policy using human feedback. It leverages both pairwise comparisons (winner vs. loser) and group-level rankings to extract richer information about user preferences.

**Stabilization via Clipping and Anchoring:**
  To prevent large policy updates (which can lead to divergence), AGDPO uses a clipping mechanism similar to PPO. It anchors the policy update with respect to a reference policy so that the new policy does not drift too far from the original, ensuring stable training.

**Adaptive Loss Combination:**
AGDPO defines two loss components:
  - Pairwise Loss: Encourages the policy to prefer responses that humans ranked higher.
  - Group Loss: Aggregates information from a group of ranked responses to capture overall preference trends.

An adaptive weighting parameter α balances these two losses:

<img width="310" alt="two loss components" src="https://github.com/user-attachments/assets/cf869445-9f68-4e09-9ba9-ccc0986ecdfb" />


## AGDPO Algorithm Outline

<img width="680" alt="AGDPO Algorithm Outline" src="https://github.com/user-attachments/assets/85a01272-0061-46ea-a3af-cf16423f2fb5" />


## Pseudocode for AGDPO
```
for each training iteration:
    # 1. Collect data: For each input x, get outputs with human rankings.
    batch_data = collect_preference_data()

    # 2. For each sample in the batch:
    for x, ranked_outputs in batch_data:
        # ranked_outputs may include pairs and group rankings
        rewards = {}
        for y in ranked_outputs:
            rewards[y] = log(pi_theta(y|x)) - log(pi_ref(y|x))
        
        # 3. Compute pairwise loss for all pairs (winner, loser)
        L_pair = 0
        for (y_w, y_l) in pairwise_pairs(ranked_outputs):
            L_pair += -log(sigmoid(rewards[y_w] - rewards[y_l]))
        
        # 4. Compute group loss if available
        if group_ranking_available(ranked_outputs):
            y_top = get_top_response(ranked_outputs)
            L_group = 0
            for y in ranked_outputs excluding y_top:
                L_group += (rewards[y_top] - rewards[y])
            L_group = -log(sigmoid(L_group / (len(ranked_outputs)-1)))
        else:
            L_group = 0
        
        # 5. Combine losses with adaptive weighting
        L_total = alpha * L_group + (1 - alpha) * L_pair
        
        # 6. Apply clipping to ratios (if using a PPO-style update)
        # Adjust rewards using:
        # clipped_ratio = clip(pi_theta(y|x)/pi_ref(y|x), 1-eps, 1+eps)
        # [Implementation details depend on the chosen framework]
        
        # 7. Update policy parameters theta using gradient descent on L_total
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
    
    # Optionally update reference policy pi_ref with current pi_theta periodically.
```


Summary
AGDPO is an innovative approach that:
- Directly optimizes the language model policy based on human preferences.
- Leverages both pairwise and group ranking signals.
- Incorporates clipping to maintain stable policy updates.
- Uses an adaptive loss that dynamically balances group and pairwise components.
