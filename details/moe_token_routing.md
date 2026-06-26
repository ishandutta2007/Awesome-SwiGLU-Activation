# Mixture-of-Experts (MoE) Token Routing Networks

In massive scale networks like GLaM, DeepSeek-V3, and Mixtral, Mixture-of-Experts (MoE) topologies are used to scale parameter count while maintaining active compute constant.

## Gated activations in MoE

An MoE layer routes each input token to a subset of available "expert" networks. Each expert is typically structured as a Feed-Forward Network. 
By employing gated linear architectures (like GEGLU in GLaM, or SwiGLU in DeepSeek-V3) within the expert networks:
1.  **High-Capacity Specialized Representation:** The gate within each expert dynamically filters which dimensions of the token's features are processed by that expert.
2.  **Smooth Routing Compatibility:** The continuous gating output integrates cleanly with the top-k router's soft gating weights.

## Diagram: MoE Gated Expert Routing

```mermaid
graph TD
    Token[Token Input] --> Router{Router Gate}
    Router -->|Top-k Expert 1| Expert1[Expert 1: GEGLU / SwiGLU FFN]
    Router -->|Top-k Expert 2| Expert2[Expert 2: GEGLU / SwiGLU FFN]
    Expert1 --> Combine[Weighted Sum]
    Expert2 --> Combine
    Combine --> Out[Output]
```

---
[← Back to README](../README.md)
