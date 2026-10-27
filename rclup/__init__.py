"""Receiver-Conditioned Clarification for Multi-Agent Food Assistance: fan-out topology (paper Appendix G).

intake -> three parallel receivers {access, schedule, logistics} -> deterministic joint planner -> retriever, evaluated on
HoosierHelp with a frozen Qwen3-8B backbone served by vLLM and an OpenAI-simulated user.  The package name `rclup`
abbreviates the project's working title (receiver-conditioned latent uncertainty propagation).
"""
