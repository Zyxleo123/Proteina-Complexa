"""Reward-weighted terminal-sample replay for CPSea fine-tuning.

See `proteinfoundation.replay.buffer.ReplayBuffer` (disk-backed FIFO storage of
generated terminal flow-space endpoints) and
`proteinfoundation.replay.weighting` (turns stored rewards into per-example
training weights).
"""
