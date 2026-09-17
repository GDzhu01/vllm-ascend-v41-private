import pytest
import torch

from vllm_ascend.ops.triton.spec_decode.next_token_ids import prepare_next_token_ids


def _reference(sampled, backup, discard_indices, num_discarded, vocab_size):
    filtered = sampled.clone()
    filtered.index_fill_(0, discard_indices[:num_discarded], -1)
    valid = (filtered != -1) & (filtered < vocab_size)
    counts = valid.sum(dim=1)
    selected = torch.gather(filtered, 1, (counts - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    return torch.where(counts != 0, selected, backup), counts


@pytest.mark.parametrize(("num_reqs", "max_new_tokens"), [(1, 1), (7, 6), (16, 6)])
@pytest.mark.parametrize("num_discarded", [0, 1, 4])
def test_prepare_next_token_ids(num_reqs, max_new_tokens, num_discarded):
    num_discarded = min(num_discarded, num_reqs)
    vocab_size = 129280
    sampled = torch.randint(
        0,
        vocab_size,
        (num_reqs, max_new_tokens),
        dtype=torch.int64,
        device="npu",
    )
    valid_lengths = torch.randint(0, max_new_tokens + 1, (num_reqs,))
    for req_idx, valid_length in enumerate(valid_lengths.tolist()):
        sampled[req_idx, valid_length:] = -1
    if num_reqs > 1:
        sampled[1, 0] = vocab_size

    backup = torch.arange(num_reqs, dtype=torch.int64, device="npu") + 900000
    discard_indices = torch.arange(num_reqs - 1, -1, -1, dtype=torch.int64, device="npu")

    expected = _reference(sampled, backup, discard_indices, num_discarded, vocab_size)
    actual = prepare_next_token_ids(
        sampled,
        backup,
        discard_indices,
        num_discarded,
        vocab_size,
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
