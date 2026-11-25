"""Local validation entry point for the SASRec implementation."""

import torch

from src.model import (
    AuthorSASRecBlock,
    PaperSASRecBlock,
    SASRec,
    _build_attention_mask,
)


def test_attention_mask() -> None:
    causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    padding_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )

    mask_1h = _build_attention_mask(causal_mask, padding_mask, num_heads=1)
    assert mask_1h.shape == (2, 4, 4)

    # 왼쪽 padding 시퀀스 >> padding query는 대각선만 열고 실제 query는 padding/미래 key 차단
    expected_padded = torch.tensor(
        [
            [False, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
            [True, True, False, False],
        ]
    )
    assert torch.equal(mask_1h[0], expected_padded)
    # padding이 없는 시퀀스는 causal_mask와 동일해야 함
    assert torch.equal(mask_1h[1], causal_mask)
    # 모든 query 행에 참조 가능한 key가 최소 1개 이상 있어야 softmax NaN을 피함
    assert (~mask_1h).any(dim=-1).all()

    # multi-head(B * num_heads, S, S) 배치별 head 복제 순서 확인
    mask_2h = _build_attention_mask(causal_mask, padding_mask, num_heads=2)
    assert mask_2h.shape == (4, 4, 4)
    assert torch.equal(mask_2h[0], mask_1h[0])
    assert torch.equal(mask_2h[1], mask_1h[0])
    assert torch.equal(mask_2h[2], mask_1h[1])
    assert torch.equal(mask_2h[3], mask_1h[1])

    print("Attention mask test passed")


def test_block_padding_isolation_and_need_weights() -> None:
    torch.manual_seed(0)
    causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    padding_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )

    for block_cls in (PaperSASRecBlock, AuthorSASRecBlock):
        block = block_cls(hidden_dim=16, num_heads=2, dropout=0.0)
        block.eval()

        # padding 위치 입력을 크게 바꿔도 실제 토큰 출력은 변하지 않아야 함
        x_base = torch.randn(2, 4, 16)
        x_perturbed = x_base.clone()
        x_perturbed[0, :2] = torch.randn(2, 16) * 100.0

        out_base = block(x_base, causal_mask, padding_mask)
        out_perturbed = block(x_perturbed, causal_mask, padding_mask)
        assert torch.allclose(out_base[0, 2:], out_perturbed[0, 2:], atol=0.0)
        assert torch.all(out_base[padding_mask] == 0.0)

        # 수정 전 NaN이 나던 need_weights=True 경로에서도 출력·weight·gradient가 유한해야 함
        x_req = x_base.clone().requires_grad_(True)
        query = block.attention_norm(x_req)
        kv = x_req if block_cls is AuthorSASRecBlock else query
        attn_mask = _build_attention_mask(
            causal_mask,
            padding_mask,
            block.attention.num_heads,
        )
        attn_out, attn_weights = block.attention(
            query=query,
            key=kv,
            value=kv,
            attn_mask=attn_mask,
            need_weights=True,
        )
        assert torch.isfinite(attn_out).all()
        assert torch.isfinite(attn_weights).all()

        attn_out.masked_fill(padding_mask.unsqueeze(-1), 0.0).sum().backward()
        assert x_req.grad is not None
        assert torch.isfinite(x_req.grad).all()

    print("Block padding isolation & need_weights test passed")


def test_sasrec_padding_forward_backward_no_nan() -> None:
    left_padded = torch.tensor(
        [
            [0, 0, 0, 1, 2, 3],
            [0, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    all_padded = torch.zeros(2, 4, dtype=torch.long)

    for variant in ("paper", "author"):
        for num_heads in (1, 2):
            torch.manual_seed(0)
            model = SASRec(
                num_items=20,
                max_len=10,
                hidden_dim=16,
                num_blocks=2,
                num_heads=num_heads,
                dropout=0.2,
                block_variant=variant,
            )

            for item_ids in (left_padded, all_padded):
                model.zero_grad()
                out = model(item_ids)
                assert torch.isfinite(out).all()
                assert torch.all(out[item_ids == 0] == 0.0)

                out.sum().backward()
                assert all(
                    torch.isfinite(param.grad).all()
                    for param in model.parameters()
                    if param.grad is not None
                )
                assert torch.all(model.item_embedding.weight.grad[0] == 0.0)

    print("SASRec padding forward/backward test passed")


if __name__ == "__main__":
    test_attention_mask()
    test_block_padding_isolation_and_need_weights()
    test_sasrec_padding_forward_backward_no_nan()
