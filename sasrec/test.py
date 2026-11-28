"""Local validation entry point for the SASRec implementation."""

import math
from pathlib import Path
import tempfile

import torch
from torch.utils.data import DataLoader

from src.data import load_sasrec_interactions, split_user_histories
from src.dataset import (
    SASRecTrainDataset,
    create_sasrec_collate_fn,
    pad_sequences_left,
)
from src.metrics import (
    compute_ranking_metrics,
    compute_target_ranks,
    hit_rate_at_k,
    ndcg_at_k,
)
from src.model import (
    AuthorSASRecBlock,
    PaperSASRecBlock,
    SASRec,
    _build_attention_mask,
)
from src.training import (
    TrainingResumeState,
    capture_rng_state,
    compute_bce_loss,
    create_sasrec_optimizer,
    evaluate,
    extract_model_config,
    load_model_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
    set_seed,
    train_epoch,
)


def test_attention_mask() -> None:
    # 결합 attention mask가 padding query의 대각선만 열고 미래/padding key를 차단하는지 확인
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
    # Paper/Author 블록에서 padding 입력 격리와 need_weights=True 경로의 NaN 방지를 확인
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
    # 왼쪽 패딩 및 전체 패딩 배치에서 forward/backward 시 NaN 없이 동작하는지 확인
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


def test_sasrec_model_contracts() -> None:
    # Xavier 초기화(0번 패딩 유지), causality, 고정/가변 길이 일치, 입력 길이 검증, score_items 확인
    for variant in ("paper", "author"):
        torch.manual_seed(42)
        model = SASRec(
            num_items=30,
            max_len=8,
            hidden_dim=16,
            num_blocks=2,
            num_heads=2,
            dropout=0.0,
            block_variant=variant,
        )
        model.eval()

        # padding_idx=0 행은 초기화 후에도 정확히 0이어야 함
        assert torch.all(model.item_embedding.weight[0] == 0.0)
        assert model.item_embedding.weight[1:].abs().max().item() > 0.0

        # 미래 시점 토큰 변경이 과거 시점 출력에 영향을 주지 않아야 함(causality)
        seq_a = torch.tensor([[0, 0, 1, 2, 3, 4]])
        seq_b = torch.tensor([[0, 0, 1, 2, 9, 4]])
        assert torch.allclose(model(seq_a)[:, :4], model(seq_b)[:, :4], atol=0.0)

        # 고정 길이(max_len) 왼쪽 패딩과 가변 길이 입력의 유효 위치 출력이 일치해야 함
        fixed_seq = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4]])
        dynamic_seq = torch.tensor([[1, 2, 3, 4]])
        assert torch.allclose(model(fixed_seq)[:, -4:], model(dynamic_seq), atol=1e-6)

        # 학습용 (B, S, K)와 평가용 (B, K) 및 전체 아이템 (C,) 채점 shape 확인
        features = model(seq_a)
        assert model.score_items(features, torch.randint(1, 31, (1, 6, 2))).shape == (1, 6, 2)
        assert model.score_items(features[:, -1, :], torch.randint(1, 31, (1, 5))).shape == (1, 5)
        assert model.score_items(features[:, -1, :], torch.arange(1, 31)).shape == (1, 30)

        # 잘못된 차원이나 max_len 초과 시퀀스는 ValueError를 발생시켜야 함
        for bad_input in (
            torch.tensor([1, 2, 3]),
            torch.empty(1, 0, dtype=torch.long),
            torch.ones(1, 9, dtype=torch.long),
        ):
            try:
                model(bad_input)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Expected ValueError for shape {tuple(bad_input.shape)}")

    print("SASRec model contracts test passed")


def test_data_and_dataset_padding() -> None:
    # 상호작용 로딩/분할과 전체 이력 기준 negative 샘플링 및 fixed/dynamic 왼쪽 패딩 확인
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = Path(tmp_dir) / "interactions.txt"
        data_file.write_text(
            "1 2\n1 4\n1 6\n1 8\n1 10\n2 1\n2 3\n3 5\n",
            encoding="utf-8",
        )
        histories = load_sasrec_interactions(data_file)
        split = split_user_histories(histories)

        # 3개 이상인 사용자 1만 valid/test로 분리되고 나머지는 train에만 남아야 함
        assert split.train == {1: [2, 4, 6], 2: [1, 3], 3: [5]}
        assert split.valid == {1: 8}
        assert split.test == {1: 10}
        assert split.num_items == 10

        set_seed(0)
        dataset = SASRecTrainDataset(split.train, num_items=split.num_items, max_len=4)
        # 이력 길이 1인 사용자 3은 제외되어 2명만 남아야 함
        assert len(dataset) == 2

        sample_u1 = dataset[0]
        assert torch.equal(sample_u1[0], torch.tensor([2, 4]))
        assert torch.equal(sample_u1[1], torch.tensor([4, 6]))
        assert all(int(neg) not in {2, 4, 6} for neg in sample_u1[2])

        # fixed는 max_len(4), dynamic은 배치 최장 길이(2)로 왼쪽 패딩되어야 함
        batch_samples = [dataset[0], dataset[1]]
        fixed_batch = create_sasrec_collate_fn(padding_mode="fixed", max_len=4)(batch_samples)
        dynamic_batch = create_sasrec_collate_fn(padding_mode="dynamic", max_len=4)(batch_samples)
        assert fixed_batch[0].shape == (2, 4)
        assert dynamic_batch[0].shape == (2, 2)
        assert torch.equal(
            pad_sequences_left([[3], [1, 2]], padding_mode="dynamic", max_len=4),
            torch.tensor([[0, 3], [1, 2]]),
        )

    print("Data & dataset padding test passed")


def test_ranking_metrics() -> None:
    # HR@K, NDCG@K 수식 정확도와 동점 보수적 처리 및 -inf 마스킹 동작 확인
    ranks = torch.tensor([0, 1, 9, 10])
    assert math.isclose(hit_rate_at_k(ranks, k=10), 0.75)
    expected_ndcg = (1.0 + 1.0 / math.log2(3.0) + 1.0 / math.log2(11.0)) / 4.0
    assert math.isclose(ndcg_at_k(ranks, k=10), expected_ndcg, rel_tol=1e-12)

    # 모든 후보 점수가 같을 때 상수 모델이 HR=1.0을 받지 않아야 함
    tied_metrics = compute_ranking_metrics(torch.zeros(4, 20), k=10)
    assert tied_metrics.hr == 0.0 and tied_metrics.ndcg == 0.0

    # Full Ranking에서 이미 소비한 아이템(-inf)은 순위에서 제외되어야 함
    masked_scores = torch.tensor([[5.0, float("-inf"), 6.0, 2.0]])
    assert torch.equal(
        compute_target_ranks(masked_scores, target_indices=torch.tensor([0])),
        torch.tensor([1]),
    )

    print("Ranking metrics test passed")


def test_training_and_checkpoint() -> None:
    # 패딩 제외 BCE loss, 1-epoch 학습 갱신, valid/test 평가, 체크포인트 복원을 확인
    pos_scores = torch.tensor([[50.0, 0.5, -0.3]], requires_grad=True)
    neg_scores = torch.tensor([[-50.0, -0.2, 0.4]], requires_grad=True)
    valid_mask = torch.tensor([[False, True, True]])
    loss = compute_bce_loss(pos_scores, neg_scores, valid_mask)
    loss.backward()
    # 패딩 위치(0번 열)는 gradient가 0이어야 함
    assert pos_scores.grad is not None and pos_scores.grad[0, 0].item() == 0.0
    assert neg_scores.grad is not None and neg_scores.grad[0, 0].item() == 0.0

    histories = {
        u: [((u + step) % 12) + 1 for step in range(6)]
        for u in range(1, 7)
    }
    split = split_user_histories(histories)
    dataset = SASRecTrainDataset(split.train, num_items=split.num_items, max_len=4)
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=create_sasrec_collate_fn(padding_mode="dynamic", max_len=4),
    )

    for variant in ("paper", "author"):
        set_seed(7)
        model = SASRec(
            num_items=split.num_items,
            max_len=4,
            hidden_dim=16,
            num_blocks=2,
            num_heads=1,
            dropout=0.0,
            block_variant=variant,
        )
        optimizer = create_sasrec_optimizer(model, lr=5e-3)
        before = model.item_embedding.weight[1:].detach().clone()

        epoch_loss = train_epoch(model, loader, optimizer, "cpu", show_progress=False)
        assert epoch_loss > 0.0
        assert not torch.equal(model.item_embedding.weight[1:], before)
        assert torch.all(model.item_embedding.weight[0] == 0.0)

        # Sampled 평가와 Full Ranking 평가에서 fixed/dynamic 결과가 일치해야 함
        val_dyn = evaluate(
            model,
            split,
            "cpu",
            mode="valid",
            k=5,
            num_negatives=5,
            padding_mode="dynamic",
            seed=0,
            show_progress=False,
        )
        val_fix = evaluate(
            model,
            split,
            "cpu",
            mode="valid",
            k=5,
            num_negatives=5,
            padding_mode="fixed",
            seed=0,
            show_progress=False,
        )
        test_full = evaluate(
            model,
            split,
            "cpu",
            mode="test",
            k=5,
            num_negatives=None,
            show_progress=False,
        )
        assert val_dyn == val_fix
        assert 0.0 <= test_full.hr <= 1.0 and 0.0 <= test_full.ndcg <= 1.0

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "last.pt"
            sample_history = [
                {
                    "epoch": 1,
                    "train_loss": epoch_loss,
                    "elapsed_sec": 0.1,
                    "valid_hr": val_dyn.hr,
                    "valid_ndcg": val_dyn.ndcg,
                }
            ]
            run_config = {"lr": 5e-3, "eval_k": 5, "padding_mode": "dynamic"}
            save_training_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch=1,
                train_loss=epoch_loss,
                valid_metrics=val_dyn,
                best_epoch=1,
                best_valid_metrics=val_dyn,
                history=sample_history,
                run_config=run_config,
            )
            reloaded = load_model_checkpoint(ckpt_path, device="cpu")
            assert extract_model_config(reloaded) == extract_model_config(model)
            probe = torch.tensor([[0, 1, 2, 3]])
            assert torch.equal(reloaded(probe), model(probe))

            # 학습 재개용 체크포인트 복원(모델·옵티마이저 상태·진행 기록·설정 검증) 확인
            resumed_model = SASRec(
                num_items=split.num_items,
                max_len=4,
                hidden_dim=16,
                num_blocks=2,
                num_heads=1,
                dropout=0.0,
                block_variant=variant,
            )
            resumed_optimizer = create_sasrec_optimizer(resumed_model, lr=5e-3)
            resume_state = load_training_checkpoint(
                ckpt_path,
                resumed_model,
                resumed_optimizer,
                expected_run_config=run_config,
            )
            assert resume_state == TrainingResumeState(
                epoch=1,
                train_loss=epoch_loss,
                valid_metrics=val_dyn,
                best_epoch=1,
                best_valid_hr=val_dyn.hr,
                best_valid_ndcg=val_dyn.ndcg,
                history=sample_history,
                rng_restored=False,
            )
            resumed_model.eval()
            model.eval()
            assert torch.equal(resumed_model(probe), model(probe))
            assert len(resumed_optimizer.state) == len(optimizer.state)

            # 실험 설정 불일치 시 ValueError를 발생시켜야 함
            try:
                load_training_checkpoint(
                    ckpt_path,
                    resumed_model,
                    resumed_optimizer,
                    expected_run_config={"lr": 1e-3, "eval_k": 5, "padding_mode": "dynamic"},
                )
            except ValueError:
                pass
            else:
                raise AssertionError("Expected ValueError for mismatched run_config")

            # 모델 구조가 다른 체크포인트로 복원 시 ValueError를 발생시켜야 함
            wrong_model = SASRec(
                num_items=split.num_items,
                max_len=4,
                hidden_dim=8,
                num_blocks=2,
                num_heads=1,
                dropout=0.0,
                block_variant=variant,
            )
            wrong_optimizer = create_sasrec_optimizer(wrong_model, lr=5e-3)
            try:
                load_training_checkpoint(ckpt_path, wrong_model, wrong_optimizer)
            except ValueError:
                pass
            else:
                raise AssertionError("Expected ValueError for mismatched model_config")

            # 난수 상태(RNG) 저장/복원 시 연속 2-epoch 학습과 중단 후 재개 학습이 정확히 일치해야 함
            def make_shuffled_loader(gen: torch.Generator) -> DataLoader:
                return DataLoader(
                    dataset,
                    batch_size=3,
                    shuffle=True,
                    collate_fn=create_sasrec_collate_fn(
                        padding_mode="dynamic",
                        max_len=4,
                    ),
                    generator=gen,
                )

            set_seed(99)
            gen_cont = torch.Generator().manual_seed(99)
            loader_cont = make_shuffled_loader(gen_cont)
            model_cont = SASRec(
                num_items=split.num_items,
                max_len=4,
                hidden_dim=16,
                num_blocks=2,
                num_heads=1,
                dropout=0.2,
                block_variant=variant,
            )
            opt_cont = create_sasrec_optimizer(model_cont, lr=5e-3)
            loss_cont_1 = train_epoch(model_cont, loader_cont, opt_cont, "cpu", show_progress=False)
            loss_cont_2 = train_epoch(model_cont, loader_cont, opt_cont, "cpu", show_progress=False)

            rng_ckpt_path = Path(tmp_dir) / "rng_last.pt"
            set_seed(99)
            gen_part1 = torch.Generator().manual_seed(99)
            loader_part1 = make_shuffled_loader(gen_part1)
            model_part1 = SASRec(
                num_items=split.num_items,
                max_len=4,
                hidden_dim=16,
                num_blocks=2,
                num_heads=1,
                dropout=0.2,
                block_variant=variant,
            )
            opt_part1 = create_sasrec_optimizer(model_part1, lr=5e-3)
            loss_part1 = train_epoch(model_part1, loader_part1, opt_part1, "cpu", show_progress=False)
            assert loss_part1 == loss_cont_1
            save_training_checkpoint(
                rng_ckpt_path,
                model_part1,
                opt_part1,
                epoch=1,
                train_loss=loss_part1,
                rng_state=capture_rng_state(gen_part1),
            )

            set_seed(0)
            gen_part2 = torch.Generator().manual_seed(0)
            loader_part2 = make_shuffled_loader(gen_part2)
            model_part2 = SASRec(
                num_items=split.num_items,
                max_len=4,
                hidden_dim=16,
                num_blocks=2,
                num_heads=1,
                dropout=0.2,
                block_variant=variant,
            )
            opt_part2 = create_sasrec_optimizer(model_part2, lr=5e-3)
            rng_resume = load_training_checkpoint(
                rng_ckpt_path,
                model_part2,
                opt_part2,
                loader_generator=gen_part2,
            )
            assert rng_resume.rng_restored is True
            loss_part2 = train_epoch(model_part2, loader_part2, opt_part2, "cpu", show_progress=False)
            assert math.isclose(loss_part2, loss_cont_2, rel_tol=0.0, abs_tol=1e-12)
            for p_cont, p_res in zip(model_cont.parameters(), model_part2.parameters()):
                assert torch.equal(p_cont, p_res)

    print("Training & checkpoint test passed")


if __name__ == "__main__":
    test_attention_mask()
    test_block_padding_isolation_and_need_weights()
    test_sasrec_padding_forward_backward_no_nan()
    test_sasrec_model_contracts()
    test_data_and_dataset_padding()
    test_ranking_metrics()
    test_training_and_checkpoint()
