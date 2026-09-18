"""LoRA attach/detach behaviour, on a tiny model with a synthetic adapter.

The real adapters target a 28-layer 2048-wide model; these tests are about the
mapping and the arithmetic, which do not care about size.
"""
import pytest
import torch
from safetensors.torch import save_file

from audiogen import lora
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM


@pytest.fixture
def model():
    torch.manual_seed(11)
    config = YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                        vocab_size=32, max_position_embeddings=128,
                        latent_dim=64, vae_latent_dim=64, max_latent_frames=128)
    return YuE2ForCausalLM(config).eval()


def make_adapter(tmp_path, model, branch="nar", rank=4, scale=1e-2, name="a.safetensors"):
    """A low-rank pair per targeted projection, plus a dense diff for llm2vae."""
    attn = "nar_self_attn" if branch == "nar" else "self_attn"
    mlp = "nar_mlp" if branch == "nar" else "mlp"
    tensors, torch.manual_seed = {}, torch.manual_seed
    torch.manual_seed(5)
    for index in range(len(model.model.layers)):
        for module, projection in [(attn, "q_proj"), (attn, "o_proj"), (mlp, "down_proj")]:
            target = getattr(getattr(model.model.layers[index], module), projection)
            out_features, in_features = target.weight.shape
            stem = f"model.layers.{index}.{module}.{projection}"
            tensors[f"{stem}.lora_down.weight"] = torch.randn(rank, in_features) * scale
            tensors[f"{stem}.lora_up.weight"] = torch.randn(out_features, rank) * scale
    tensors["llm2vae.diff"] = torch.randn_like(model.llm2vae.weight) * scale
    tensors["llm2vae.diff_b"] = torch.randn_like(model.llm2vae.bias) * scale
    path = tmp_path / name
    save_file(tensors, str(path), metadata={"branch": branch, "rank": str(rank)})
    return path


def snapshot(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def changed(before, model):
    return {name for name, tensor in model.state_dict().items()
            if not torch.equal(tensor, before[name])}


class TestStrengthZero:
    def test_strength_zero_is_bit_identical_to_no_adapter(self, model, tmp_path):
        # If the mapping is wrong this is where it shows: a delta scaled by zero
        # must land on exactly the weights it came from.
        path = make_adapter(tmp_path, model)
        before = snapshot(model)
        lora.hook([lora.Adapter(path, "nar", strength=0.0)])(model, for_nar=True, fresh=True, loaded=True)
        assert changed(before, model) == set()


class TestApply:
    def test_delta_is_up_times_down(self, model, tmp_path):
        path = make_adapter(tmp_path, model)
        tensors, _ = lora.read(path)
        stem = "model.layers.0.nar_self_attn.q_proj"
        expected_delta = tensors[f"{stem}.lora_up.weight"].float() @ tensors[f"{stem}.lora_down.weight"].float()
        target = model.model.layers[0].nar_self_attn.q_proj
        base = target.weight.detach().clone()

        lora.hook([lora.Adapter(path, "nar", strength=1.0)])(model, for_nar=True, fresh=True, loaded=True)
        torch.testing.assert_close(target.weight, base + expected_delta.to(target.weight.dtype),
                                   atol=1e-6, rtol=0)

    def test_strength_scales_the_delta(self, model, tmp_path):
        path = make_adapter(tmp_path, model)
        target = model.model.layers[0].nar_self_attn.q_proj
        base = target.weight.detach().clone()

        lora.hook([lora.Adapter(path, "nar", strength=1.0)])(model, for_nar=True, fresh=True, loaded=True)
        full = (target.weight - base).clone()
        lora.hook([lora.Adapter(path, "nar", strength=0.5)])(model, for_nar=True, fresh=False, loaded=False)
        half = target.weight - base
        torch.testing.assert_close(half * 2, full, atol=1e-6, rtol=0)

    def test_dense_diff_reaches_weight_and_bias(self, model, tmp_path):
        path = make_adapter(tmp_path, model)
        before_w = model.llm2vae.weight.detach().clone()
        before_b = model.llm2vae.bias.detach().clone()
        lora.hook([lora.Adapter(path, "nar", strength=1.0)])(model, for_nar=True, fresh=True, loaded=True)
        assert not torch.equal(model.llm2vae.weight, before_w)
        assert not torch.equal(model.llm2vae.bias, before_b)


class TestIdempotence:
    def test_repeated_application_does_not_compound(self, model, tmp_path):
        # The engine fires on_model_ready on EVERY stage entry, so a naive
        # additive apply would keep stacking the same delta.
        path = make_adapter(tmp_path, model)
        apply = lora.hook([lora.Adapter(path, "nar", strength=1.0)])
        apply(model, for_nar=True, fresh=True, loaded=True)
        once = snapshot(model)
        for _ in range(3):
            apply(model, for_nar=True, fresh=False, loaded=False)
        assert changed(once, model) == set()

    def test_leaving_the_stage_restores_the_base_weights(self, model, tmp_path):
        path = make_adapter(tmp_path, model)
        before = snapshot(model)
        apply = lora.hook([lora.Adapter(path, "nar", strength=1.0)])
        apply(model, for_nar=True, fresh=True, loaded=True)
        assert changed(before, model)
        apply(model, for_nar=False, fresh=False, loaded=False)   # AR stage: NAR adapter off
        assert changed(before, model) == set()


class TestBranches:
    def test_a_nar_adapter_does_not_touch_the_ar_path(self, model, tmp_path):
        path = make_adapter(tmp_path, model, branch="nar")
        before = snapshot(model)
        lora.hook([lora.Adapter(path, "nar", 1.0)])(model, for_nar=True, fresh=True, loaded=True)
        touched = changed(before, model)
        assert touched, "adapter changed nothing at all"
        assert not any(".self_attn." in name or ".mlp." in name for name in touched), touched

    def test_an_ar_adapter_does_not_touch_the_nar_path(self, model, tmp_path):
        path = make_adapter(tmp_path, model, branch="ar")
        before = snapshot(model)
        lora.hook([lora.Adapter(path, "ar", 1.0)])(model, for_nar=False, fresh=True, loaded=True)
        touched = changed(before, model)
        assert touched
        assert not any("nar_" in name for name in touched), touched

    def test_a_nar_adapter_is_inert_during_the_ar_stage(self, model, tmp_path):
        path = make_adapter(tmp_path, model, branch="nar")
        before = snapshot(model)
        lora.hook([lora.Adapter(path, "nar", 1.0)])(model, for_nar=False, fresh=True, loaded=True)
        assert changed(before, model) == set()


class TestRejection:
    def test_an_unknown_branch_is_rejected(self, tmp_path):
        (tmp_path / "x.safetensors").write_bytes(b"")
        with pytest.raises(ValueError, match="branch must be"):
            lora.Adapter(tmp_path / "x.safetensors", "both")

    def test_a_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lora.Adapter(tmp_path / "nope.safetensors", "nar")

    def test_a_delta_that_does_not_fit_is_rejected(self, model, tmp_path):
        path = tmp_path / "bad.safetensors"
        save_file({"llm2vae.diff": torch.zeros(3, 3)}, str(path))
        with pytest.raises(ValueError, match="does not fit"):
            lora.deltas(model, lora.Adapter(path, "nar"))

    def test_a_target_the_model_lacks_is_rejected(self, model, tmp_path):
        path = tmp_path / "bad.safetensors"
        save_file({"model.layers.0.ghost_proj.lora_down.weight": torch.zeros(2, 4),
                   "model.layers.0.ghost_proj.lora_up.weight": torch.zeros(4, 2)}, str(path))
        with pytest.raises(KeyError, match="does not have"):
            lora.deltas(model, lora.Adapter(path, "nar"))

    def test_an_adapter_contributing_nothing_is_rejected(self, model, tmp_path):
        path = make_adapter(tmp_path, model, branch="nar")
        with pytest.raises(ValueError, match="contributed nothing"):
            lora.deltas(model, lora.Adapter(path, "ar"))
