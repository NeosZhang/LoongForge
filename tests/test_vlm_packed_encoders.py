# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

import re
from types import SimpleNamespace

import pytest

from loongforge.data.multimodal import vlm_task_encoder


def _make_case(kind):
    common = {
        "__key__": "pack",
        "__restore_key__": (),
        "__subflavors__": {},
    }
    if kind == "vqa":
        sample = SimpleNamespace(
            **common,
            images=[object()],
            answers=["answer"],
            contexts=["question"],
        )
        return (
            vlm_task_encoder.VLMTaskEncoder.encode_packed_vqa,
            "encode_vqa",
            "VQASample",
            sample,
            "pack.img000_jpg",
        )

    sample = SimpleNamespace(
        **common,
        images=None,
        videos=None,
        answers=[["answer"]],
        contexts=[["question"]],
    )
    return (
        vlm_task_encoder.VLMTaskEncoder.encode_packed_multi_mix_qa,
        "encode_multi_mix_qa",
        "MultiMixQASample",
        sample,
        "pack.q000",
    )


@pytest.mark.parametrize("kind", ["vqa", "multi_mix"])
def test_packed_member_encoding_starts_in_packing_mode(monkeypatch, kind):
    method, encoder_name, sample_type, sample, member_key = _make_case(kind)
    monkeypatch.setattr(vlm_task_encoder, "_ENERGON_NEEDS_SUBFLAVOR", False)
    monkeypatch.setattr(vlm_task_encoder, sample_type, SimpleNamespace)

    encoded_member = object()
    packed_sample = object()
    encoder = SimpleNamespace(is_packing_enabled=False)

    def encode_member(member):
        assert encoder.is_packing_enabled is True
        assert member.__key__ == member_key
        return encoded_member

    def pack_members(members):
        assert members == [encoded_member]
        return packed_sample

    setattr(encoder, encoder_name, encode_member)
    encoder.pack_selected_samples = pack_members

    assert method(encoder, sample) is packed_sample


@pytest.mark.parametrize("kind", ["vqa", "multi_mix"])
def test_packed_member_rejects_nullable_encoder_result(monkeypatch, kind):
    method, encoder_name, sample_type, sample, member_key = _make_case(kind)
    monkeypatch.setattr(vlm_task_encoder, "_ENERGON_NEEDS_SUBFLAVOR", False)
    monkeypatch.setattr(vlm_task_encoder, sample_type, SimpleNamespace)

    encoder = SimpleNamespace(is_packing_enabled=False)
    setattr(encoder, encoder_name, lambda member: None)
    encoder.pack_selected_samples = lambda members: pytest.fail(
        "packer must not receive a dropped member"
    )

    with pytest.raises(ValueError, match=re.escape(member_key)):
        method(encoder, sample)
