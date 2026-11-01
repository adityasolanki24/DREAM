from __future__ import annotations

import pytest

from cloud import tailnet


def test_join_commands_keep_the_key_out_and_disable_extras():
    (cmd,) = tailnet.join_commands("dream-isaac-1a2b3c")
    assert cmd[:2] == ["tailscale", "up"]
    assert "--accept-dns=false" in cmd and "--ssh=false" in cmd
    assert not any("tskey" in part for part in cmd)  # the key is appended only at run time


def test_hostname_must_be_a_dns_label():
    for bad in ["", "Dream", "dream_isaac", "-dream", "dream.isaac"]:
        with pytest.raises(ValueError):
            tailnet.join_commands(bad)


def test_upload_rejects_files_that_are_not_auth_keys(tmp_path):
    f = tmp_path / "k"
    f.write_text("not-a-key")
    with pytest.raises(ValueError):
        tailnet.upload_key("bucket", f)


def test_key_lives_only_in_the_session_bucket():
    assert (
        tailnet.key_uri("dream-qwen-isaac-x") == "gs://dream-qwen-isaac-x/secrets/tailscale_authkey"
    )
