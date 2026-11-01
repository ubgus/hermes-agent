"""The smart-approval guardian must never be routed by guesswork.

``approval`` gates a trust decision, so its auxiliary call may only use the
operator's explicit ``auxiliary.approval`` provider and that task's own
``fallback_chain``. Main-model, discovery, and payment fallbacks are forbidden;
exhaustion reaches the normal unavailable/error path, which
``tools.approval_smart`` already turns into an ESCALATE verdict.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _fail_if_called(name):
    def _fail(*_args, **_kwargs):
        pytest.fail(f"{name} must not be used for task='approval'")

    return _fail


def test_approval_auto_route_refuses_main_model_and_discovery():
    from agent.auxiliary_client import _resolve_auto_route

    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"),
        patch(
            "agent.auxiliary_client._read_main_model", return_value="paid-main-model"
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
        patch(
            "agent.auxiliary_client._try_main_provider_route",
            side_effect=_fail_if_called("_try_main_provider_route"),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_discovery_chain",
            side_effect=_fail_if_called("_try_discovery_chain"),
        ),
    ):
        assert _resolve_auto_route(task="approval") == (None, None, "")


def test_approval_auto_route_uses_only_the_task_fallback_chain():
    from agent.auxiliary_client import _resolve_auto_route

    task_client = MagicMock()
    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"),
        patch(
            "agent.auxiliary_client._read_main_model", return_value="paid-main-model"
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(task_client, "guard-model", "fallback_chain[0](local-guard)"),
        ),
        patch(
            "agent.auxiliary_client._fallback_provider_from_label",
            return_value="local-guard",
        ),
        patch(
            "agent.auxiliary_client._try_main_provider_route",
            side_effect=_fail_if_called("_try_main_provider_route"),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_discovery_chain",
            side_effect=_fail_if_called("_try_discovery_chain"),
        ),
    ):
        client, model, provider = _resolve_auto_route(task="approval")

    assert client is task_client
    assert (model, provider) == ("guard-model", "local-guard")


def test_approval_quarantine_walk_stops_at_the_task_chain():
    from agent.auxiliary_client import _next_fallback_after_quarantine

    route = MagicMock(base_info="https://guard.example/v1")
    with (
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ) as task_chain,
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=_fail_if_called("_try_payment_fallback"),
        ),
    ):
        fb = _next_fallback_after_quarantine(
            "approval", "auto", True, route, None, None
        )

    assert fb == (None, None, "")
    task_chain.assert_called_once()


def test_approval_recovery_ladder_does_not_escape_to_main_or_discovery():
    from agent.auxiliary_client import _LadderRoute, _ladder_provider_fallback

    route = _LadderRoute(
        client=MagicMock(),
        task="approval",
        tag="",
        async_mode=False,
        base_info="https://guard.example/v1",
        resolved_provider="openrouter",
        resolved_model="guard-model",
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model="guard-model",
        main_runtime=None,
        route_info=None,
        timeout=30,
    )
    with (
        patch("agent.auxiliary_client._is_auth_error", return_value=False),
        patch("agent.auxiliary_client._is_payment_error", return_value=True),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ) as task_chain,
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_payment_fallback",
            side_effect=_fail_if_called("_try_payment_fallback"),
        ),
        patch(
            "agent.auxiliary_client._try_main_agent_model_fallback",
            side_effect=_fail_if_called("_try_main_agent_model_fallback"),
        ),
    ):
        assert list(_ladder_provider_fallback(RuntimeError("402 credits"), route)) == []

    task_chain.assert_called_once()


def test_approval_fallback_chain_skips_auto_sentinel():
    from agent.auxiliary_client import _resolve_auto_route

    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"),
        patch(
            "agent.auxiliary_client._read_main_model", return_value="paid-main-model"
        ),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={
                "fallback_chain": [{"provider": "auto", "model": "paid-main-model"}]
            },
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            side_effect=_fail_if_called("resolve_provider_client"),
        ) as resolve_client,
        patch(
            "agent.auxiliary_client._try_main_provider_route",
            side_effect=_fail_if_called("_try_main_provider_route"),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_discovery_chain",
            side_effect=_fail_if_called("_try_discovery_chain"),
        ),
    ):
        assert _resolve_auto_route(task="approval") == (None, None, "")

    resolve_client.assert_not_called()


@pytest.mark.parametrize("sentinel", ["main", "custom:auto", "custom:main", "custom:"])
def test_approval_fallback_chain_skips_sentinel_aliases(sentinel):
    """``main``/``custom:auto`` normalize back into main-model routing; a fallback entry
    must name a concrete provider, not an inheritance sentinel."""
    from agent.auxiliary_client import _try_configured_fallback_chain

    with (
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={
                "fallback_chain": [{"provider": sentinel, "model": "paid-main-model"}]
            },
        ),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            side_effect=_fail_if_called("_resolve_fallback_entry"),
        ) as resolve_entry,
    ):
        assert _try_configured_fallback_chain("approval", "auto") == (
            None,
            None,
            "",
        )

    resolve_entry.assert_not_called()


@pytest.mark.parametrize("sentinel", ["main", "custom:auto", "custom:main"])
def test_approval_task_config_refuses_sentinel_primary_route(sentinel):
    """A task config naming a sentinel must resolve as bare auto (policy-limited), not as
    the main provider or a concrete custom route."""
    from agent.auxiliary_client import _resolve_task_provider_model

    with patch(
        "agent.auxiliary_client._get_auxiliary_task_config",
        return_value={
            "provider": sentinel,
            "model": "paid-main-model",
            "base_url": "https://main.example/v1",
            "api_key": "sk-main",
        },
    ):
        assert _resolve_task_provider_model("approval") == (
            "auto",
            None,
            None,
            None,
            None,
        )


@pytest.mark.parametrize("sentinel", ["main", "custom:auto", "custom:main"])
def test_resolve_provider_client_rejects_sentinel_for_approval(sentinel):
    from agent.auxiliary_client import resolve_provider_client

    with patch(
        "agent.auxiliary_client._resolve_auto_route",
        side_effect=_fail_if_called("_resolve_auto_route"),
    ):
        assert resolve_provider_client(sentinel, model="guard", task="approval") == (
            None,
            None,
        )


def test_approval_custom_route_requires_explicit_base_url():
    """Bare ``custom`` must not borrow the live main runtime or fall into discovery."""
    from agent.auxiliary_client import resolve_provider_client

    runtime = {
        "provider": "custom",
        "model": "main-model",
        "base_url": "https://main.example/v1",
        "api_key": "sk-main",
    }
    with (
        patch(
            "agent.auxiliary_client._create_openai_client",
            side_effect=_fail_if_called("_create_openai_client"),
        ),
        patch(
            "agent.auxiliary_client._try_custom_endpoint",
            side_effect=_fail_if_called("_try_custom_endpoint"),
        ),
        patch(
            "agent.auxiliary_client._resolve_api_key_provider",
            side_effect=_fail_if_called("_resolve_api_key_provider"),
        ),
    ):
        assert resolve_provider_client(
            "custom", model="guard", task="approval", main_runtime=runtime
        ) == (None, None)


def test_approval_custom_route_does_not_borrow_main_key():
    """Even with an explicit base_url, a fail-closed task must not reuse the main key."""
    from agent.auxiliary_client import resolve_provider_client

    with (
        patch("agent.auxiliary_client._scoped_key_env", return_value=""),
        patch(
            "agent.auxiliary_client._read_main_api_key_if_same_origin",
            side_effect=_fail_if_called("_read_main_api_key_if_same_origin"),
        ),
        patch(
            "agent.auxiliary_client._create_openai_client",
            return_value=MagicMock(),
        ) as create_client,
    ):
        resolve_provider_client(
            "custom",
            model="guard",
            explicit_base_url="https://guard.example/v1",
            task="approval",
        )

    assert create_client.call_args.kwargs["api_key"] == "no-key-required"


def test_get_text_auxiliary_client_passes_task_policy():
    """The text helper must forward ``task`` or the resolver cannot enforce approval policy."""
    from agent import auxiliary_client as aux

    with (
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", None, None, None, None),
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ) as resolve_client,
    ):
        assert aux.get_text_auxiliary_client("approval") == (None, None)

    assert resolve_client.call_args.kwargs["task"] == "approval"


def test_approval_fallback_may_use_the_untried_main_provider():
    """The main provider was never tried, so an explicit fallback entry naming it is not a
    failed backend and must still be used."""
    from agent.auxiliary_client import _resolve_auto_route

    guard = MagicMock()
    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"),
        patch(
            "agent.auxiliary_client._read_main_model", return_value="paid-main-model"
        ),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={
                "fallback_chain": [{"provider": "openrouter", "model": "guard-model"}]
            },
        ),
        patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            return_value=(guard, "guard-model"),
        ) as resolve_entry,
        patch(
            "agent.auxiliary_client._fallback_provider_from_label",
            return_value="openrouter",
        ),
        patch(
            "agent.auxiliary_client._try_main_provider_route",
            side_effect=_fail_if_called("_try_main_provider_route"),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_discovery_chain",
            side_effect=_fail_if_called("_try_discovery_chain"),
        ),
    ):
        assert _resolve_auto_route(task="approval") == (
            guard,
            "guard-model",
            "openrouter",
        )

    resolve_entry.assert_called_once()


def test_approval_cache_does_not_reuse_a_non_fail_closed_client():
    """A client warmed by an ordinary task must never answer a trust decision."""
    from agent import auxiliary_client as aux

    ordinary, guarded = MagicMock(name="ordinary"), MagicMock(name="guarded")
    aux._client_cache.clear()
    try:
        with (
            patch("agent.auxiliary_client._pool_cache_hint", return_value=""),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                side_effect=[(ordinary, "guard-model"), (guarded, "guard-model")],
            ) as resolve_client,
        ):
            assert aux._get_cached_client("openrouter", "guard-model")[0] is ordinary
            assert (
                aux._get_cached_client("openrouter", "guard-model", task="approval")[0]
                is guarded
            )

        assert ordinary is not guarded
        assert resolve_client.call_count == 2
        assert aux._client_cache_key(
            "openrouter", async_mode=False, task="approval", model="guard-model"
        ) != aux._client_cache_key("openrouter", async_mode=False, model="guard-model")
    finally:
        aux._client_cache.clear()


def test_approval_cache_policy_is_checked_before_lookup():
    """A sentinel provider is refused without even computing a cache key."""
    from agent import auxiliary_client as aux

    with patch(
        "agent.auxiliary_client._client_cache_key",
        side_effect=_fail_if_called("_client_cache_key"),
    ):
        assert aux._get_cached_client("main", "guard-model", task="approval") == (
            None,
            None,
        )


@pytest.mark.parametrize("aggregator", ["main", "auto", "custom:auto", "custom:main"])
def test_approval_moa_aggregator_is_revalidated(aggregator):
    """MoA expansion must not smuggle main/auto back in as the guardian's route."""
    from agent.auxiliary_client import resolve_provider_client

    with (
        patch(
            "agent.auxiliary_client._resolve_moa_aggregator",
            return_value=(aggregator, "aggregator-model"),
        ),
    ):
        assert resolve_provider_client("moa", model="preset", task="approval") == (
            None,
            None,
        )


def test_approval_task_config_rejects_sentinel_moa_aggregator():
    from agent.auxiliary_client import _resolve_task_provider_model

    with (
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"provider": "moa", "model": "preset"},
        ),
        patch(
            "agent.auxiliary_client._resolve_moa_aggregator",
            return_value=("main", "aggregator-model"),
        ),
    ):
        assert _resolve_task_provider_model("approval") == (
            "auto",
            None,
            None,
            None,
            None,
        )


def test_approval_named_provider_uses_its_own_model():
    from agent.auxiliary_client import resolve_provider_client

    with (
        patch("agent.auxiliary_client._get_aux_model_for_provider", return_value=""),
        patch(
            "agent.auxiliary_client._read_main_model_for_aux",
            side_effect=_fail_if_called("_read_main_model_for_aux"),
        ),
        patch(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            return_value={
                "base_url": "https://guard.example/v1",
                "model": "guard-model",
            },
        ),
        patch(
            "agent.auxiliary_client._named_custom_api_key",
            return_value="sk-guard",
        ),
        patch(
            "agent.auxiliary_client._create_openai_client",
            return_value=MagicMock(),
        ),
        patch(
            "agent.auxiliary_client._wrap_transport",
            side_effect=lambda _req, client, *_args: client,
        ),
    ):
        _client, model = resolve_provider_client("custom:guard", task="approval")

    assert model == "guard-model"


def test_approval_named_provider_without_model_does_not_inherit_main():
    from agent.auxiliary_client import resolve_provider_client

    with (
        patch("agent.auxiliary_client._get_aux_model_for_provider", return_value=""),
        patch(
            "agent.auxiliary_client._read_main_model_for_aux",
            side_effect=_fail_if_called("_read_main_model_for_aux"),
        ),
        patch(
            "hermes_cli.runtime_provider._get_named_custom_provider",
            return_value={"base_url": "https://guard.example/v1"},
        ),
        patch(
            "agent.auxiliary_client._create_openai_client",
            side_effect=_fail_if_called("_create_openai_client"),
        ),
    ):
        assert resolve_provider_client("custom:guard", task="approval") == (
            None,
            None,
        )


def test_approval_custom_route_without_model_does_not_inherit_main():
    from agent.auxiliary_client import resolve_provider_client

    with (
        patch("agent.auxiliary_client._get_aux_model_for_provider", return_value=""),
        patch(
            "agent.auxiliary_client._read_main_model_for_aux",
            side_effect=_fail_if_called("_read_main_model_for_aux"),
        ),
        patch(
            "agent.auxiliary_client._create_openai_client",
            side_effect=_fail_if_called("_create_openai_client"),
        ),
    ):
        assert resolve_provider_client(
            "custom",
            explicit_base_url="https://guard.example/v1",
            task="approval",
        ) == (None, None)


def test_approval_azure_foundry_rejects_unset_model_before_default_deployment():
    """Azure's ``model.default`` is the main deployment; a trust-gating task must not inherit it."""
    from agent.auxiliary_client import resolve_provider_client

    with (
        patch("agent.auxiliary_client._get_aux_model_for_provider", return_value=""),
        patch(
            "agent.auxiliary_client._read_main_model_for_aux",
            side_effect=_fail_if_called("_read_main_model_for_aux"),
        ),
        patch(
            "agent.auxiliary_client._resolve_azure_foundry_branch",
            side_effect=_fail_if_called("_resolve_azure_foundry_branch"),
        ),
        patch(
            "agent.auxiliary_client._try_azure_foundry",
            side_effect=_fail_if_called("_try_azure_foundry"),
        ),
    ):
        assert resolve_provider_client("azure-foundry", task="approval") == (
            None,
            None,
        )


def test_ordinary_azure_foundry_still_uses_its_configured_default():
    """The fail-closed model guard is scoped to the task, not to azure-foundry itself."""
    from agent.auxiliary_client import resolve_provider_client

    client = MagicMock()
    with (
        patch("agent.auxiliary_client._get_aux_model_for_provider", return_value=""),
        patch(
            "agent.auxiliary_client._try_azure_foundry",
            return_value=(client, "gpt-4o"),
        ) as try_azure,
    ):
        assert resolve_provider_client("azure-foundry") == (client, "gpt-4o")

    try_azure.assert_called_once()


def test_approval_openrouter_configuration_is_preserved():
    """OpenRouter resolves its own explicit ``auxiliary.openrouter_model``; the Azure guard
    must not turn an unset caller model into a blanket rejection."""
    from agent.auxiliary_client import resolve_provider_client

    client = MagicMock()
    with (
        patch("agent.auxiliary_client._get_aux_model_for_provider", return_value=""),
        patch(
            "agent.auxiliary_client._try_openrouter",
            return_value=(client, "guard-model"),
        ) as try_openrouter,
    ):
        assert resolve_provider_client("openrouter", task="approval") == (
            client,
            "guard-model",
        )

    try_openrouter.assert_called_once()


class _GuardHttpError(RuntimeError):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


@pytest.mark.parametrize("provider", ["custom", "custom:guard"])
@pytest.mark.parametrize("status", [401, 402])
def test_approval_custom_guard_on_nous_host_skips_global_nous_recovery(
    provider, status
):
    """A custom endpoint that happens to share the Nous hostname keeps its own credentials:
    the 401/payment ladder must not refresh or rotate the global Nous credential."""
    from agent.auxiliary_client import _aux_recovery_ladder, _drive_ladder

    err = _GuardHttpError(status)
    client = MagicMock()
    client.base_url = "https://inference-api.nousresearch.com/v1"
    client.api_key = "sk-guard"
    recovered: list[str] = []
    marked: list[str] = []

    def _record_recovery(candidate, *_args, **_kwargs):
        recovered.append(candidate)
        return False

    with (
        patch(
            "agent.auxiliary_client._refresh_nous_auxiliary_client",
            side_effect=_fail_if_called("_refresh_nous_auxiliary_client"),
        ),
        patch(
            "agent.auxiliary_client._nous_portal_account_has_fresh_paid_access",
            side_effect=_fail_if_called("_nous_portal_account_has_fresh_paid_access"),
        ),
        patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            return_value=False,
        ),
        patch(
            "agent.auxiliary_client._recover_provider_pool",
            side_effect=_record_recovery,
        ),
        patch(
            "agent.auxiliary_client._mark_provider_unhealthy",
            side_effect=lambda candidate, **_kwargs: marked.append(candidate),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
    ):
        ladder = _aux_recovery_ladder(
            err,
            client=client,
            kwargs={"model": "guard-model"},
            task="approval",
            async_mode=False,
            base_info="https://inference-api.nousresearch.com/v1",
            resolved_provider=provider,
            resolved_model="guard-model",
            resolved_base_url=None,
            resolved_api_key=None,
            resolved_api_mode=None,
            final_model="guard-model",
            max_tokens=None,
            main_runtime=None,
            route_info=None,
        )
        with pytest.raises(_GuardHttpError):
            _drive_ladder(ladder, lambda _step: (_ for _ in ()).throw(err))

    assert "nous" not in recovered
    assert "nous" not in marked


def test_route_is_nous_keeps_ordinary_host_inference():
    """Only fail-closed tasks opt out of Nous recovery."""
    from agent.auxiliary_client import _LadderRoute, _route_is_nous

    route = _LadderRoute(
        client=MagicMock(),
        task="compression",
        tag="",
        async_mode=False,
        base_info="https://inference-api.nousresearch.com/v1",
        resolved_provider="custom",
        resolved_model="m",
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model="m",
        main_runtime=None,
        route_info=None,
        timeout=30,
    )
    assert _route_is_nous(route) is True


_NOUS_GUARD_BASE = "https://inference-api.nousresearch.com/v1"


def _assert_approval_ladder_skips_global_nous(client, provider, status):
    from agent.auxiliary_client import _aux_recovery_ladder, _drive_ladder

    err = _GuardHttpError(status)
    recovered: list[str] = []
    marked: list[str] = []

    with (
        patch(
            "agent.auxiliary_client._refresh_nous_auxiliary_client",
            side_effect=_fail_if_called("_refresh_nous_auxiliary_client"),
        ),
        patch(
            "agent.auxiliary_client._nous_portal_account_has_fresh_paid_access",
            side_effect=_fail_if_called("_nous_portal_account_has_fresh_paid_access"),
        ),
        patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            return_value=False,
        ),
        patch(
            "agent.auxiliary_client._recover_provider_pool",
            side_effect=lambda candidate, *_a, **_k: recovered.append(candidate) or False,
        ),
        patch(
            "agent.auxiliary_client._mark_provider_unhealthy",
            side_effect=lambda candidate, **_k: marked.append(candidate),
        ),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
    ):
        ladder = _aux_recovery_ladder(
            err,
            client=client,
            kwargs={"model": "guard-model"},
            task="approval",
            async_mode=False,
            base_info=client.base_url,
            resolved_provider=provider,
            resolved_model="guard-model",
            resolved_base_url=None,
            resolved_api_key=None,
            resolved_api_mode=None,
            final_model="guard-model",
            max_tokens=None,
            main_runtime=None,
            route_info=None,
        )
        with pytest.raises(_GuardHttpError):
            _drive_ladder(ladder, lambda _step: (_ for _ in ()).throw(err))

    assert "nous" not in recovered
    assert "nous" not in marked


@pytest.mark.parametrize("status", [401, 402])
def test_approval_fallback_only_custom_guardian_on_nous_host_skips_global_nous_recovery(
    status,
):
    """A guardian configured only in the task fallback chain keeps its own credentials even
    when its custom endpoint is the Nous host."""
    from agent.auxiliary_client import _resolve_auto_route

    client = MagicMock()
    client.base_url = _NOUS_GUARD_BASE
    client.api_key = "sk-guard"
    with (
        patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"),
        patch("agent.auxiliary_client._read_main_model", return_value="paid-main-model"),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={
                "fallback_chain": [
                    {
                        "provider": "custom",
                        "model": "guard-model",
                        "base_url": _NOUS_GUARD_BASE,
                    }
                ]
            },
        ),
        patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False),
        patch("agent.auxiliary_client._task_minimum_context_length", return_value=None),
        patch(
            "agent.auxiliary_client._resolve_fallback_entry",
            return_value=(client, "guard-model"),
        ),
        patch(
            "agent.auxiliary_client._try_main_provider_route",
            side_effect=_fail_if_called("_try_main_provider_route"),
        ),
        patch(
            "agent.auxiliary_client._try_main_fallback_chain",
            side_effect=_fail_if_called("_try_main_fallback_chain"),
        ),
        patch(
            "agent.auxiliary_client._try_discovery_chain",
            side_effect=_fail_if_called("_try_discovery_chain"),
        ),
    ):
        resolved = _resolve_auto_route(task="approval")

    assert resolved == (client, "guard-model", "custom")
    _assert_approval_ladder_skips_global_nous(client, "custom", status)


@pytest.mark.parametrize("status", [401, 402])
def test_approval_bare_named_guardian_on_nous_host_skips_global_nous_recovery(status):
    """A bare named custom provider on the Nous host is still an operator-supplied guardian,
    not the global Nous Portal."""
    client = MagicMock()
    client.base_url = _NOUS_GUARD_BASE
    client.api_key = "sk-guard"

    _assert_approval_ladder_skips_global_nous(client, "guard", status)



@pytest.mark.parametrize("provider", ["xai-oauth", "nous", "custom:guard", "guard"])
@pytest.mark.parametrize("status", [401, 402])
def test_approval_explicit_provider_skips_generic_refresh_and_pool_recovery(
    provider, status
):
    """An explicitly resolved provider route is not licence to refresh its OAuth grant or
    rotate a global credential pool: approval may only spend the route's own credentials."""
    from agent.auxiliary_client import _aux_recovery_ladder, _drive_ladder

    err = _GuardHttpError(status)
    client = MagicMock()
    client.base_url = "https://api.example.invalid/v1"
    client.api_key = "sk-guard"

    with (
        patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            side_effect=_fail_if_called("_refresh_provider_credentials"),
        ) as refresh,
        patch(
            "agent.auxiliary_client._recover_provider_pool",
            side_effect=_fail_if_called("_recover_provider_pool"),
        ) as recover_pool,
        patch(
            "agent.auxiliary_client._refresh_nous_auxiliary_client",
            side_effect=_fail_if_called("_refresh_nous_auxiliary_client"),
        ),
        patch(
            "agent.auxiliary_client._nous_portal_account_has_fresh_paid_access",
            side_effect=_fail_if_called("_nous_portal_account_has_fresh_paid_access"),
        ),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        patch(
            "agent.auxiliary_client._try_configured_fallback_chain",
            return_value=(None, None, ""),
        ),
    ):
        ladder = _aux_recovery_ladder(
            err,
            client=client,
            kwargs={"model": "guard-model"},
            task="approval",
            async_mode=False,
            base_info=client.base_url,
            resolved_provider=provider,
            resolved_model="guard-model",
            resolved_base_url=None,
            resolved_api_key=None,
            resolved_api_mode=None,
            final_model="guard-model",
            max_tokens=None,
            main_runtime=None,
            route_info=None,
        )
        with pytest.raises(_GuardHttpError):
            _drive_ladder(ladder, lambda _step: (_ for _ in ()).throw(err))

    refresh.assert_not_called()
    recover_pool.assert_not_called()


def test_approval_fallback_auth_retry_does_not_refresh_credentials():
    """A dead fallback candidate is quarantined; approval never refreshes its OAuth grant."""
    from agent.auxiliary_client import _FallbackDestination, _plan_fallback_auth_retry

    destination = _FallbackDestination(
        "xai-oauth", "https://api.example.invalid/v1", "chat_completions", "guard-model"
    )
    with (
        patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            side_effect=_fail_if_called("_refresh_provider_credentials"),
        ) as refresh,
        patch(
            "agent.auxiliary_client._get_cached_client",
            side_effect=_fail_if_called("_get_cached_client"),
        ),
    ):
        assert _plan_fallback_auth_retry(
            destination,
            lambda *_args: pytest.fail("rebuild must not run without a refresh"),
            async_mode=False,
            task="approval",
        ) == ("xai-oauth", None)

    refresh.assert_not_called()


def test_ordinary_fallback_auth_retry_still_refreshes_credentials():
    """Non-approval tasks keep the refresh-and-retry fallback behaviour."""
    from agent.auxiliary_client import _FallbackDestination, _plan_fallback_auth_retry

    destination = _FallbackDestination(
        "xai-oauth", "https://api.example.invalid/v1", "chat_completions", "guard-model"
    )
    rebuilt = _FallbackDestination(
        "xai-oauth", "https://api.example.invalid/v1", "chat_completions", "guard-model"
    )
    client = MagicMock()

    with (
        patch(
            "agent.auxiliary_client._refresh_provider_credentials",
            return_value=True,
        ) as refresh,
        patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "guard-model"),
        ),
    ):
        provider, retry = _plan_fallback_auth_retry(
            destination,
            lambda p, c, m: (rebuilt, {"model": m}),
            async_mode=False,
            task="compression",
        )

    refresh.assert_called_once_with("xai-oauth")
    assert provider == "xai-oauth"
    assert retry == (client, {"model": "guard-model"}, rebuilt)


@pytest.mark.parametrize(
    "provider,base_info",
    [
        ("nous", "https://api.example.invalid/v1"),
        ("custom:guard", _NOUS_GUARD_BASE),
    ],
)
def test_route_is_nous_refuses_approval_even_on_the_nous_host(provider, base_info):
    """Approval skips every Nous-only recovery rung, including an explicitly resolved pool."""
    from agent.auxiliary_client import _LadderRoute, _route_is_nous

    route = _LadderRoute(
        client=MagicMock(),
        task="approval",
        tag="",
        async_mode=False,
        base_info=base_info,
        resolved_provider=provider,
        resolved_model="guard-model",
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model="guard-model",
        main_runtime=None,
        route_info=None,
        timeout=30,
    )
    assert _route_is_nous(route) is False
