use ds_ai::{
    Api, ModelCompatibility, ModelInput, ProviderId, ThinkingLevel, anthropic_models,
    builtin_anthropic_model, builtin_catalog_info, builtin_codex_model, builtin_model,
    builtin_models, builtin_openai_model, builtin_provider_models, builtin_providers, codex_models,
    openai_models, validate_builtin_catalog, validate_model_catalog,
};

#[test]
fn loads_and_validates_the_pinned_catalogs() {
    validate_builtin_catalog().unwrap();
    assert_eq!(openai_models().len(), 38);
    assert_eq!(anthropic_models().len(), 13);
    assert_eq!(codex_models().len(), 7);
    assert_eq!(builtin_provider_models("missing"), []);

    let info = builtin_catalog_info();
    assert_eq!(
        info.source_commit,
        "5c6655e76e07996e53ebc45226ec4dc32b79323b"
    );
    assert_eq!(info.generated_at, "2026-08-25T10:34:47.107Z");
    assert_eq!(info.files.len(), 3);
    assert_eq!(
        info.files["anthropic.json"],
        "4d44dd3f78d21dd1d0fba878bd58026914ba98dba3083212059b4710dc13eeb3"
    );
    assert_eq!(
        info.files["openai-codex.json"],
        "2712c2924a4a75213dddc743c0e5f08d50a781fe807f16d5afb5fb65b41c64c7"
    );
    assert_eq!(
        info.files["openai.json"],
        "2f32f5796138f03153a0314edf4c8ee27531d62f3477de0da46e51095f5e4782"
    );

    for models in [openai_models(), anthropic_models(), codex_models()] {
        assert!(models.windows(2).all(|pair| pair[0].id < pair[1].id));
    }
}

#[test]
fn returns_api_typed_catalog_models() {
    assert_eq!(
        builtin_openai_model("gpt-5.6-sol").unwrap().api,
        Api::OpenAiResponses
    );
    assert_eq!(
        builtin_anthropic_model("claude-sonnet-5").unwrap().api,
        Api::AnthropicMessages
    );
    assert_eq!(
        builtin_codex_model("gpt-5.6-luna").unwrap().api,
        Api::OpenAiCodexResponses
    );
}

#[test]
fn exposes_catalog_capabilities_costs_and_compatibility() {
    let openai = builtin_model("openai", "gpt-5.6-sol").unwrap();
    assert_eq!(openai.api, Api::OpenAiResponses);
    assert_eq!(openai.context_window, 272_000);
    assert_eq!(openai.max_tokens, 128_000);
    assert_eq!(openai.input, [ModelInput::Text, ModelInput::Image]);
    assert_eq!(
        openai.thinking_level_map.get(&ThinkingLevel::Max),
        Some(&Some("max".into()))
    );
    assert_eq!(openai.cost.tiers[0].input_tokens_above, 272_000);
    assert_eq!(openai.cost.rates.input, 4.0);
    assert_eq!(openai.cost.rates.output, 20.0);
    assert_eq!(openai.cost.rates.cache_read, 0.4);
    assert_eq!(openai.cost.rates.cache_write, 5.0);
    assert_eq!(openai.cost.tiers[0].rates.input, 8.0);
    assert_eq!(openai.cost.tiers[0].rates.output, 30.0);
    assert_eq!(openai.cost.tiers[0].rates.cache_read, 0.8);
    assert_eq!(openai.cost.tiers[0].rates.cache_write, 10.0);
    assert!(matches!(openai.compat, Some(ModelCompatibility::OpenAi(_))));

    let anthropic = builtin_model("anthropic", "claude-opus-5").unwrap();
    let Some(ModelCompatibility::Anthropic(compat)) = anthropic.compat else {
        panic!("expected Anthropic compatibility");
    };
    assert_eq!(compat.supports_temperature, Some(false));
    assert_eq!(compat.allowed_fallback_models[0].model, "claude-opus-4-8");

    let codex = builtin_model("openai-codex", "gpt-5.6-luna").unwrap();
    assert_eq!(codex.api, Api::OpenAiCodexResponses);
    assert_eq!(codex.provider, ProviderId::new("openai-codex"));
    assert_eq!(codex.cost.rates.cache_write, 0.25);
}

#[test]
fn matches_selected_catalog_thinking_level_matrices() {
    use ThinkingLevel::*;

    let cases = [
        (
            "anthropic",
            "claude-opus-4-6",
            &[Off, Minimal, Low, Medium, High, Max][..],
        ),
        (
            "anthropic",
            "claude-opus-4-8",
            &[Off, Minimal, Low, Medium, High, XHigh, Max][..],
        ),
        (
            "anthropic",
            "claude-opus-5",
            &[Off, Minimal, Low, Medium, High, XHigh, Max][..],
        ),
        (
            "anthropic",
            "claude-sonnet-4-6",
            &[Off, Minimal, Low, Medium, High, Max][..],
        ),
        (
            "anthropic",
            "claude-sonnet-5",
            &[Off, Minimal, Low, Medium, High, XHigh, Max][..],
        ),
        (
            "anthropic",
            "claude-fable-5",
            &[Minimal, Low, Medium, High, XHigh, Max][..],
        ),
        (
            "anthropic",
            "claude-sonnet-4-5",
            &[Off, Minimal, Low, Medium, High][..],
        ),
        (
            "openai-codex",
            "gpt-5.4",
            &[Off, Minimal, Low, Medium, High, XHigh][..],
        ),
        (
            "openai-codex",
            "gpt-5.5",
            &[Off, Minimal, Low, Medium, High, XHigh][..],
        ),
        (
            "openai-codex",
            "gpt-5.6-sol",
            &[Off, Minimal, Low, Medium, High, XHigh, Max][..],
        ),
        ("openai", "gpt-5.5-pro", &[Medium, High, XHigh][..]),
        (
            "openai",
            "gpt-5.6-sol",
            &[Off, Low, Medium, High, XHigh, Max][..],
        ),
        (
            "openai",
            "gpt-5.6-terra",
            &[Off, Low, Medium, High, XHigh, Max][..],
        ),
        (
            "openai",
            "gpt-5.6-luna",
            &[Off, Low, Medium, High, XHigh, Max][..],
        ),
    ];

    for (provider, id, expected) in cases {
        let model = builtin_model(provider, id).unwrap();
        assert_eq!(
            model.supported_thinking_levels(),
            expected,
            "{provider}/{id}"
        );
    }
}

#[test]
fn built_in_providers_own_their_catalogs() {
    let providers = builtin_providers();
    assert_eq!(providers.len(), 3);
    assert_eq!(providers[0].id().as_str(), "openai");
    assert_eq!(providers[0].name(), "OpenAI");
    assert_eq!(providers[0].models(), openai_models());
    assert_eq!(providers[1].id().as_str(), "anthropic");
    assert_eq!(providers[1].models(), anthropic_models());
    assert_eq!(providers[2].id().as_str(), "openai-codex");
    assert_eq!(providers[2].models(), codex_models());

    let models = builtin_models();
    assert_eq!(models.providers().len(), 3);
    assert_eq!(models.models(None).len(), 58);
    assert!(models.model("openai", "gpt-5.6-sol").is_some());
}

#[test]
fn catalog_validation_rejects_invalid_identity_limits_costs_and_capabilities() {
    let valid = builtin_model("openai", "gpt-5.6-sol").unwrap();
    for invalid in [
        {
            let mut model = valid.clone();
            model.id.clear();
            model
        },
        {
            let mut model = valid.clone();
            model.provider = ProviderId::new("wrong");
            model
        },
        {
            let mut model = valid.clone();
            model.max_tokens = model.context_window + 1;
            model
        },
        {
            let mut model = valid.clone();
            model.cost.rates.input = f64::NAN;
            model
        },
        {
            let mut model = valid.clone();
            model.input = vec![ModelInput::Image];
            model
        },
        {
            let mut model = valid.clone();
            model.compat = Some(ModelCompatibility::Anthropic(Default::default()));
            model
        },
    ] {
        assert!(validate_model_catalog(&[invalid], &Api::OpenAiResponses, "openai").is_err());
    }
    assert!(
        validate_model_catalog(&[valid.clone(), valid], &Api::OpenAiResponses, "openai").is_err()
    );
}
