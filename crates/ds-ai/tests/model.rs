use ds_ai::{
    AnthropicOptions, Api, Model, ModelCompatibility, ModelCost, ModelCostRates, ModelCostTier,
    ModelInput, OpenAiResponsesCompatibility, OpenAiResponsesOptions, ProviderId,
    SessionAffinityFormat, ThinkingLevel, Usage,
};
use std::collections::BTreeMap;

fn model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::OpenAiResponses,
        provider: ProviderId::new("test-provider"),
        base_url: "https://example.com/v1".into(),
        reasoning: true,
        thinking_level_map: BTreeMap::new(),
        input: vec![ModelInput::Text],
        cost: ModelCost {
            rates: ModelCostRates {
                input: 2.0,
                output: 8.0,
                cache_read: 0.5,
                cache_write: 2.5,
            },
            tiers: Vec::new(),
        },
        context_window: 200_000,
        max_tokens: 32_000,
        sampling_params: BTreeMap::new(),
        headers: BTreeMap::new(),
        compat: None,
    }
}

#[test]
fn binds_models_to_matching_api_options() {
    let model = model();
    let typed = model.typed::<OpenAiResponsesOptions>().unwrap();
    assert_eq!(typed.as_model(), &model);

    let error = model.typed::<AnthropicOptions>().unwrap_err();
    assert_eq!(error.actual, Api::OpenAiResponses);
    assert_eq!(error.expected, Api::AnthropicMessages);
}

#[test]
fn serializes_api_as_wire_identifier() {
    assert_eq!(
        serde_json::to_string(&Api::OpenAiCodexResponses).unwrap(),
        r#""openai-codex-responses""#
    );
    assert_eq!(
        serde_json::from_str::<Api>(r#""custom-api""#).unwrap(),
        Api::Other("custom-api".into())
    );
}

#[test]
fn serializes_model_compatibility_with_wire_names() {
    let mut model = model();
    model.compat = Some(ModelCompatibility::OpenAi(OpenAiResponsesCompatibility {
        session_affinity_format: Some(SessionAffinityFormat::OpenAiNoSession),
        supports_open_ai_grammar_tools: Some(true),
        ..Default::default()
    }));
    let value = serde_json::to_value(model).unwrap();

    assert_eq!(value["api"], "openai-responses");
    assert_eq!(value["provider"], "test-provider");
    assert_eq!(value["baseUrl"], "https://example.com/v1");
    assert_eq!(value["compat"]["sessionAffinityFormat"], "openai-nosession");
    assert_eq!(value["compat"]["supportsOpenAIGrammarTools"], true);
    assert_eq!(value["compat"].as_object().unwrap().len(), 2);
}

#[test]
fn lists_and_clamps_supported_thinking_levels() {
    let mut model = model();
    assert_eq!(
        model.supported_thinking_levels(),
        [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
        ]
    );

    model
        .thinking_level_map
        .insert(ThinkingLevel::Minimal, None);
    model
        .thinking_level_map
        .insert(ThinkingLevel::XHigh, Some("xhigh".into()));
    assert_eq!(
        model.supported_thinking_levels(),
        [
            ThinkingLevel::Off,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::XHigh,
        ]
    );
    assert_eq!(
        model.clamp_thinking_level(ThinkingLevel::Minimal),
        ThinkingLevel::Low
    );
    assert_eq!(
        model.clamp_thinking_level(ThinkingLevel::Max),
        ThinkingLevel::XHigh
    );

    model.reasoning = false;
    assert_eq!(model.supported_thinking_levels(), [ThinkingLevel::Off]);
    assert_eq!(
        model.clamp_thinking_level(ThinkingLevel::High),
        ThinkingLevel::Off
    );
}

#[test]
fn skips_a_missing_maximum_level_when_clamping() {
    let mut model = model();
    model.thinking_level_map.insert(ThinkingLevel::XHigh, None);
    model
        .thinking_level_map
        .insert(ThinkingLevel::Max, Some("max".into()));

    assert_eq!(
        model.supported_thinking_levels(),
        [
            ThinkingLevel::Off,
            ThinkingLevel::Minimal,
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High,
            ThinkingLevel::Max,
        ]
    );
    assert_eq!(
        model.clamp_thinking_level(ThinkingLevel::XHigh),
        ThinkingLevel::Max
    );
}

#[test]
fn compares_models_by_provider_and_id() {
    let model = model();
    let mut same = model.clone();
    same.api = Api::AnthropicMessages;
    assert!(model.is_same_as(&same));

    same.provider = ProviderId::new("other");
    assert!(!model.is_same_as(&same));
}

#[test]
fn calculates_base_tier_and_long_cache_costs() {
    let mut model = model();
    model.cost.tiers.push(ModelCostTier {
        rates: ModelCostRates {
            input: 4.0,
            output: 16.0,
            cache_read: 1.0,
            cache_write: 5.0,
        },
        input_tokens_above: 1_000,
    });

    let mut usage = Usage {
        input: 800,
        output: 100,
        cache_read: 200,
        cache_write: 100,
        cache_write_1h: Some(40),
        reasoning: Some(20),
        total_tokens: 1_200,
        cost: Default::default(),
    };
    model.calculate_cost(&mut usage);

    assert_eq!(usage.cost.input, 0.0032);
    assert_eq!(usage.cost.output, 0.0016);
    assert_eq!(usage.cost.cache_read, 0.0002);
    assert_eq!(usage.cost.cache_write, 0.00062);
    assert_eq!(usage.cost.total, 0.00562);
}

#[test]
fn returns_cost_and_switches_tiers_only_above_the_threshold() {
    let mut model = model();
    model.cost.tiers.push(ModelCostTier {
        rates: ModelCostRates {
            input: 4.0,
            output: 16.0,
            cache_read: 1.0,
            cache_write: 5.0,
        },
        input_tokens_above: 1_000,
    });

    let mut boundary = Usage {
        input: 800,
        output: 100,
        cache_read: 200,
        cache_write: 0,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: 1_100,
        cost: Default::default(),
    };
    let returned = model.calculate_cost(&mut boundary).clone();
    assert_eq!(returned, boundary.cost);
    assert_eq!(boundary.cost.input, 0.0016);
    assert_eq!(boundary.cost.output, 0.0008);
    assert_eq!(boundary.cost.cache_read, 0.0001);

    let mut above = Usage {
        input: 801,
        ..boundary
    };
    model.calculate_cost(&mut above);
    assert_eq!(above.cost.input, 0.003204);
    assert_eq!(above.cost.output, 0.0016);
    assert_eq!(above.cost.cache_read, 0.0002);
}
