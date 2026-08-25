include!("live/support.rs");
include!("live/probes.rs");

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and DS_AI_OPENAI_MODEL"]
async fn openai_live_smoke() {
    let mut model = live_model("openai", "DS_AI_OPENAI_MODEL");
    if let Ok(base_url) = std::env::var("DS_AI_OPENAI_BASE_URL") {
        model.base_url = base_url;
    }
    let stream = ds_ai::openai::stream(
        &model.typed::<OpenAiResponsesOptions>().unwrap(),
        &Context::new([Message::user("Reply with OK")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some(required("OPENAI_API_KEY")),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    assert_completed(stream).await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and DS_AI_ANTHROPIC_MODEL"]
async fn anthropic_live_smoke() {
    let mut model = live_model("anthropic", "DS_AI_ANTHROPIC_MODEL");
    if let Ok(base_url) = std::env::var("DS_AI_ANTHROPIC_BASE_URL") {
        model.base_url = base_url;
    }
    let stream = ds_ai::anthropic::stream(
        &model.typed::<AnthropicOptions>().unwrap(),
        &Context::new([Message::user("Reply with OK")]),
        &AnthropicOptions {
            stream: StreamOptions {
                api_key: Some(required("ANTHROPIC_API_KEY")),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    assert_completed(stream).await;
}

#[tokio::test]
#[ignore = "requires DS_AI_CODEX_ACCESS_TOKEN and DS_AI_CODEX_MODEL"]
async fn codex_live_smoke() {
    let mut model = live_model("openai-codex", "DS_AI_CODEX_MODEL");
    if let Ok(base_url) = std::env::var("DS_AI_CODEX_BASE_URL") {
        model.base_url = base_url;
    }
    let stream = ds_ai::codex::stream(
        &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
        &Context::new([Message::user("Reply with OK")]),
        &OpenAiCodexResponsesOptions {
            stream: StreamOptions {
                api_key: Some(required("DS_AI_CODEX_ACCESS_TOKEN")),
                transport: Some(Transport::Auto),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    assert_completed(stream).await;
}

async fn assert_completed(mut stream: AssistantMessageEventStream) {
    assert!(!stream.result().await.unwrap().content.is_empty());
}

fn live_model(provider: &str, name: &str) -> Model {
    let id = required(name);
    builtin_model(provider, &id).unwrap_or_else(|| panic!("unknown {provider} model {id}"))
}
