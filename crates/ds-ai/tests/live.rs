use ds_ai::{Context, Event, Message, ResponseStream, anthropic, codex, openai};
use futures_util::StreamExt;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and DS_AI_OPENAI_MODEL"]
async fn openai_live_smoke() {
    let mut model = openai::Model::new(required("DS_AI_OPENAI_MODEL"));
    if let Ok(base_url) = std::env::var("DS_AI_OPENAI_BASE_URL") {
        model = model.with_base_url(base_url);
    }
    let stream = openai::raw_stream(
        &model,
        &Context::new([Message::user("Reply with OK")]),
        &openai::Options::new(required("OPENAI_API_KEY")),
    )
    .await
    .unwrap();
    assert_completed(stream).await;
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY and DS_AI_ANTHROPIC_MODEL"]
async fn anthropic_live_smoke() {
    let mut model = anthropic::Model::new(required("DS_AI_ANTHROPIC_MODEL"));
    if let Ok(base_url) = std::env::var("DS_AI_ANTHROPIC_BASE_URL") {
        model = model.with_base_url(base_url);
    }
    let stream = anthropic::raw_stream(
        &model,
        &Context::new([Message::user("Reply with OK")]),
        &anthropic::Options::new(required("ANTHROPIC_API_KEY")),
    )
    .await
    .unwrap();
    assert_completed(stream).await;
}

#[tokio::test]
#[ignore = "requires DS_AI_CODEX_ACCESS_TOKEN and DS_AI_CODEX_MODEL"]
async fn codex_live_smoke() {
    let mut model = codex::Model::new(required("DS_AI_CODEX_MODEL"));
    if let Ok(base_url) = std::env::var("DS_AI_CODEX_BASE_URL") {
        model = model.with_base_url(base_url);
    }
    let stream = codex::raw_stream(
        &model,
        &Context::new([Message::user("Reply with OK")]),
        &codex::Options::new(required("DS_AI_CODEX_ACCESS_TOKEN")),
    )
    .await
    .unwrap();
    assert_completed(stream).await;
}

async fn assert_completed(mut stream: ResponseStream) {
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            Event::Done(response) => {
                assert!(!response.content.is_empty());
                return;
            }
            Event::TextDelta { .. }
            | Event::ReasoningDelta { .. }
            | Event::ToolCallDelta { .. } => {}
        }
    }
    panic!("provider stream did not complete")
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"))
}
