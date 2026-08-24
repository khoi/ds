use ds_ai::{
    Api, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Context, Model,
    ModelCost, ModelInput, Models, Provider, ProviderId, SimpleStreamOptions, StopReason,
    StreamOptions,
};
use futures_util::StreamExt;
use futures_util::stream;
use std::{collections::BTreeMap, sync::Arc};

use crate::support::{Reply, serve};

struct TestProvider {
    id: ProviderId,
    name: String,
    models: Vec<Model>,
    marker: String,
}

impl Provider for TestProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> Option<&str> {
        None
    }

    fn headers(&self) -> &BTreeMap<String, Option<String>> {
        static HEADERS: std::sync::LazyLock<BTreeMap<String, Option<String>>> =
            std::sync::LazyLock::new(BTreeMap::new);
        &HEADERS
    }

    fn models(&self) -> Vec<Model> {
        self.models.clone()
    }

    fn stream(
        &self,
        model: &Model,
        _context: &Context,
        _options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        completed(model, &self.marker)
    }

    fn stream_simple(
        &self,
        model: &Model,
        _context: &Context,
        _options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        completed(model, &self.marker)
    }
}

#[tokio::test]
async fn registers_replaces_lists_routes_and_deletes_providers() {
    let model = model("openai", "gpt-test");
    let mut models = Models::new();
    models.set_provider(provider(&model, "first"));
    assert_eq!(models.providers().len(), 1);
    assert_eq!(models.models(None), std::slice::from_ref(&model));
    assert_eq!(models.model("openai", "gpt-test"), Some(model.clone()));

    models.set_provider(provider(&model, "replacement"));
    let result = models
        .complete(&model, &Context::new([]), &StreamOptions::default())
        .await
        .unwrap();
    assert_eq!(result.error_message.as_deref(), Some("replacement"));

    assert!(models.delete_provider("openai").is_some());
    assert!(models.providers().is_empty());
    models.set_provider(provider(&model, "again"));
    models.clear_providers();
    assert!(models.providers().is_empty());
}

#[tokio::test]
async fn returns_terminal_stream_errors_for_unknown_providers() {
    let model = model("missing", "gpt-test");
    let result = Models::new()
        .complete(&model, &Context::new([]), &StreamOptions::default())
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert_eq!(
        result.error_message.as_deref(),
        Some("Unknown provider missing")
    );
}

#[tokio::test]
async fn returns_terminal_stream_errors_for_unknown_provider_apis() {
    let mut model = model("openai", "gpt-test");
    model.api = Api::AnthropicMessages;
    let mut models = Models::new();
    models.set_provider(ds_ai::openai::provider([model.clone()]));
    let result = models
        .complete(&model, &Context::new([]), &StreamOptions::default())
        .await
        .unwrap();
    assert_eq!(result.stop_reason, StopReason::Error);
    assert!(
        result
            .error_message
            .as_deref()
            .is_some_and(|message| message.contains("no API implementation"))
    );
}

#[tokio::test]
async fn openai_provider_returns_a_stream_before_setup_and_emits_pi_events() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":2}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let mut model = model("openai", "gpt-test");
    model.base_url = server.base_url;
    let mut models = Models::new();
    models.set_provider(ds_ai::openai::provider([model.clone()]));
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let events = models
        .stream(
            &model,
            &Context::new([ds_ai::Message::user("Hello")]),
            &options,
        )
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events[0], AssistantMessageEvent::Start { .. }));
    assert!(matches!(
        events[1],
        AssistantMessageEvent::TextStart {
            content_index: 0,
            ..
        }
    ));
    assert!(matches!(
        &events[2],
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Hello"
    ));
    assert!(matches!(events[3], AssistantMessageEvent::TextEnd { .. }));
    assert!(matches!(events[4], AssistantMessageEvent::Done { .. }));
}

fn provider(model: &Model, marker: &str) -> Arc<dyn Provider> {
    Arc::new(TestProvider {
        id: model.provider.clone(),
        name: "Test".into(),
        models: vec![model.clone()],
        marker: marker.into(),
    })
}

fn model(provider: &str, id: &str) -> Model {
    Model {
        id: id.into(),
        name: id.into(),
        api: Api::OpenAiResponses,
        provider: ProviderId::new(provider),
        base_url: "https://example.com".into(),
        reasoning: false,
        thinking_level_map: BTreeMap::new(),
        input: vec![ModelInput::Text],
        cost: ModelCost::default(),
        context_window: 100,
        max_tokens: 20,
        sampling_params: BTreeMap::new(),
        headers: BTreeMap::new(),
        compat: None,
    }
}

fn completed(model: &Model, marker: &str) -> AssistantMessageEventStream {
    let message = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Stop,
        error_message: Some(marker.into()),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 42,
    };
    AssistantMessageEventStream::new(stream::iter([AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message,
    }]))
}
