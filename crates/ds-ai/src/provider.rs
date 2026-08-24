use crate::{
    AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageStreamError, CacheRetention, Context, Model, ProviderId, StopReason,
    ThinkingLevel,
};
use futures_util::stream;
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Transport {
    Sse,
    WebSocket,
    WebSocketCached,
    Auto,
}

#[derive(Clone, Debug)]
pub struct StreamOptions {
    pub cancellation: CancellationToken,
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, Option<String>>,
    pub timeout: Option<Duration>,
    pub max_retries: Option<usize>,
    pub max_retry_delay: Option<Duration>,
    pub temperature: Option<f64>,
    pub sampling_params: BTreeMap<String, serde_json::Value>,
    pub max_tokens: Option<u64>,
    pub transport: Option<Transport>,
    pub cache_retention: CacheRetention,
    pub session_id: Option<String>,
    pub websocket_connect_timeout: Option<Duration>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            api_key: None,
            headers: BTreeMap::new(),
            timeout: None,
            max_retries: Some(2),
            max_retry_delay: Some(Duration::from_secs(60)),
            temperature: None,
            sampling_params: BTreeMap::new(),
            max_tokens: None,
            transport: None,
            cache_retention: CacheRetention::default(),
            session_id: None,
            websocket_connect_timeout: None,
            metadata: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
}

#[derive(Clone, Debug, Default)]
pub struct SimpleStreamOptions {
    pub stream: StreamOptions,
    pub thinking: Option<ThinkingLevel>,
    pub tool_choice: ToolChoice,
}

pub trait Provider: Send + Sync {
    fn id(&self) -> &ProviderId;
    fn name(&self) -> &str;
    fn base_url(&self) -> Option<&str>;
    fn headers(&self) -> &BTreeMap<String, Option<String>>;
    fn models(&self) -> Vec<Model>;
    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream;
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream;
}

#[derive(Default)]
pub struct Models {
    providers: Vec<Arc<dyn Provider>>,
}

impl Models {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_provider(&mut self, provider: Arc<dyn Provider>) {
        if let Some(current) = self
            .providers
            .iter_mut()
            .find(|current| current.id() == provider.id())
        {
            *current = provider;
        } else {
            self.providers.push(provider);
        }
    }

    pub fn delete_provider(&mut self, id: &str) -> Option<Arc<dyn Provider>> {
        let index = self
            .providers
            .iter()
            .position(|provider| provider.id().as_str() == id)?;
        Some(self.providers.remove(index))
    }

    pub fn clear_providers(&mut self) {
        self.providers.clear();
    }

    pub fn providers(&self) -> Vec<Arc<dyn Provider>> {
        self.providers.clone()
    }

    pub fn provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|provider| provider.id().as_str() == id)
            .cloned()
    }

    pub fn models(&self, provider: Option<&str>) -> Vec<Model> {
        match provider {
            Some(provider) => self
                .providers
                .iter()
                .find(|entry| entry.id().as_str() == provider)
                .map_or_else(Vec::new, |entry| entry.models()),
            None => self
                .providers
                .iter()
                .flat_map(|provider| provider.models())
                .collect(),
        }
    }

    pub fn model(&self, provider: &str, id: &str) -> Option<Model> {
        self.models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    pub fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> AssistantMessageEventStream {
        self.provider(model.provider.as_str()).map_or_else(
            || error_stream(model, format!("Unknown provider {}", model.provider)),
            |provider| provider.stream(model, context, options),
        )
    }

    pub async fn complete(
        &self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<AssistantMessage, AssistantMessageStreamError> {
        self.stream(model, context, options).result().await
    }

    pub fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.provider(model.provider.as_str()).map_or_else(
            || error_stream(model, format!("Unknown provider {}", model.provider)),
            |provider| provider.stream_simple(model, context, options),
        )
    }

    pub async fn complete_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> Result<AssistantMessage, AssistantMessageStreamError> {
        self.stream_simple(model, context, options).result().await
    }
}

fn error_stream(model: &Model, message: String) -> AssistantMessageEventStream {
    let error = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Error,
        error_message: Some(message),
        raw_stop_reason: None,
        end_turn: None,
        timestamp: timestamp(),
    };
    AssistantMessageEventStream::new(stream::iter([AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error,
    }]))
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
