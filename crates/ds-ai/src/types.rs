use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::{pin::Pin, time::Duration};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Message {
    User(Vec<InputContent>),
    Assistant(Box<Response>),
    ToolResult(ToolResult),
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self::User(vec![InputContent::text(content)])
    }

    pub fn user_content(content: impl IntoIterator<Item = InputContent>) -> Self {
        Self::User(content.into_iter().collect())
    }

    pub fn assistant(response: Response) -> Self {
        Self::Assistant(Box::new(response))
    }

    pub fn tool_result(result: ToolResult) -> Self {
        Self::ToolResult(result)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub content: Vec<InputContent>,
    pub is_error: bool,
}

impl ToolResult {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        content: impl IntoIterator<Item = InputContent>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            content: content.into_iter().collect(),
            is_error: false,
        }
    }

    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum InputContent {
    Text(String),
    Image { media_type: String, data: String },
}

impl InputContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn image(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image {
            media_type: media_type.into(),
            data: data.into(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Context {
    system: Option<String>,
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

impl Context {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            system: None,
            messages: messages.into_iter().collect(),
            tools: Vec::new(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    pub(crate) fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub(crate) fn system(&self) -> Option<&str> {
        self.system.as_deref()
    }

    pub(crate) fn tools(&self) -> &[Tool] {
        &self.tools
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    strict: bool,
}

impl Tool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            strict: false,
        }
    }

    pub fn with_strict(mut self) -> Self {
        self.strict = true;
        self
    }

    pub(crate) fn strict(&self) -> bool {
        self.strict
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Content {
    Text(String),
    Reasoning(String),
    ToolCall(ToolCall),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub reasoning: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: Option<String>,
    pub content: Vec<Content>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    pub raw_stop_reason: Option<String>,
    pub metadata: ResponseMetadata,
    #[serde(default, rename = "_provider")]
    provider: ProviderState,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub request_id: Option<String>,
    pub rate_limits: RateLimits,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RateLimits {
    pub limit_requests: Option<u64>,
    pub remaining_requests: Option<u64>,
    pub reset_requests: Option<String>,
    pub limit_tokens: Option<u64>,
    pub remaining_tokens: Option<u64>,
    pub reset_tokens: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    #[default]
    Pending,
    Stop,
    Length,
    ToolUse,
    Pause,
    Error,
    Aborted,
}

impl Response {
    pub(crate) fn openai(model: String) -> Self {
        Self {
            provider: ProviderState::OpenAi(OpenAiState {
                model,
                items: Vec::new(),
            }),
            ..Self::default()
        }
    }

    pub(crate) fn openai_items(&self, model: &str) -> Option<&[OpenAiReplay]> {
        match &self.provider {
            ProviderState::OpenAi(state) if state.model == model => Some(&state.items),
            _ => None,
        }
    }

    pub(crate) fn add_openai_item(&mut self, item: OpenAiReplay) {
        if let ProviderState::OpenAi(state) = &mut self.provider {
            state.items.push(item);
        }
    }

    pub(crate) fn anthropic(model: String) -> Self {
        Self {
            provider: ProviderState::Anthropic(AnthropicState {
                model,
                reasoning: Vec::new(),
            }),
            ..Self::default()
        }
    }

    pub(crate) fn anthropic_reasoning(&self, model: &str) -> Option<&[AnthropicReasoning]> {
        match &self.provider {
            ProviderState::Anthropic(state) if state.model == model => Some(&state.reasoning),
            _ => None,
        }
    }

    pub(crate) fn add_anthropic_reasoning(&mut self, reasoning: AnthropicReasoning) {
        if let ProviderState::Anthropic(state) = &mut self.provider {
            state.reasoning.push(reasoning);
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
enum ProviderState {
    #[default]
    None,
    OpenAi(OpenAiState),
    Anthropic(AnthropicState),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct OpenAiState {
    model: String,
    items: Vec<OpenAiReplay>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum OpenAiReplay {
    Reasoning {
        content_index: usize,
        id: String,
        encrypted_content: Option<String>,
    },
    Message {
        content_index: usize,
        id: String,
        phase: Option<String>,
    },
    ToolCall {
        content_index: usize,
        item_id: String,
        namespace: Option<String>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct AnthropicState {
    model: String,
    reasoning: Vec<AnthropicReasoning>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub(crate) enum AnthropicReasoning {
    Thinking {
        content_index: usize,
        signature: String,
    },
    Redacted {
        content_index: usize,
        data: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    TextDelta { content_index: usize, delta: String },
    ReasoningDelta { content_index: usize, delta: String },
    ToolCallDelta { content_index: usize, delta: String },
    Done(Box<Response>),
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("provider returned HTTP {status}: {message}")]
    Provider {
        status: u16,
        code: Option<String>,
        message: String,
        request_id: Option<String>,
        retry_after: Option<Duration>,
        rate_limits: RateLimits,
    },
    #[error("invalid provider stream: {message}")]
    Stream { message: String, partial: Response },
    #[error("provider stream ended before a terminal event")]
    IncompleteStream { partial: Response },
    #[error("provider response failed: {message}")]
    Response {
        code: Option<String>,
        message: String,
        partial: Response,
    },
    #[error("request cancelled")]
    Cancelled { partial: Option<Response> },
    #[error("provider timed out during {phase:?}")]
    Timeout {
        phase: TimeoutPhase,
        partial: Option<Response>,
    },
    #[error("provider retry delay {requested:?} exceeds {maximum:?}")]
    RetryDelayExceeded {
        requested: Duration,
        maximum: Duration,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimeoutPhase {
    Connection,
    FirstEvent,
    Idle,
    Overall,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum CacheRetention {
    None,
    #[default]
    Short,
    Long,
}

pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
