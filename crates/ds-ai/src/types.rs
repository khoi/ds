use crate::{
    Api, AssistantContent, AssistantMessage, AssistantToolCall, ImageContent, ModelInput,
    ProviderId, TextContent, ThinkingContent,
};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::{pin::Pin, time::Duration};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Message {
    User(UserMessage),
    Assistant(Box<AssistantMessage>),
    ToolResult(Box<ToolResultMessage>),
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self::User(UserMessage::new(content, timestamp()))
    }

    pub fn user_content(content: impl IntoIterator<Item = InputContent>) -> Self {
        Self::User(UserMessage::with_blocks(content, timestamp()))
    }

    pub fn assistant(message: impl Into<AssistantMessage>) -> Self {
        Self::Assistant(Box::new(message.into()))
    }

    pub fn tool_result(result: ToolResultMessage) -> Self {
        Self::ToolResult(Box::new(result))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UserMessage {
    pub role: UserRole,
    pub content: UserContent,
    pub timestamp: u64,
}

impl UserMessage {
    pub fn new(content: impl Into<String>, timestamp: u64) -> Self {
        Self {
            role: UserRole::User,
            content: UserContent::Text(content.into()),
            timestamp,
        }
    }

    pub fn with_blocks(content: impl IntoIterator<Item = InputContent>, timestamp: u64) -> Self {
        Self {
            role: UserRole::User,
            content: UserContent::Blocks(content.into_iter().collect()),
            timestamp,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum UserRole {
    User,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<InputContent>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultMessage {
    pub role: ToolResultRole,
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<InputContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
    pub timestamp: u64,
}

impl ToolResultMessage {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        content: impl IntoIterator<Item = InputContent>,
    ) -> Self {
        Self {
            role: ToolResultRole::ToolResult,
            tool_call_id: id.into(),
            tool_name: name.into(),
            content: content.into_iter().collect(),
            details: None,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: timestamp(),
        }
    }

    pub fn with_error(mut self, is_error: bool) -> Self {
        self.is_error = is_error;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ToolResultRole {
    ToolResult,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum InputContent {
    Text(TextContent),
    Image(ImageContent),
}

impl InputContent {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextContent {
            text: text.into(),
            text_signature: None,
        })
    }

    pub fn image(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Image(ImageContent {
            mime_type: media_type.into(),
            data: data.into(),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Context {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
}

impl Context {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            system_prompt: None,
            messages: messages.into_iter().collect(),
            tools: Vec::new(),
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
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
        self.system_prompt.as_deref()
    }

    pub(crate) fn for_model(&self, model: &crate::Model) -> Self {
        let mut context = self.clone();
        if model.input.contains(&ModelInput::Image) {
            return context;
        }
        for message in &mut context.messages {
            let (content, placeholder) = match message {
                Message::User(message) => {
                    let UserContent::Blocks(content) = &mut message.content else {
                        continue;
                    };
                    (content, "(image omitted: model does not support images)")
                }
                Message::ToolResult(message) => (
                    &mut message.content,
                    "(tool image omitted: model does not support images)",
                ),
                Message::Assistant(_) => continue,
            };
            for item in content {
                if matches!(item, InputContent::Image(_)) {
                    *item = InputContent::text(placeholder);
                }
            }
        }
        context
    }

    pub(crate) fn tools(&self) -> &[Tool] {
        &self.tools
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSampling>,
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
            constrained_sampling: None,
        }
    }

    pub fn with_strict(mut self) -> Self {
        self.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: ConstrainedSamplingStrictness::Require,
        });
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstrainedSampling {
    Disabled,
    JsonSchema {
        strict: ConstrainedSamplingStrictness,
    },
    Grammar {
        variants: GrammarVariants,
    },
}

impl Serialize for ConstrainedSampling {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Disabled => serializer.serialize_bool(false),
            Self::JsonSchema { strict } => {
                ConstrainedSamplingConfig::JsonSchema { strict: *strict }.serialize(serializer)
            }
            Self::Grammar { variants } => ConstrainedSamplingConfig::Grammar {
                variants: variants.clone(),
            }
            .serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ConstrainedSampling {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Bool(false) => Ok(Self::Disabled),
            serde_json::Value::Bool(true) => Err(serde::de::Error::custom(
                "constrained sampling boolean must be false",
            )),
            value => match serde_json::from_value(value).map_err(serde::de::Error::custom)? {
                ConstrainedSamplingConfig::JsonSchema { strict } => Ok(Self::JsonSchema { strict }),
                ConstrainedSamplingConfig::Grammar { variants } => Ok(Self::Grammar { variants }),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ConstrainedSamplingConfig {
    JsonSchema {
        strict: ConstrainedSamplingStrictness,
    },
    Grammar {
        variants: GrammarVariants,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstrainedSamplingStrictness {
    Prefer,
    Require,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GrammarVariants {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
    pub total_tokens: u64,
    pub cost: UsageCost,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub id: Option<String>,
    pub content: Vec<Content>,
    pub usage: Usage,
    pub stop_reason: StopReason,
    pub raw_stop_reason: Option<String>,
    pub service_tier: Option<String>,
    pub end_turn: Option<bool>,
    pub metadata: ResponseMetadata,
    #[serde(default, skip_serializing)]
    diagnostics: Option<Vec<crate::AssistantMessageDiagnostic>>,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StopReason {
    #[default]
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
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

    pub(crate) fn codex(model: String) -> Self {
        Self {
            provider: ProviderState::Codex(OpenAiState {
                model,
                items: Vec::new(),
            }),
            ..Self::default()
        }
    }

    pub(crate) fn add_diagnostic(&mut self, diagnostic: crate::AssistantMessageDiagnostic) {
        self.diagnostics
            .get_or_insert_with(Vec::new)
            .push(diagnostic);
    }

    pub(crate) fn add_openai_item(&mut self, item: OpenAiReplay) {
        if let ProviderState::OpenAi(state) | ProviderState::Codex(state) = &mut self.provider {
            state.items.push(item);
        }
    }

    pub(crate) fn backfill_openai_reasoning(&mut self, id: &str, encrypted: &str) {
        let (ProviderState::OpenAi(state) | ProviderState::Codex(state)) = &mut self.provider
        else {
            return;
        };
        for item in &mut state.items {
            let OpenAiReplay::Reasoning {
                id: item_id,
                encrypted_content,
                ..
            } = item
            else {
                continue;
            };
            if item_id != id {
                continue;
            }
            if encrypted_content.is_none() {
                *encrypted_content = Some(encrypted.to_owned());
            }
            return;
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

    pub(crate) fn add_anthropic_reasoning(&mut self, reasoning: AnthropicReasoning) {
        if let ProviderState::Anthropic(state) = &mut self.provider {
            state.reasoning.push(reasoning);
        }
    }

    pub(crate) fn set_anthropic_model(&mut self, model: String) {
        if let ProviderState::Anthropic(state) = &mut self.provider {
            state.model = model;
        }
    }

    pub fn into_assistant_message(self, timestamp: u64) -> AssistantMessage {
        let (api, provider, model) = match &self.provider {
            ProviderState::OpenAi(state) => (
                Api::OpenAiResponses,
                ProviderId::new("openai"),
                state.model.clone(),
            ),
            ProviderState::Codex(state) => (
                Api::OpenAiCodexResponses,
                ProviderId::new("openai-codex"),
                state.model.clone(),
            ),
            ProviderState::Anthropic(state) => (
                Api::AnthropicMessages,
                ProviderId::new("anthropic"),
                state.model.clone(),
            ),
            ProviderState::None => (
                Api::Other("unknown".into()),
                ProviderId::new("unknown"),
                String::new(),
            ),
        };
        let content = self
            .content
            .iter()
            .enumerate()
            .map(|(content_index, content)| self.assistant_content(content_index, content))
            .collect();
        AssistantMessage {
            content,
            api,
            provider,
            model,
            response_model: None,
            response_id: self.id,
            diagnostics: self.diagnostics,
            usage: self.usage,
            stop_reason: self.stop_reason,
            error_message: None,
            raw_stop_reason: self.raw_stop_reason,
            end_turn: self.end_turn,
            timestamp,
        }
    }

    fn assistant_content(&self, content_index: usize, content: &Content) -> AssistantContent {
        match content {
            Content::Text(text) => AssistantContent::Text(TextContent {
                text: text.clone(),
                text_signature: self.openai_text_signature(content_index),
            }),
            Content::Reasoning(thinking) => AssistantContent::Thinking(ThinkingContent {
                thinking: thinking.clone(),
                thinking_signature: self.thinking_signature(content_index, thinking),
                redacted: self.anthropic_redacted(content_index),
            }),
            Content::ToolCall(call) => {
                let (id, namespace) = self.openai_tool_identity(content_index, &call.id);
                AssistantContent::ToolCall(AssistantToolCall {
                    id,
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    thought_signature: None,
                    namespace,
                })
            }
        }
    }

    fn openai_text_signature(&self, content_index: usize) -> Option<String> {
        let items = match &self.provider {
            ProviderState::OpenAi(state) | ProviderState::Codex(state) => &state.items,
            _ => return None,
        };
        let OpenAiReplay::Message { id, phase, .. } = items.iter().find(|item| {
            matches!(item, OpenAiReplay::Message { content_index: index, .. } if *index == content_index)
        })?
        else {
            return None;
        };
        serde_json::to_string(&serde_json::json!({
            "v": 1,
            "id": id,
            "phase": phase
        }))
        .ok()
    }

    fn thinking_signature(&self, content_index: usize, thinking: &str) -> Option<String> {
        match &self.provider {
            ProviderState::OpenAi(state) | ProviderState::Codex(state) => {
                let OpenAiReplay::Reasoning {
                    id,
                    encrypted_content,
                    ..
                } = state.items.iter().find(|item| {
                    matches!(item, OpenAiReplay::Reasoning { content_index: index, .. } if *index == content_index)
                })?
                else {
                    return None;
                };
                let mut item = serde_json::json!({
                    "type": "reasoning",
                    "id": id,
                    "summary": [{"type": "summary_text", "text": thinking}]
                });
                if let Some(encrypted_content) = encrypted_content {
                    item["encrypted_content"] = encrypted_content.clone().into();
                }
                serde_json::to_string(&item).ok()
            }
            ProviderState::Anthropic(state) => {
                state
                    .reasoning
                    .iter()
                    .find_map(|reasoning| match reasoning {
                        AnthropicReasoning::Thinking {
                            content_index: index,
                            signature,
                        } if *index == content_index => Some(signature.clone()),
                        AnthropicReasoning::Redacted {
                            content_index: index,
                            data,
                        } if *index == content_index => Some(data.clone()),
                        _ => None,
                    })
            }
            ProviderState::None => None,
        }
    }

    fn anthropic_redacted(&self, content_index: usize) -> Option<bool> {
        match &self.provider {
            ProviderState::Anthropic(state)
                if state.reasoning.iter().any(|reasoning| {
                    matches!(reasoning, AnthropicReasoning::Redacted { content_index: index, .. } if *index == content_index)
                }) => Some(true),
            _ => None,
        }
    }

    fn openai_tool_identity(
        &self,
        content_index: usize,
        call_id: &str,
    ) -> (String, Option<String>) {
        let (ProviderState::OpenAi(state) | ProviderState::Codex(state)) = &self.provider else {
            return (call_id.to_owned(), None);
        };
        let Some(OpenAiReplay::ToolCall {
            item_id, namespace, ..
        }) = state.items.iter().find(|item| {
            matches!(item, OpenAiReplay::ToolCall { content_index: index, .. } if *index == content_index)
        })
        else {
            return (call_id.to_owned(), None);
        };
        let id = if call_id.contains('|') {
            call_id.to_owned()
        } else {
            format!("{call_id}|{item_id}")
        };
        (id, namespace.clone())
    }
}

impl From<Response> for AssistantMessage {
    fn from(response: Response) -> Self {
        response.into_assistant_message(0)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
enum ProviderState {
    #[default]
    None,
    OpenAi(OpenAiState),
    Codex(OpenAiState),
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

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    TextDelta { content_index: usize, delta: String },
    ReasoningDelta { content_index: usize, delta: String },
    ToolCallDelta { content_index: usize, delta: String },
    Done(Box<Response>),
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("request hook failed: {0}")]
    Hook(String),
    #[error("request compression failed: {0}")]
    Compression(String),
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

pub(crate) fn normalize_id(id: &str) -> String {
    let id = id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character
            } else {
                '_'
            }
        })
        .take(64)
        .collect::<String>();
    if id.is_empty() { "_".into() } else { id }
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
