use crate::{AssistantContent, AssistantMessage, ImageContent, Model, ModelInput, TextContent};
use serde::{
    Deserialize, Deserializer, Serialize, Serializer, de::Error as _, ser::SerializeStruct,
};
use std::time::Duration;
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UserMessage {
    pub content: UserContent,
    pub timestamp: u64,
}

impl UserMessage {
    pub fn new(content: impl Into<String>, timestamp: u64) -> Self {
        Self {
            content: UserContent::Text(content.into()),
            timestamp,
        }
    }

    pub fn with_blocks(content: impl IntoIterator<Item = InputContent>, timestamp: u64) -> Self {
        Self {
            content: UserContent::Blocks(content.into_iter().collect()),
            timestamp,
        }
    }
}

impl Serialize for UserMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("UserMessage", 3)?;
        state.serialize_field("role", "user")?;
        state.serialize_field("content", &self.content)?;
        state.serialize_field("timestamp", &self.timestamp)?;
        state.end()
    }
}

#[derive(Deserialize)]
struct UserMessageWire {
    role: String,
    content: UserContent,
    timestamp: u64,
}

impl<'de> Deserialize<'de> for UserMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = UserMessageWire::deserialize(deserializer)?;
        if wire.role != "user" {
            return Err(D::Error::custom("user message role must be user"));
        }
        Ok(Self {
            content: wire.content,
            timestamp: wire.timestamp,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Blocks(Vec<InputContent>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<InputContent>,
    pub details: Option<serde_json::Value>,
    pub usage: Option<Usage>,
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

impl Serialize for ToolResultMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ToolResultMessage", 9)?;
        state.serialize_field("role", "toolResult")?;
        state.serialize_field("toolCallId", &self.tool_call_id)?;
        state.serialize_field("toolName", &self.tool_name)?;
        state.serialize_field("content", &self.content)?;
        if let Some(value) = &self.details {
            state.serialize_field("details", value)?;
        }
        if let Some(value) = &self.usage {
            state.serialize_field("usage", value)?;
        }
        if let Some(value) = &self.added_tool_names {
            state.serialize_field("addedToolNames", value)?;
        }
        state.serialize_field("isError", &self.is_error)?;
        state.serialize_field("timestamp", &self.timestamp)?;
        state.end()
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolResultMessageWire {
    role: String,
    tool_call_id: String,
    tool_name: String,
    content: Vec<InputContent>,
    details: Option<serde_json::Value>,
    usage: Option<Usage>,
    added_tool_names: Option<Vec<String>>,
    is_error: bool,
    timestamp: u64,
}

impl<'de> Deserialize<'de> for ToolResultMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = ToolResultMessageWire::deserialize(deserializer)?;
        if wire.role != "toolResult" {
            return Err(D::Error::custom(
                "tool result message role must be toolResult",
            ));
        }
        Ok(Self {
            tool_call_id: wire.tool_call_id,
            tool_name: wire.tool_name,
            content: wire.content,
            details: wire.details,
            usage: wire.usage,
            added_tool_names: wire.added_tool_names,
            is_error: wire.is_error,
            timestamp: wire.timestamp,
        })
    }
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
        for message in &mut context.messages {
            normalize_assistant_for_model(message, model);
        }
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

fn normalize_assistant_for_model(message: &mut Message, model: &crate::Model) {
    let Message::Assistant(message) = message else {
        return;
    };
    if message.api == model.api && message.provider == model.provider && message.model == model.id {
        return;
    }
    message.content = std::mem::take(&mut message.content)
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::Thinking(thinking) if thinking.redacted == Some(true) => None,
            AssistantContent::Thinking(thinking) if thinking.thinking.trim().is_empty() => None,
            AssistantContent::Thinking(thinking) => Some(AssistantContent::Text(TextContent {
                text: thinking.thinking,
                text_signature: None,
            })),
            AssistantContent::Text(mut text) => {
                text.text_signature = None;
                Some(AssistantContent::Text(text))
            }
            AssistantContent::ToolCall(mut call) => {
                call.thought_signature = None;
                Some(AssistantContent::ToolCall(call))
            }
        })
        .collect();
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

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Response {
    pub(crate) response_model: String,
    pub(crate) id: Option<String>,
    pub(crate) content: Vec<AssistantContent>,
    pub(crate) usage: Usage,
    pub(crate) stop_reason: StopReason,
    pub(crate) raw_stop_reason: Option<String>,
    pub(crate) service_tier: Option<String>,
    pub(crate) end_turn: Option<bool>,
    diagnostics: Option<Vec<crate::AssistantMessageDiagnostic>>,
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
    pub(crate) fn new(response_model: String) -> Self {
        Self {
            response_model,
            ..Self::default()
        }
    }

    pub(crate) fn add_diagnostic(&mut self, diagnostic: crate::AssistantMessageDiagnostic) {
        self.diagnostics
            .get_or_insert_with(Vec::new)
            .push(diagnostic);
    }

    pub(crate) fn into_assistant_message(self, model: &Model, timestamp: u64) -> AssistantMessage {
        let response_model = (self.response_model != model.id).then_some(self.response_model);
        AssistantMessage {
            content: self.content,
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            response_model,
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
}

#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum Error {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("request hook failed: {0}")]
    Hook(String),
    #[error("request compression failed: {0}")]
    Compression(String),
    #[error("provider returned HTTP {status}: {message}")]
    Provider { status: u16, message: String },
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
pub(crate) enum TimeoutPhase {
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
