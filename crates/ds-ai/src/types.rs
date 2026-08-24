use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::{pin::Pin, time::Duration};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Message {
    User(String),
    Assistant(Response),
    ToolResult(ToolResult),
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self::User(content.into())
    }

    pub fn assistant(response: Response) -> Self {
        Self::Assistant(response)
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
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

impl Context {
    pub fn new(messages: impl IntoIterator<Item = Message>) -> Self {
        Self {
            messages: messages.into_iter().collect(),
            tools: Vec::new(),
        }
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    pub(crate) fn messages(&self) -> &[Message] {
        &self.messages
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
        }
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
    #[serde(default, rename = "_provider")]
    provider: ProviderState,
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
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
enum ProviderState {
    #[default]
    None,
    OpenAi(OpenAiState),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    TextDelta { content_index: usize, delta: String },
    ReasoningDelta { content_index: usize, delta: String },
    ToolCallDelta { content_index: usize, delta: String },
    Done(Response),
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum Error {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("provider returned HTTP {status}: {body}")]
    Provider { status: u16, body: String },
    #[error("invalid provider stream: {0}")]
    Stream(String),
    #[error("provider stream ended before a terminal event")]
    IncompleteStream { partial: Response },
    #[error("request cancelled")]
    Cancelled,
    #[error("provider retry delay {requested:?} exceeds {maximum:?}")]
    RetryDelayExceeded {
        requested: Duration,
        maximum: Duration,
    },
}

pub type ResponseStream = Pin<Box<dyn Stream<Item = Result<Event, Error>> + Send>>;
