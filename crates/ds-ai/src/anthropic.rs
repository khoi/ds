use crate::{
    CacheRetention, Content, Context, Error, Event, InputContent, Message, Response,
    ResponseStream, StopReason, ToolResult, Usage, http, retry, transport,
    types::AnthropicReasoning,
};
use async_stream::stream;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Model {
    id: String,
    base_url: String,
}

impl Model {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: DEFAULT_BASE_URL.into(),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

pub struct Options {
    api_key: String,
    max_tokens: u64,
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
    connection_timeout: Option<Duration>,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_timeout: Option<Duration>,
    temperature: Option<f64>,
    stop_sequences: Vec<String>,
    thinking: Option<Thinking>,
    metadata_user_id: Option<String>,
    tool_choice: Option<ToolChoice>,
    cache_retention: CacheRetention,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Thinking {
    Disabled,
    Enabled {
        budget_tokens: u64,
        display: ThinkingDisplay,
    },
    Adaptive {
        effort: Effort,
        display: ThinkingDisplay,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingDisplay {
    Summarized,
    Omitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Low,
    Medium,
    High,
    Max,
    XHigh,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoice {
    Auto,
    Any,
    None,
    Tool(String),
}

impl Options {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_tokens: 4096,
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
            connection_timeout: None,
            first_event_timeout: None,
            idle_timeout: None,
            overall_timeout: None,
            temperature: None,
            stop_sequences: Vec::new(),
            thinking: None,
            metadata_user_id: None,
            tool_choice: None,
            cache_retention: CacheRetention::Short,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_max_retry_delay(mut self, max_retry_delay: Option<Duration>) -> Self {
        self.max_retry_delay = max_retry_delay;
        self
    }

    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn with_connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = Some(timeout);
        self
    }

    pub fn with_first_event_timeout(mut self, timeout: Duration) -> Self {
        self.first_event_timeout = Some(timeout);
        self
    }

    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = Some(timeout);
        self
    }

    pub fn with_overall_timeout(mut self, timeout: Duration) -> Self {
        self.overall_timeout = Some(timeout);
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_stop_sequences(
        mut self,
        stop_sequences: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.stop_sequences = stop_sequences.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_thinking(mut self, thinking: Thinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    pub fn with_metadata_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.metadata_user_id = Some(user_id.into());
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_cache_retention(mut self, retention: CacheRetention) -> Self {
        self.cache_retention = retention;
        self
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<serde_json::Value>,
    messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<RequestTool<'a>>,
    max_tokens: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct RequestMessage {
    role: &'static str,
    content: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct RequestTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Clone, Serialize)]
struct CacheControl {
    r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<&'static str>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: StartedMessage },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: ContentDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        #[serde(default)]
        usage: AnthropicUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: ErrorDetail },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct StartedMessage {
    id: String,
    #[serde(default)]
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ContentBlock {
    r#type: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    thinking: String,
    #[serde(default)]
    signature: String,
    #[serde(default)]
    data: String,
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ContentDelta {
    r#type: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    thinking: String,
    #[serde(default)]
    signature: String,
    #[serde(default)]
    partial_json: String,
}

#[derive(Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    output_tokens_details: Option<OutputTokenDetails>,
}

#[derive(Deserialize)]
struct OutputTokenDetails {
    thinking_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct ErrorDetail {
    r#type: String,
    message: String,
}

enum Slot {
    Text(usize),
    Thinking {
        content_index: usize,
        signature: String,
    },
    Redacted,
    ToolCall {
        content_index: usize,
        partial_json: String,
    },
}

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<ResponseStream, Error> {
    let overall_deadline = options
        .overall_timeout
        .map(|timeout| Instant::now() + timeout);
    let cache_control = cache_control(options.cache_retention);
    let (thinking, output_config) = thinking(&options.thinking);
    let tool_count = context.tools().len();
    let request = Request {
        model: &model.id,
        system: context
            .system()
            .map(|system| {
                let mut block = serde_json::json!({"type": "text", "text": system});
                add_cache_control(&mut block, cache_control.as_ref());
                vec![block]
            })
            .unwrap_or_default(),
        messages: messages(model, context, cache_control.as_ref()),
        tools: context
            .tools()
            .iter()
            .enumerate()
            .map(|(index, tool)| RequestTool {
                name: &tool.name,
                description: &tool.description,
                input_schema: &tool.parameters,
                cache_control: if index + 1 == tool_count {
                    cache_control.clone()
                } else {
                    None
                },
            })
            .collect(),
        max_tokens: options.max_tokens,
        stream: true,
        stop_sequences: options.stop_sequences.clone(),
        temperature: if matches!(
            options.thinking.as_ref(),
            Some(Thinking::Enabled { .. } | Thinking::Adaptive { .. })
        ) {
            None
        } else {
            options.temperature
        },
        thinking,
        output_config,
        metadata: options
            .metadata_user_id
            .as_ref()
            .map(|user_id| serde_json::json!({"user_id": user_id})),
        tool_choice: options.tool_choice.as_ref().map(tool_choice),
    };
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", model.base_url.trim_end_matches('/'));
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: options.max_retries,
                max_delay: options.max_retry_delay,
                cancellation: &options.cancellation,
            },
            || {
                client
                    .post(&url)
                    .header("x-api-key", &options.api_key)
                    .header("anthropic-version", "2023-06-01")
                    .json(&request)
                    .send()
            },
        ),
        options.connection_timeout,
        overall_deadline,
    )
    .await?;
    if !response.status().is_success() {
        return Err(http::provider_error(response).await);
    }

    let metadata = http::metadata(response.headers());
    let response_model = model.id.clone();
    let stream_cancellation = options.cancellation.clone();
    let first_event_timeout = options.first_event_timeout;
    let idle_timeout = options.idle_timeout;
    let output = stream! {
        let mut events = transport::EventStream::new(
            response,
            stream_cancellation,
            first_event_timeout,
            idle_timeout,
            overall_deadline,
        );
        let mut result = Response::anthropic(response_model);
        result.metadata = metadata;
        let mut slots = HashMap::new();

        loop {
            let data = match events.next().await {
                Ok(Some(data)) => data,
                Ok(None) => break,
                Err(transport::ReadError::Cancelled) => {
                    result.stop_reason = StopReason::Aborted;
                    result.raw_stop_reason = Some("cancelled".into());
                    yield Err(Error::Cancelled { partial: Some(result) });
                    return;
                }
                Err(transport::ReadError::Timeout(phase)) => {
                    result.stop_reason = StopReason::Error;
                    result.raw_stop_reason = Some(match phase {
                        crate::TimeoutPhase::FirstEvent => "timeout.first_event".into(),
                        crate::TimeoutPhase::Idle => "timeout.idle".into(),
                        crate::TimeoutPhase::Overall => "timeout.overall".into(),
                        crate::TimeoutPhase::Connection => unreachable!(),
                    });
                    yield Err(Error::Timeout {
                        phase,
                        partial: Some(result),
                    });
                    return;
                }
                Err(transport::ReadError::Stream(message)) => {
                    yield Err(Error::Stream { message, partial: result });
                    return;
                }
            };
            let event = match serde_json::from_str::<StreamEvent>(&data) {
                Ok(event) => event,
                Err(error) => {
                    yield Err(Error::Stream {
                        message: error.to_string(),
                        partial: result,
                    });
                    return;
                }
            };
            match event {
                    StreamEvent::MessageStart { message } => {
                        result.id = Some(message.id);
                        apply_usage(&mut result.usage, message.usage);
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "text" =>
                    {
                        let content_index = result.content.len();
                        result.content.push(Content::Text(content_block.text));
                        slots.insert(index, Slot::Text(content_index));
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "thinking" =>
                    {
                        let content_index = result.content.len();
                        result.content.push(Content::Reasoning(content_block.thinking));
                        slots.insert(index, Slot::Thinking {
                            content_index,
                            signature: content_block.signature,
                        });
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "redacted_thinking" =>
                    {
                        let content_index = result.content.len();
                        result.content.push(Content::Reasoning("[Reasoning redacted]".into()));
                        result.add_anthropic_reasoning(AnthropicReasoning::Redacted {
                            content_index,
                            data: content_block.data,
                        });
                        slots.insert(index, Slot::Redacted);
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "tool_use" =>
                    {
                        let (Some(id), Some(name)) = (content_block.id, content_block.name) else {
                            continue;
                        };
                        let content_index = result.content.len();
                        result.content.push(Content::ToolCall(crate::ToolCall {
                            id,
                            name,
                            arguments: content_block.input.unwrap_or_else(|| serde_json::json!({})),
                        }));
                        slots.insert(index, Slot::ToolCall {
                            content_index,
                            partial_json: String::new(),
                        });
                    }
                    StreamEvent::ContentBlockDelta { index, delta } => {
                        match (delta.r#type.as_str(), slots.get_mut(&index)) {
                            ("text_delta", Some(Slot::Text(content_index))) => {
                                if let Content::Text(text) = &mut result.content[*content_index] {
                                    text.push_str(&delta.text);
                                }
                                yield Ok(Event::TextDelta {
                                    content_index: *content_index,
                                    delta: delta.text,
                                });
                            }
                            ("thinking_delta", Some(Slot::Thinking { content_index, .. })) => {
                                if let Content::Reasoning(reasoning) = &mut result.content[*content_index] {
                                    reasoning.push_str(&delta.thinking);
                                }
                                yield Ok(Event::ReasoningDelta {
                                    content_index: *content_index,
                                    delta: delta.thinking,
                                });
                            }
                            ("signature_delta", Some(Slot::Thinking { signature, .. })) => {
                                signature.push_str(&delta.signature);
                            }
                            ("input_json_delta", Some(Slot::ToolCall { content_index, partial_json })) => {
                                partial_json.push_str(&delta.partial_json);
                                if let Content::ToolCall(call) = &mut result.content[*content_index] {
                                    call.arguments = parse_arguments(partial_json);
                                }
                                yield Ok(Event::ToolCallDelta {
                                    content_index: *content_index,
                                    delta: delta.partial_json,
                                });
                            }
                            _ => {}
                        }
                    }
                    StreamEvent::ContentBlockStop { index } => {
                        match slots.remove(&index) {
                            Some(Slot::Thinking { content_index, signature }) => {
                                result.add_anthropic_reasoning(AnthropicReasoning::Thinking {
                                    content_index,
                                    signature,
                                });
                            }
                            Some(Slot::ToolCall { content_index, partial_json }) => {
                                if let Content::ToolCall(call) = &mut result.content[content_index]
                                    && !partial_json.is_empty()
                                {
                                    call.arguments = parse_arguments(&partial_json);
                                }
                            }
                            _ => {}
                        }
                    }
                    StreamEvent::MessageDelta { delta, usage } => {
                        apply_usage(&mut result.usage, usage);
                        if let Some(reason) = delta.stop_reason {
                            result.stop_reason = match reason.as_str() {
                                "max_tokens" => StopReason::Length,
                                "tool_use" => StopReason::ToolUse,
                                "pause_turn" => StopReason::Pause,
                                _ => StopReason::Stop,
                            };
                            result.raw_stop_reason = Some(reason);
                        }
                    }
                StreamEvent::MessageStop => {
                        if result.stop_reason == StopReason::Pending {
                            yield Err(Error::Stream {
                                message: "message_stop arrived without a stop reason".into(),
                                partial: result,
                            });
                        } else {
                            yield Ok(Event::Done(Box::new(result)));
                        }
                    return;
                }
                StreamEvent::Error { error } => {
                    result.stop_reason = StopReason::Error;
                    result.raw_stop_reason = Some(format!("error.{}", error.r#type));
                    yield Err(Error::Response {
                        code: Some(error.r#type),
                        message: error.message,
                        partial: result,
                    });
                    return;
                }
                _ => {}
            }
        }

        yield Err(Error::IncompleteStream { partial: result });
    };
    Ok(Box::pin(output))
}

fn messages(
    model: &Model,
    context: &Context,
    cache_control: Option<&CacheControl>,
) -> Vec<RequestMessage> {
    let mut messages = context
        .messages()
        .iter()
        .map(|message| match message {
            Message::User(content) => RequestMessage {
                role: "user",
                content: content.iter().map(input_content).collect(),
            },
            Message::Assistant(response) => RequestMessage {
                role: "assistant",
                content: assistant_content(model, response),
            },
            Message::ToolResult(result) => RequestMessage {
                role: "user",
                content: vec![tool_result(result)],
            },
        })
        .collect::<Vec<_>>();
    if let Some(message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.role == "user")
        && let Some(content) = message.content.last_mut()
    {
        add_cache_control(content, cache_control);
    }
    messages
}

fn assistant_content(model: &Model, response: &Response) -> Vec<serde_json::Value> {
    let reasoning = response.anthropic_reasoning(&model.id);
    response
        .content
        .iter()
        .enumerate()
        .filter_map(|(content_index, content)| match content {
            Content::Text(text) => Some(serde_json::json!({"type": "text", "text": text})),
            Content::ToolCall(call) => Some(serde_json::json!({
                "type": "tool_use",
                "id": call.id,
                "name": call.name,
                "input": call.arguments
            })),
            Content::Reasoning(text) => reasoning.and_then(|reasoning| {
                reasoning.iter().find_map(|reasoning| match reasoning {
                    AnthropicReasoning::Thinking {
                        content_index: index,
                        signature,
                    } if *index == content_index && !signature.is_empty() => {
                        Some(serde_json::json!({
                            "type": "thinking",
                            "thinking": text,
                            "signature": signature
                        }))
                    }
                    AnthropicReasoning::Redacted {
                        content_index: index,
                        data,
                    } if *index == content_index => Some(serde_json::json!({
                        "type": "redacted_thinking",
                        "data": data
                    })),
                    _ => None,
                })
            }),
        })
        .collect()
}

fn parse_arguments(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}))
}

fn input_content(content: &InputContent) -> serde_json::Value {
    match content {
        InputContent::Text(text) => serde_json::json!({"type": "text", "text": text}),
        InputContent::Image { media_type, data } => serde_json::json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data}
        }),
    }
}

fn cache_control(retention: CacheRetention) -> Option<CacheControl> {
    match retention {
        CacheRetention::None => None,
        CacheRetention::Short => Some(CacheControl {
            r#type: "ephemeral",
            ttl: None,
        }),
        CacheRetention::Long => Some(CacheControl {
            r#type: "ephemeral",
            ttl: Some("1h"),
        }),
    }
}

fn add_cache_control(content: &mut serde_json::Value, cache_control: Option<&CacheControl>) {
    let (Some(content), Some(cache_control)) = (content.as_object_mut(), cache_control) else {
        return;
    };
    content.insert(
        "cache_control".into(),
        serde_json::to_value(cache_control).expect("cache control serializes"),
    );
}

fn thinking(thinking: &Option<Thinking>) -> (Option<serde_json::Value>, Option<serde_json::Value>) {
    match thinking {
        None => (None, None),
        Some(Thinking::Disabled) => (Some(serde_json::json!({"type": "disabled"})), None),
        Some(Thinking::Enabled {
            budget_tokens,
            display,
        }) => (
            Some(serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget_tokens,
                "display": display
            })),
            None,
        ),
        Some(Thinking::Adaptive { effort, display }) => (
            Some(serde_json::json!({"type": "adaptive", "display": display})),
            Some(serde_json::json!({"effort": effort})),
        ),
    }
}

fn tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Any => serde_json::json!({"type": "any"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Tool(name) => serde_json::json!({"type": "tool", "name": name}),
    }
}

fn tool_result(result: &ToolResult) -> serde_json::Value {
    serde_json::json!({
        "type": "tool_result",
        "tool_use_id": result.id,
        "content": result.content.iter().map(input_content).collect::<Vec<_>>(),
        "is_error": result.is_error
    })
}

fn apply_usage(usage: &mut Usage, update: AnthropicUsage) {
    if let Some(input) = update.input_tokens {
        usage.input = input;
    }
    if let Some(output) = update.output_tokens {
        usage.output = output;
    }
    if let Some(cache_read) = update.cache_read_input_tokens {
        usage.cache_read = cache_read;
    }
    if let Some(cache_write) = update.cache_creation_input_tokens {
        usage.cache_write = cache_write;
    }
    if let Some(reasoning) = update
        .output_tokens_details
        .and_then(|details| details.thinking_tokens)
    {
        usage.reasoning = reasoning;
    }
}
