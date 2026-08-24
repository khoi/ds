use crate::{
    Content, Context, Error, Event, InputContent, Message, Response, ResponseStream, StopReason,
    ToolResult, Usage, http, retry, sse, types::AnthropicReasoning,
};
use async_stream::stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};
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
    cancellation: CancellationToken,
}

impl Options {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_tokens: 4096,
            cancellation: CancellationToken::new(),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<RequestTool<'a>>,
    max_tokens: u64,
    stream: bool,
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
    let request = Request {
        model: &model.id,
        system: context.system(),
        messages: messages(model, context),
        tools: context
            .tools()
            .iter()
            .map(|tool| RequestTool {
                name: &tool.name,
                description: &tool.description,
                input_schema: &tool.parameters,
            })
            .collect(),
        max_tokens: options.max_tokens,
        stream: true,
    };
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", model.base_url.trim_end_matches('/'));
    let response = retry::send(
        retry::Policy {
            max_retries: 0,
            max_delay: Some(DEFAULT_MAX_RETRY_DELAY),
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
    )
    .await?;
    if !response.status().is_success() {
        return Err(http::provider_error(response).await);
    }

    let metadata = http::metadata(response.headers());
    let response_model = model.id.clone();
    let output = stream! {
        let mut chunks = response.bytes_stream();
        let mut decoder = sse::Decoder::default();
        let mut result = Response::anthropic(response_model);
        result.metadata = metadata;
        let mut slots = HashMap::new();

        while let Some(chunk) = chunks.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(error) => {
                    yield Err(Error::Stream(error.to_string()));
                    return;
                }
            };
            decoder.push(&chunk);
            loop {
                let data = match decoder.next_data() {
                    Ok(Some(data)) => data,
                    Ok(None) => break,
                    Err(error) => {
                        yield Err(Error::Stream(error));
                        return;
                    }
                };
                let event = match serde_json::from_str::<StreamEvent>(&data) {
                    Ok(event) => event,
                    Err(error) => {
                        yield Err(Error::Stream(error.to_string()));
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
                                _ => StopReason::Stop,
                            };
                            result.raw_stop_reason = Some(reason);
                        }
                    }
                    StreamEvent::MessageStop => {
                        if result.stop_reason == StopReason::Pending {
                            yield Err(Error::Stream("message_stop arrived without a stop reason".into()));
                        } else {
                            yield Ok(Event::Done(Box::new(result)));
                        }
                        return;
                    }
                    _ => {}
                }
            }
        }

        yield Err(Error::IncompleteStream { partial: result });
    };
    Ok(Box::pin(output))
}

fn messages(model: &Model, context: &Context) -> Vec<RequestMessage> {
    context
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
        .collect()
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
