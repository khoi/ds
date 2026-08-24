use crate::{
    Content, Context, Error, Event, InputContent, Message, Response, ResponseStream, StopReason,
    ToolResult, Usage, http, retry, sse,
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
}

#[derive(Deserialize)]
struct ContentDelta {
    r#type: String,
    #[serde(default)]
    text: String,
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

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<ResponseStream, Error> {
    let request = Request {
        model: &model.id,
        system: context.system(),
        messages: messages(context),
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
                        slots.insert(index, content_index);
                    }
                    StreamEvent::ContentBlockDelta { index, delta }
                        if delta.r#type == "text_delta" =>
                    {
                        let Some(content_index) = slots.get(&index) else {
                            continue;
                        };
                        if let Content::Text(text) = &mut result.content[*content_index] {
                            text.push_str(&delta.text);
                        }
                        yield Ok(Event::TextDelta {
                            content_index: *content_index,
                            delta: delta.text,
                        });
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

fn messages(context: &Context) -> Vec<RequestMessage> {
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
                content: response
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        Content::Text(text) => {
                            Some(serde_json::json!({"type": "text", "text": text}))
                        }
                        Content::ToolCall(call) => Some(serde_json::json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments
                        })),
                        Content::Reasoning(_) => None,
                    })
                    .collect(),
            },
            Message::ToolResult(result) => RequestMessage {
                role: "user",
                content: vec![tool_result(result)],
            },
        })
        .collect()
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
