use crate::{
    Content, Context, Error, Event, Message, Response, ResponseStream, Usage, retry, sse,
    types::OpenAiReplay,
};
use async_stream::stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
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
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
}

impl Options {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
        }
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn with_max_retry_delay(mut self, max_retry_delay: Option<Duration>) -> Self {
        self.max_retry_delay = max_retry_delay;
        self
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: Vec<RequestItem<'a>>,
    stream: bool,
    store: bool,
}

#[derive(Serialize)]
#[serde(untagged)]
enum RequestItem<'a> {
    User(RequestUser<'a>),
    Reasoning(RequestReasoning<'a>),
    Assistant(RequestAssistant<'a>),
}

#[derive(Serialize)]
struct RequestUser<'a> {
    role: &'static str,
    content: [RequestUserContent<'a>; 1],
}

#[derive(Serialize)]
struct RequestUserContent<'a> {
    r#type: &'static str,
    text: &'a str,
}

#[derive(Serialize)]
struct RequestReasoning<'a> {
    r#type: &'static str,
    id: &'a str,
    summary: [RequestSummary<'a>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    encrypted_content: Option<&'a str>,
}

#[derive(Serialize)]
struct RequestSummary<'a> {
    r#type: &'static str,
    text: &'a str,
}

#[derive(Serialize)]
struct RequestAssistant<'a> {
    r#type: &'static str,
    role: &'static str,
    content: [RequestAssistantContent<'a>; 1],
    status: &'static str,
    id: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    phase: Option<&'a str>,
}

#[derive(Serialize)]
struct RequestAssistantContent<'a> {
    r#type: &'static str,
    text: &'a str,
    annotations: [(); 0],
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    #[serde(rename = "response.created")]
    Created { response: IdentifiedResponse },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta { output_index: usize, delta: String },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta { output_index: usize, delta: String },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.completed")]
    Completed { response: CompletedResponse },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct IdentifiedResponse {
    id: String,
}

#[derive(Deserialize)]
struct OutputItem {
    id: Option<String>,
    r#type: String,
    #[serde(default)]
    content: Vec<OutputContent>,
    #[serde(default)]
    summary: Vec<SummaryContent>,
    encrypted_content: Option<String>,
    phase: Option<String>,
}

#[derive(Deserialize)]
struct OutputContent {
    r#type: String,
    #[serde(default)]
    text: String,
}

#[derive(Deserialize)]
struct SummaryContent {
    text: String,
}

enum Slot {
    Text(usize),
    Reasoning(usize),
}

#[derive(Deserialize)]
struct CompletedResponse {
    id: String,
    usage: CompletedUsage,
}

#[derive(Deserialize)]
struct CompletedUsage {
    input_tokens: u64,
    input_tokens_details: InputTokenDetails,
    output_tokens: u64,
    output_tokens_details: OutputTokenDetails,
}

#[derive(Deserialize)]
struct InputTokenDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: u64,
}

#[derive(Deserialize)]
struct OutputTokenDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<ResponseStream, Error> {
    let mut input = Vec::new();
    for message in context.messages() {
        match message {
            Message::User(text) => input.push(RequestItem::User(RequestUser {
                role: "user",
                content: [RequestUserContent {
                    r#type: "input_text",
                    text,
                }],
            })),
            Message::Assistant(response) => {
                let Some(items) = response.openai_items(&model.id) else {
                    continue;
                };
                for (content_index, content) in response.content.iter().enumerate() {
                    match content {
                        Content::Reasoning(text) => {
                            let Some(OpenAiReplay::Reasoning {
                                id,
                                encrypted_content,
                                ..
                            }) = items.iter().find(|item| {
                                matches!(
                                    item,
                                    OpenAiReplay::Reasoning {
                                        content_index: index,
                                        ..
                                    } if *index == content_index
                                )
                            })
                            else {
                                continue;
                            };
                            input.push(RequestItem::Reasoning(RequestReasoning {
                                r#type: "reasoning",
                                id,
                                summary: [RequestSummary {
                                    r#type: "summary_text",
                                    text,
                                }],
                                encrypted_content: encrypted_content.as_deref(),
                            }));
                        }
                        Content::Text(text) => {
                            let Some(OpenAiReplay::Message { id, phase, .. }) =
                                items.iter().find(|item| {
                                    matches!(
                                        item,
                                        OpenAiReplay::Message {
                                            content_index: index,
                                            ..
                                        } if *index == content_index
                                    )
                                })
                            else {
                                continue;
                            };
                            input.push(RequestItem::Assistant(RequestAssistant {
                                r#type: "message",
                                role: "assistant",
                                content: [RequestAssistantContent {
                                    r#type: "output_text",
                                    text,
                                    annotations: [],
                                }],
                                status: "completed",
                                id,
                                phase: phase.as_deref(),
                            }));
                        }
                    }
                }
            }
        }
    }
    let request = Request {
        model: &model.id,
        input,
        stream: true,
        store: false,
    };
    let client = reqwest::Client::new();
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
    let response = retry::send(
        retry::Policy {
            max_retries: options.max_retries,
            max_delay: options.max_retry_delay,
            cancellation: &options.cancellation,
        },
        || {
            client
                .post(&url)
                .bearer_auth(&options.api_key)
                .json(&request)
                .send()
        },
    )
    .await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Provider {
            status: status.as_u16(),
            body,
        });
    }

    let response_model = model.id.clone();
    let output = stream! {
        let mut chunks = response.bytes_stream();
        let mut decoder = sse::Decoder::default();
        let mut result = Response::openai(response_model);
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
                    StreamEvent::Created { response } => result.id = Some(response.id),
                    StreamEvent::OutputItemAdded { output_index, item } => {
                        let content_index = result.content.len();
                        match item.r#type.as_str() {
                            "message" => {
                                result.content.push(Content::Text(String::new()));
                                slots.insert(output_index, Slot::Text(content_index));
                            }
                            "reasoning" => {
                                result.content.push(Content::Reasoning(String::new()));
                                slots.insert(output_index, Slot::Reasoning(content_index));
                            }
                            _ => {}
                        }
                    }
                    StreamEvent::OutputTextDelta { output_index, delta } => {
                        let Some(Slot::Text(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        if let Content::Text(text) = &mut result.content[*content_index] {
                            text.push_str(&delta);
                        }
                        yield Ok(Event::TextDelta { content_index: *content_index, delta });
                    }
                    StreamEvent::ReasoningSummaryTextDelta { output_index, delta } => {
                        let Some(Slot::Reasoning(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        if let Content::Reasoning(reasoning) = &mut result.content[*content_index] {
                            reasoning.push_str(&delta);
                        }
                        yield Ok(Event::ReasoningDelta { content_index: *content_index, delta });
                    }
                    StreamEvent::OutputItemDone { output_index, item } if item.r#type == "message" => {
                        let Some(Slot::Text(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        let text = item
                            .content
                            .iter()
                            .filter(|content| content.r#type == "output_text")
                            .map(|content| content.text.as_str())
                            .collect::<String>();
                        if !text.is_empty() {
                            result.content[*content_index] = Content::Text(text);
                        }
                        if let Some(id) = item.id {
                            result.add_openai_item(OpenAiReplay::Message {
                                content_index: *content_index,
                                id,
                                phase: item.phase,
                            });
                        }
                    }
                    StreamEvent::OutputItemDone { output_index, item } if item.r#type == "reasoning" => {
                        let Some(Slot::Reasoning(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        let summary = item
                            .summary
                            .iter()
                            .map(|content| content.text.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        if !summary.is_empty() {
                            result.content[*content_index] = Content::Reasoning(summary);
                        }
                        if let Some(id) = item.id {
                            result.add_openai_item(OpenAiReplay::Reasoning {
                                content_index: *content_index,
                                id,
                                encrypted_content: item.encrypted_content,
                            });
                        }
                    }
                    StreamEvent::Completed { response } => {
                        result.id = Some(response.id);
                        result.usage = Usage {
                            input: response
                                .usage
                                .input_tokens
                                .saturating_sub(response.usage.input_tokens_details.cached_tokens)
                                .saturating_sub(response.usage.input_tokens_details.cache_write_tokens),
                            output: response.usage.output_tokens,
                            cache_read: response.usage.input_tokens_details.cached_tokens,
                            cache_write: response.usage.input_tokens_details.cache_write_tokens,
                            reasoning: response.usage.output_tokens_details.reasoning_tokens,
                        };
                        yield Ok(Event::Done(result));
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
