use crate::{Content, Context, Error, Event, Message, Response, ResponseStream, Usage, sse};
use async_stream::stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

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
}

impl Options {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_retries: 0,
        }
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: Vec<RequestMessage<'a>>,
    stream: bool,
    store: bool,
}

#[derive(Serialize)]
struct RequestMessage<'a> {
    role: &'static str,
    content: Vec<RequestContent<'a>>,
}

#[derive(Serialize)]
struct RequestContent<'a> {
    r#type: &'static str,
    text: &'a str,
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
    r#type: String,
    #[serde(default)]
    content: Vec<OutputContent>,
}

#[derive(Deserialize)]
struct OutputContent {
    r#type: String,
    #[serde(default)]
    text: String,
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
    let input = context
        .messages()
        .iter()
        .map(|message| match message {
            Message::User(text) => RequestMessage {
                role: "user",
                content: vec![RequestContent {
                    r#type: "input_text",
                    text,
                }],
            },
        })
        .collect();
    let request = Request {
        model: &model.id,
        input,
        stream: true,
        store: false,
    };
    let client = reqwest::Client::new();
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
    let mut retries = 0;
    let response = loop {
        let response = client
            .post(&url)
            .bearer_auth(&options.api_key)
            .json(&request)
            .send()
            .await
            .map_err(|error| Error::Http(error.to_string()))?;
        let status = response.status();
        if status.is_success() {
            break response;
        }
        if retries < options.max_retries && matches!(status.as_u16(), 408 | 409 | 429 | 500..=599) {
            retries += 1;
            continue;
        }
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Provider {
            status: status.as_u16(),
            body,
        });
    };

    let output = stream! {
        let mut chunks = response.bytes_stream();
        let mut decoder = sse::Decoder::default();
        let mut result = Response::default();
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
                    StreamEvent::OutputItemAdded { output_index, item } if item.r#type == "message" => {
                        let content_index = result.content.len();
                        result.content.push(Content::Text(String::new()));
                        slots.insert(output_index, content_index);
                    }
                    StreamEvent::OutputTextDelta { output_index, delta } => {
                        let Some(&content_index) = slots.get(&output_index) else {
                            continue;
                        };
                        let Content::Text(text) = &mut result.content[content_index];
                        text.push_str(&delta);
                        yield Ok(Event::TextDelta { content_index, delta });
                    }
                    StreamEvent::OutputItemDone { output_index, item } if item.r#type == "message" => {
                        let Some(&content_index) = slots.get(&output_index) else {
                            continue;
                        };
                        let text = item
                            .content
                            .iter()
                            .filter(|content| content.r#type == "output_text")
                            .map(|content| content.text.as_str())
                            .collect::<String>();
                        if !text.is_empty() {
                            result.content[content_index] = Content::Text(text);
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
