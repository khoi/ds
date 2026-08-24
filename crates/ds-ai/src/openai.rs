use crate::{
    Content, Context, Error, Event, InputContent, Message, Response, ResponseStream, StopReason,
    ToolCall, ToolResult, Usage, retry, sse, types::OpenAiReplay,
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
    max_output_tokens: Option<u64>,
    temperature: Option<f64>,
    reasoning: Option<Reasoning>,
    tool_choice: Option<ToolChoice>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningSummary {
    Auto,
    Detailed,
    Concise,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
}

#[derive(Clone, Copy, Debug)]
struct Reasoning {
    effort: ReasoningEffort,
    summary: ReasoningSummary,
}

impl Options {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
            max_output_tokens: None,
            temperature: None,
            reasoning: None,
            tool_choice: None,
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

    pub fn with_max_output_tokens(mut self, max_output_tokens: u64) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_reasoning(mut self, effort: ReasoningEffort, summary: ReasoningSummary) -> Self {
        self.reasoning = Some(Reasoning { effort, summary });
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    input: Vec<RequestItem<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<RequestTool<'a>>,
    stream: bool,
    store: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<RequestReasoningOptions>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    include: Vec<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum RequestItem<'a> {
    User(RequestUser<'a>),
    Reasoning(RequestReasoning<'a>),
    Assistant(RequestAssistant<'a>),
    FunctionCall(RequestFunctionCall<'a>),
    FunctionOutput(RequestFunctionOutput<'a>),
}

#[derive(Serialize)]
struct RequestUser<'a> {
    role: &'static str,
    content: Vec<RequestInputContent<'a>>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum RequestInputContent<'a> {
    Text {
        r#type: &'static str,
        text: &'a str,
    },
    Image {
        r#type: &'static str,
        detail: &'static str,
        image_url: String,
    },
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

#[derive(Serialize)]
struct RequestFunctionCall<'a> {
    r#type: &'static str,
    id: &'a str,
    call_id: &'a str,
    name: &'a str,
    arguments: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    namespace: Option<&'a str>,
}

#[derive(Serialize)]
struct RequestFunctionOutput<'a> {
    r#type: &'static str,
    call_id: &'a str,
    output: serde_json::Value,
}

#[derive(Serialize)]
struct RequestTool<'a> {
    r#type: &'static str,
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
    strict: bool,
}

#[derive(Clone, Copy, Serialize)]
struct RequestReasoningOptions {
    effort: ReasoningEffort,
    summary: ReasoningSummary,
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
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { output_index: usize, delta: String },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        output_index: usize,
        arguments: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.completed")]
    Completed { response: CompletedResponse },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: IncompleteResponse },
    #[serde(rename = "response.failed")]
    Failed { response: FailedResponse },
    #[serde(rename = "error")]
    Error {
        code: Option<String>,
        message: String,
    },
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
    call_id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
    #[serde(default)]
    content: Vec<OutputContent>,
    #[serde(default)]
    summary: Vec<SummaryContent>,
    encrypted_content: Option<String>,
    phase: Option<String>,
    namespace: Option<String>,
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
    ToolCall {
        content_index: usize,
        arguments: String,
    },
}

#[derive(Deserialize)]
struct CompletedResponse {
    id: String,
    usage: CompletedUsage,
}

#[derive(Deserialize)]
struct IncompleteResponse {
    id: String,
    incomplete_details: IncompleteDetails,
    usage: CompletedUsage,
}

#[derive(Deserialize)]
struct IncompleteDetails {
    reason: String,
}

#[derive(Deserialize)]
struct FailedResponse {
    id: Option<String>,
    error: Option<FailedDetail>,
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(Deserialize)]
struct FailedDetail {
    code: Option<String>,
    message: Option<String>,
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
    if let Some(system) = context.system() {
        input.push(RequestItem::User(RequestUser {
            role: "developer",
            content: vec![RequestInputContent::Text {
                r#type: "input_text",
                text: system,
            }],
        }));
    }
    for message in context.messages() {
        match message {
            Message::User(content) => input.push(RequestItem::User(RequestUser {
                role: "user",
                content: request_input_content(content),
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
                        Content::ToolCall(call) => {
                            let Some(OpenAiReplay::ToolCall {
                                item_id, namespace, ..
                            }) = items.iter().find(|item| {
                                matches!(
                                    item,
                                    OpenAiReplay::ToolCall {
                                        content_index: index,
                                        ..
                                    } if *index == content_index
                                )
                            })
                            else {
                                continue;
                            };
                            input.push(RequestItem::FunctionCall(RequestFunctionCall {
                                r#type: "function_call",
                                id: item_id,
                                call_id: &call.id,
                                name: &call.name,
                                arguments: serde_json::to_string(&call.arguments)
                                    .expect("tool arguments serialize"),
                                namespace: namespace.as_deref(),
                            }));
                        }
                    }
                }
            }
            Message::ToolResult(result) => {
                input.push(RequestItem::FunctionOutput(RequestFunctionOutput {
                    r#type: "function_call_output",
                    call_id: &result.id,
                    output: tool_result_output(result),
                }));
            }
        }
    }
    let tools = context
        .tools()
        .iter()
        .map(|tool| RequestTool {
            r#type: "function",
            name: &tool.name,
            description: &tool.description,
            parameters: &tool.parameters,
            strict: false,
        })
        .collect();
    let request = Request {
        model: &model.id,
        input,
        tools,
        stream: true,
        store: false,
        max_output_tokens: options.max_output_tokens,
        temperature: options.temperature,
        reasoning: options.reasoning.map(|reasoning| RequestReasoningOptions {
            effort: reasoning.effort,
            summary: reasoning.summary,
        }),
        include: options
            .reasoning
            .map(|_| vec!["reasoning.encrypted_content"])
            .unwrap_or_default(),
        tool_choice: options.tool_choice,
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
    let stream_cancellation = options.cancellation.clone();
    let output = stream! {
        let mut chunks = response.bytes_stream();
        let mut decoder = sse::Decoder::default();
        let mut result = Response::openai(response_model);
        let mut slots = HashMap::new();

        loop {
            let chunk = tokio::select! {
                biased;
                _ = stream_cancellation.cancelled() => {
                    result.stop_reason = StopReason::Aborted;
                    result.raw_stop_reason = Some("cancelled".into());
                    yield Err(Error::Cancelled { partial: Some(result) });
                    return;
                }
                chunk = chunks.next() => chunk,
            };
            let Some(chunk) = chunk else {
                break;
            };
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
                            "function_call" => {
                                let (Some(id), Some(name)) = (item.call_id, item.name) else {
                                    continue;
                                };
                                let arguments = item.arguments.unwrap_or_default();
                                result.content.push(Content::ToolCall(ToolCall {
                                    id,
                                    name,
                                    arguments: parse_arguments(&arguments),
                                }));
                                slots.insert(
                                    output_index,
                                    Slot::ToolCall {
                                        content_index,
                                        arguments,
                                    },
                                );
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
                    StreamEvent::FunctionCallArgumentsDelta { output_index, delta } => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get_mut(&output_index) else {
                            continue;
                        };
                        arguments.push_str(&delta);
                        if let Content::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = parse_arguments(arguments);
                        }
                        yield Ok(Event::ToolCallDelta { content_index: *content_index, delta });
                    }
                    StreamEvent::FunctionCallArgumentsDone { output_index, arguments: completed } => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get_mut(&output_index) else {
                            continue;
                        };
                        let delta = completed
                            .strip_prefix(arguments.as_str())
                            .filter(|delta| !delta.is_empty())
                            .map(str::to_owned);
                        *arguments = completed;
                        if let Content::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = parse_arguments(arguments);
                        }
                        if let Some(delta) = delta {
                            yield Ok(Event::ToolCallDelta { content_index: *content_index, delta });
                        }
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
                    StreamEvent::OutputItemDone { output_index, item } if item.r#type == "function_call" => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get(&output_index) else {
                            continue;
                        };
                        let final_arguments = item.arguments.as_deref().unwrap_or(arguments);
                        if let Content::ToolCall(call) = &mut result.content[*content_index] {
                            if let Some(id) = item.call_id {
                                call.id = id;
                            }
                            if let Some(name) = item.name {
                                call.name = name;
                            }
                            call.arguments = parse_arguments(final_arguments);
                        }
                        if let Some(item_id) = item.id {
                            result.add_openai_item(OpenAiReplay::ToolCall {
                                content_index: *content_index,
                                item_id,
                                namespace: item.namespace,
                            });
                        }
                    }
                    StreamEvent::Completed { response } => {
                        result.id = Some(response.id);
                        result.usage = usage(response.usage);
                        result.stop_reason = if result
                            .content
                            .iter()
                            .any(|content| matches!(content, Content::ToolCall(_)))
                        {
                            StopReason::ToolUse
                        } else {
                            StopReason::Stop
                        };
                        result.raw_stop_reason = Some("completed".into());
                        yield Ok(Event::Done(result));
                        return;
                    }
                    StreamEvent::Incomplete { response }
                        if response.incomplete_details.reason == "max_output_tokens" =>
                    {
                        result.id = Some(response.id);
                        result.usage = usage(response.usage);
                        result.stop_reason = StopReason::Length;
                        result.raw_stop_reason = Some("incomplete.max_output_tokens".into());
                        yield Ok(Event::Done(result));
                        return;
                    }
                    StreamEvent::Incomplete { response } => {
                        let reason = response.incomplete_details.reason;
                        result.id = Some(response.id);
                        result.usage = usage(response.usage);
                        result.stop_reason = StopReason::Error;
                        result.raw_stop_reason = Some(format!("incomplete.{reason}"));
                        yield Err(Error::Response {
                            code: None,
                            message: format!("Response incomplete: {reason}"),
                            partial: result,
                        });
                        return;
                    }
                    StreamEvent::Failed { response } => {
                        let code = response.error.as_ref().and_then(|error| error.code.clone());
                        let message = response
                            .error
                            .and_then(|error| error.message)
                            .or_else(|| {
                                response
                                    .incomplete_details
                                    .map(|details| format!("Response incomplete: {}", details.reason))
                            })
                            .unwrap_or_else(|| "Unknown provider error".into());
                        if response.id.is_some() {
                            result.id = response.id;
                        }
                        result.stop_reason = StopReason::Error;
                        result.raw_stop_reason = Some("failed".into());
                        yield Err(Error::Response {
                            code,
                            message,
                            partial: result,
                        });
                        return;
                    }
                    StreamEvent::Error { code, message } => {
                        result.stop_reason = StopReason::Error;
                        result.raw_stop_reason = Some("error".into());
                        yield Err(Error::Response {
                            code,
                            message,
                            partial: result,
                        });
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

fn parse_arguments(arguments: &str) -> serde_json::Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}))
}

fn request_input_content(content: &[InputContent]) -> Vec<RequestInputContent<'_>> {
    content
        .iter()
        .map(|content| match content {
            InputContent::Text(text) => RequestInputContent::Text {
                r#type: "input_text",
                text,
            },
            InputContent::Image { media_type, data } => RequestInputContent::Image {
                r#type: "input_image",
                detail: "auto",
                image_url: format!("data:{media_type};base64,{data}"),
            },
        })
        .collect()
}

fn usage(usage: CompletedUsage) -> Usage {
    Usage {
        input: usage
            .input_tokens
            .saturating_sub(usage.input_tokens_details.cached_tokens)
            .saturating_sub(usage.input_tokens_details.cache_write_tokens),
        output: usage.output_tokens,
        cache_read: usage.input_tokens_details.cached_tokens,
        cache_write: usage.input_tokens_details.cache_write_tokens,
        reasoning: usage.output_tokens_details.reasoning_tokens,
    }
}

fn tool_result_output(result: &ToolResult) -> serde_json::Value {
    let text = result
        .content
        .iter()
        .filter_map(|content| match content {
            InputContent::Text(text) => Some(text.as_str()),
            InputContent::Image { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images = result
        .content
        .iter()
        .filter_map(|content| match content {
            InputContent::Image { media_type, data } => Some((media_type, data)),
            InputContent::Text(_) => None,
        })
        .collect::<Vec<_>>();
    if images.is_empty() {
        return serde_json::Value::String(if text.is_empty() {
            "(no tool output)".into()
        } else {
            text
        });
    }

    let mut output = Vec::new();
    if !text.is_empty() {
        output.push(serde_json::json!({"type": "input_text", "text": text}));
    }
    output.extend(images.into_iter().map(|(media_type, data)| {
        serde_json::json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{media_type};base64,{data}")
        })
    }));
    serde_json::Value::Array(output)
}
