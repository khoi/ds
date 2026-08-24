use crate::{
    CacheRetention, Content, Context, Error, Event, InputContent, Message, Response,
    ResponseMetadata, ResponseStream, StopReason, TimeoutPhase, ToolCall, ToolResult, Usage, http,
    json, retry, schema, transport,
    types::{OpenAiReplay, normalize_id},
};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, pin::Pin, time::Duration};
use tokio::time::Instant;
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
    service_tier: Option<ServiceTier>,
    connection_timeout: Option<Duration>,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_timeout: Option<Duration>,
    session_id: Option<String>,
    cache_retention: CacheRetention,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Priority,
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
            service_tier: None,
            connection_timeout: None,
            first_event_timeout: None,
            idle_timeout: None,
            overall_timeout: None,
            session_id: None,
            cache_retention: CacheRetention::Short,
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

    pub fn with_service_tier(mut self, service_tier: ServiceTier) -> Self {
        self.service_tier = Some(service_tier);
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

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
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
    input: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<RequestTool>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_retention: Option<&'static str>,
}

#[derive(Serialize)]
struct RequestTool {
    r#type: &'static str,
    name: String,
    description: String,
    parameters: serde_json::Value,
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
    #[serde(rename = "response.completed", alias = "response.done")]
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
    id: Option<String>,
    status: Option<String>,
    service_tier: Option<String>,
    end_turn: Option<bool>,
    incomplete_details: Option<IncompleteDetails>,
    error: Option<FailedDetail>,
    #[serde(default)]
    usage: CompletedUsage,
}

#[derive(Deserialize)]
struct IncompleteResponse {
    id: Option<String>,
    service_tier: Option<String>,
    end_turn: Option<bool>,
    incomplete_details: IncompleteDetails,
    #[serde(default)]
    usage: CompletedUsage,
}

#[derive(Deserialize)]
struct IncompleteDetails {
    reason: String,
}

#[derive(Deserialize)]
struct FailedResponse {
    id: Option<String>,
    service_tier: Option<String>,
    end_turn: Option<bool>,
    error: Option<FailedDetail>,
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(Deserialize)]
struct FailedDetail {
    code: Option<String>,
    message: Option<String>,
}

#[derive(Default, Deserialize)]
struct CompletedUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    input_tokens_details: InputTokenDetails,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    output_tokens_details: OutputTokenDetails,
}

#[derive(Default, Deserialize)]
struct InputTokenDetails {
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    cache_write_tokens: u64,
}

#[derive(Default, Deserialize)]
struct OutputTokenDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<ResponseStream, Error> {
    let overall_deadline = options
        .overall_timeout
        .map(|timeout| Instant::now() + timeout);
    let input = response_input(&model.id, context, true);
    let tools = context
        .tools()
        .iter()
        .map(|tool| {
            Ok(RequestTool {
                r#type: "function",
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: if tool.strict() {
                    schema::strict(&tool.parameters).map_err(|error| {
                        format!("tool {:?} has an invalid strict schema: {error}", tool.name)
                    })?
                } else {
                    tool.parameters.clone()
                },
                strict: tool.strict(),
            })
        })
        .collect::<Result<Vec<_>, String>>()
        .map_err(Error::InvalidRequest)?;
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
        service_tier: options.service_tier,
        prompt_cache_key: match options.cache_retention {
            CacheRetention::None => None,
            CacheRetention::Short | CacheRetention::Long => {
                options.session_id.as_deref().map(clamp_cache_key)
            }
        },
        prompt_cache_retention: match options.cache_retention {
            CacheRetention::Long => Some("24h"),
            CacheRetention::None | CacheRetention::Short => None,
        },
    };
    let client = reqwest::Client::new();
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
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
                    .bearer_auth(&options.api_key)
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
    Ok(decode_stream(
        response,
        model.id.clone(),
        options.cancellation.clone(),
        options.first_event_timeout,
        options.idle_timeout,
        overall_deadline,
    ))
}

fn fallback_message_id(message_index: usize, text_index: usize) -> String {
    format!("msg_ds_{message_index}_{text_index}")
}

fn openai_call_id(id: &str) -> String {
    normalize_id(id.split('|').next().unwrap_or(id))
}

pub(crate) fn response_input(
    model: &str,
    context: &Context,
    include_system: bool,
) -> Vec<serde_json::Value> {
    let mut input = Vec::new();
    if include_system && let Some(system) = context.system() {
        input.push(serde_json::json!({
            "role": "developer",
            "content": [{"type": "input_text", "text": system}]
        }));
    }
    for (message_index, message) in context.messages().iter().enumerate() {
        match message {
            Message::User(content) => input.push(serde_json::json!({
                "role": "user",
                "content": content.iter().map(response_input_content).collect::<Vec<_>>()
            })),
            Message::Assistant(response) => {
                let items = response.openai_items(model).unwrap_or_default();
                let mut text_index = 0;
                for (content_index, content) in response.content.iter().enumerate() {
                    let replay = items.iter().find(|item| match item {
                        OpenAiReplay::Reasoning {
                            content_index: index,
                            ..
                        }
                        | OpenAiReplay::Message {
                            content_index: index,
                            ..
                        }
                        | OpenAiReplay::ToolCall {
                            content_index: index,
                            ..
                        } => *index == content_index,
                    });
                    match (content, replay) {
                        (
                            Content::Reasoning(text),
                            Some(OpenAiReplay::Reasoning {
                                id,
                                encrypted_content,
                                ..
                            }),
                        ) => {
                            let mut item = serde_json::json!({
                                "type": "reasoning",
                                "id": id,
                                "summary": [{"type": "summary_text", "text": text}]
                            });
                            if let Some(encrypted_content) = encrypted_content {
                                item["encrypted_content"] = encrypted_content.clone().into();
                            }
                            input.push(item);
                        }
                        (Content::Reasoning(text) | Content::Text(text), _) if !text.is_empty() => {
                            let mut item = serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": []
                                }],
                                "status": "completed",
                                "id": fallback_message_id(message_index, text_index)
                            });
                            if let Some(OpenAiReplay::Message { id, phase, .. }) = replay {
                                item["id"] = id.clone().into();
                                if let Some(phase) = phase {
                                    item["phase"] = phase.clone().into();
                                }
                            }
                            input.push(item);
                            text_index += 1;
                        }
                        (Content::ToolCall(call), replay) => {
                            let mut item = serde_json::json!({
                                "type": "function_call",
                                "call_id": openai_call_id(&call.id),
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .expect("tool arguments serialize")
                            });
                            if let Some(OpenAiReplay::ToolCall {
                                item_id, namespace, ..
                            }) = replay
                            {
                                item["id"] = item_id.clone().into();
                                if let Some(namespace) = namespace {
                                    item["namespace"] = namespace.clone().into();
                                }
                            }
                            input.push(item);
                        }
                        _ => {}
                    }
                }
            }
            Message::ToolResult(result) => input.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": openai_call_id(&result.id),
                "output": tool_result_output(result)
            })),
        }
    }
    input
}

fn response_input_content(content: &InputContent) -> serde_json::Value {
    match content {
        InputContent::Text(text) => serde_json::json!({"type": "input_text", "text": text}),
        InputContent::Image { media_type, data } => serde_json::json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{media_type};base64,{data}")
        }),
    }
}

pub(crate) fn decode_stream(
    response: reqwest::Response,
    response_model: String,
    stream_cancellation: CancellationToken,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
) -> ResponseStream {
    let metadata = http::metadata(response.headers());
    let mut events = transport::EventStream::new(
        response,
        stream_cancellation,
        first_event_timeout,
        idle_timeout,
        overall_deadline,
    );
    let events = stream! {
        loop {
            match events.next().await {
                Ok(Some(data)) => yield Ok(data),
                Ok(None) => return,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }
        }
    };
    decode_events(Box::pin(events), response_model, metadata)
}

pub(crate) type ProviderEvents =
    Pin<Box<dyn Stream<Item = Result<String, transport::ReadError>> + Send>>;

pub(crate) fn decode_events(
    mut events: ProviderEvents,
    response_model: String,
    metadata: ResponseMetadata,
) -> ResponseStream {
    let output = stream! {
        let mut result = Response::openai(response_model);
        result.metadata = metadata;
        let mut slots = HashMap::new();

        loop {
            let data = match events.next().await {
                Some(Ok(data)) => data,
                None => break,
                Some(Err(transport::ReadError::Cancelled)) => {
                    result.stop_reason = StopReason::Aborted;
                    result.raw_stop_reason = Some("cancelled".into());
                    yield Err(Error::Cancelled { partial: Some(result) });
                    return;
                }
                Some(Err(transport::ReadError::Timeout(phase))) => {
                    result.stop_reason = StopReason::Error;
                    result.raw_stop_reason = Some(match phase {
                        TimeoutPhase::FirstEvent => "timeout.first_event".into(),
                        TimeoutPhase::Idle => "timeout.idle".into(),
                        TimeoutPhase::Overall => "timeout.overall".into(),
                        TimeoutPhase::Connection => unreachable!(),
                    });
                    yield Err(Error::Timeout {
                        phase,
                        partial: Some(result),
                    });
                    return;
                }
                Some(Err(transport::ReadError::Stream(message))) => {
                    yield Err(Error::Stream { message, partial: result });
                    return;
                }
            };
            let event = match json::parse::<StreamEvent>(&data) {
                Ok(event) => event,
                Err(error) => {
                    yield Err(Error::Stream {
                        message: error,
                        partial: result,
                    });
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
                        if response.id.is_some() {
                            result.id = response.id;
                        }
                        result.service_tier = response.service_tier;
                        result.end_turn = response.end_turn;
                        result.usage = usage(response.usage);
                        if response.status.as_deref() == Some("incomplete") {
                            let reason = response
                                .incomplete_details
                                .map(|details| details.reason)
                                .unwrap_or_else(|| "unknown".into());
                            if reason == "max_output_tokens" {
                                result.stop_reason = StopReason::Length;
                                result.raw_stop_reason =
                                    Some("incomplete.max_output_tokens".into());
                                yield Ok(Event::Done(Box::new(result)));
                            } else {
                                result.stop_reason = StopReason::Error;
                                result.raw_stop_reason = Some(format!("incomplete.{reason}"));
                                yield Err(Error::Response {
                                    code: None,
                                    message: format!("Response incomplete: {reason}"),
                                    partial: result,
                                });
                            }
                            return;
                        }
                        if matches!(response.status.as_deref(), Some("failed" | "cancelled")) {
                            let code = response.error.as_ref().and_then(|error| error.code.clone());
                            let message = response
                                .error
                                .and_then(|error| error.message)
                                .unwrap_or_else(|| "Provider response failed".into());
                            result.stop_reason = StopReason::Error;
                            result.raw_stop_reason = response.status;
                            yield Err(Error::Response {
                                code,
                                message,
                                partial: result,
                            });
                            return;
                        }
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
                        yield Ok(Event::Done(Box::new(result)));
                        return;
                    }
                    StreamEvent::Incomplete { response }
                        if response.incomplete_details.reason == "max_output_tokens" =>
                    {
                        if response.id.is_some() {
                            result.id = response.id;
                        }
                        result.service_tier = response.service_tier;
                        result.end_turn = response.end_turn;
                        result.usage = usage(response.usage);
                        result.stop_reason = StopReason::Length;
                        result.raw_stop_reason = Some("incomplete.max_output_tokens".into());
                        yield Ok(Event::Done(Box::new(result)));
                        return;
                    }
                    StreamEvent::Incomplete { response } => {
                        let reason = response.incomplete_details.reason;
                        if response.id.is_some() {
                            result.id = response.id;
                        }
                        result.service_tier = response.service_tier;
                        result.end_turn = response.end_turn;
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
                        result.service_tier = response.service_tier;
                        result.end_turn = response.end_turn;
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

        yield Err(Error::IncompleteStream { partial: result });
    };

    Box::pin(output)
}

fn parse_arguments(arguments: &str) -> serde_json::Value {
    json::value(arguments)
}

pub(crate) fn clamp_cache_key(key: &str) -> String {
    key.chars().take(64).collect()
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

pub(crate) fn tool_result_output(result: &ToolResult) -> serde_json::Value {
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
