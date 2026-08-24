use crate::{
    AssistantContent, CacheRetention, Content, Context, Error, Event, InputContent, Message,
    Response, ResponseMetadata, ResponseStream, StopReason, TimeoutPhase, ToolCall,
    ToolResultMessage, Usage, UserContent, http, json, retry, schema, transport,
    types::{OpenAiReplay, normalize_id},
};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};
use std::{collections::HashMap, pin::Pin, time::Duration};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);

pub struct Provider {
    id: crate::ProviderId,
    models: Vec<crate::Model>,
    headers: BTreeMap<String, Option<String>>,
}

impl Provider {
    pub fn new(models: impl IntoIterator<Item = crate::Model>) -> Self {
        Self {
            id: crate::ProviderId::new("openai"),
            models: models.into_iter().collect(),
            headers: BTreeMap::new(),
        }
    }

    fn request(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::StreamOptions,
        thinking: Option<crate::ThinkingLevel>,
        tool_choice: Option<crate::ToolChoice>,
    ) -> crate::AssistantMessageEventStream {
        if model.api != crate::Api::OpenAiResponses {
            let model = model.clone();
            let api = model.api.clone();
            return crate::legacy::adapt(model, async move {
                Err(Error::InvalidRequest(format!(
                    "OpenAI provider has no API implementation for {api}"
                )))
            });
        }
        let requested_model = model.clone();
        let context = context.clone();
        let options = options.clone();
        crate::legacy::adapt(requested_model.clone(), async move {
            let api_key = options
                .api_key
                .ok_or_else(|| Error::InvalidRequest("OpenAI API key is required".into()))?;
            let provider_model =
                Model::new(&requested_model.id).with_base_url(requested_model.base_url.clone());
            let mut provider_options = Options::new(api_key)
                .with_cancellation(options.cancellation)
                .with_max_retries(options.max_retries.unwrap_or_default())
                .with_max_retry_delay(options.max_retry_delay)
                .with_cache_retention(options.cache_retention);
            if let Some(max_tokens) = options.max_tokens {
                provider_options = provider_options.with_max_output_tokens(max_tokens);
            }
            if let Some(temperature) = options.temperature {
                provider_options = provider_options.with_temperature(temperature);
            }
            if let Some(timeout) = options.timeout {
                provider_options = provider_options.with_overall_timeout(timeout);
            }
            if let Some(session_id) = options.session_id {
                provider_options = provider_options.with_session_id(session_id);
            }
            if let Some(thinking) = thinking.and_then(reasoning_effort) {
                provider_options =
                    provider_options.with_reasoning(thinking, ReasoningSummary::Auto);
            }
            if let Some(tool_choice) = tool_choice {
                provider_options = provider_options.with_tool_choice(match tool_choice {
                    crate::ToolChoice::Auto => ToolChoice::Auto,
                    crate::ToolChoice::None => ToolChoice::None,
                });
            }
            stream(&provider_model, &context, &provider_options).await
        })
    }
}

impl crate::Provider for Provider {
    fn id(&self) -> &crate::ProviderId {
        &self.id
    }

    fn name(&self) -> &str {
        "OpenAI"
    }

    fn base_url(&self) -> Option<&str> {
        Some(DEFAULT_BASE_URL)
    }

    fn headers(&self) -> &BTreeMap<String, Option<String>> {
        &self.headers
    }

    fn models(&self) -> Vec<crate::Model> {
        self.models.clone()
    }

    fn stream(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::StreamOptions,
    ) -> crate::AssistantMessageEventStream {
        self.request(model, context, options, None, None)
    }

    fn stream_simple(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::SimpleStreamOptions,
    ) -> crate::AssistantMessageEventStream {
        self.request(
            model,
            context,
            &options.stream,
            options.thinking,
            Some(options.tool_choice),
        )
    }
}

pub fn provider(models: impl IntoIterator<Item = crate::Model>) -> Arc<dyn crate::Provider> {
    Arc::new(Provider::new(models))
}

fn reasoning_effort(level: crate::ThinkingLevel) -> Option<ReasoningEffort> {
    match level {
        crate::ThinkingLevel::Off => None,
        crate::ThinkingLevel::Minimal => Some(ReasoningEffort::Minimal),
        crate::ThinkingLevel::Low => Some(ReasoningEffort::Low),
        crate::ThinkingLevel::Medium => Some(ReasoningEffort::Medium),
        crate::ThinkingLevel::High => Some(ReasoningEffort::High),
        crate::ThinkingLevel::XHigh => Some(ReasoningEffort::XHigh),
        crate::ThinkingLevel::Max => Some(ReasoningEffort::Max),
    }
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_options: Option<PromptCacheOptions>,
}

#[derive(Serialize)]
struct PromptCacheOptions {
    mode: &'static str,
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
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta { output_index: usize, delta: String },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta { output_index: usize, delta: String },
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
    #[serde(default)]
    refusal: String,
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
    #[serde(flatten)]
    terminal: TerminalResponse,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
    error: Option<FailedDetail>,
}

#[derive(Deserialize)]
struct IncompleteResponse {
    #[serde(flatten)]
    terminal: TerminalResponse,
    incomplete_details: IncompleteDetails,
}

#[derive(Deserialize)]
struct TerminalResponse {
    id: Option<String>,
    service_tier: Option<String>,
    end_turn: Option<bool>,
    #[serde(default)]
    usage: CompletedUsage,
    #[serde(default)]
    output: Vec<OutputItem>,
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
        prompt_cache_options: matches!(options.cache_retention, CacheRetention::None)
            .then_some(PromptCacheOptions { mode: "explicit" }),
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
        let metadata = http::metadata(response.headers());
        return Err(http::provider_error(
            response,
            metadata,
            &options.cancellation,
            overall_deadline,
        )
        .await);
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
            Message::User(message) => {
                let content = match &message.content {
                    UserContent::Text(text) => vec![serde_json::json!({
                        "type": "input_text",
                        "text": text
                    })],
                    UserContent::Blocks(content) => {
                        content.iter().map(response_input_content).collect()
                    }
                };
                if !content.is_empty() {
                    input.push(serde_json::json!({"role": "user", "content": content}));
                }
            }
            Message::Assistant(message) => {
                let mut text_index = 0;
                for content in &message.content {
                    match content {
                        AssistantContent::Thinking(thinking) => {
                            if let Some(signature) = &thinking.thinking_signature
                                && let Ok(item) = serde_json::from_str(signature)
                            {
                                input.push(item);
                            }
                        }
                        AssistantContent::Text(text) if !text.text.is_empty() => {
                            let signature = text
                                .text_signature
                                .as_deref()
                                .and_then(parse_text_signature);
                            let mut item = serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text.text,
                                    "annotations": []
                                }],
                                "status": "completed",
                                "id": fallback_message_id(message_index, text_index)
                            });
                            if let Some((id, phase)) = signature {
                                item["id"] = id.into();
                                if let Some(phase) = phase {
                                    item["phase"] = phase.into();
                                }
                            }
                            input.push(item);
                            text_index += 1;
                        }
                        AssistantContent::ToolCall(call) => {
                            let mut ids = call.id.splitn(2, '|');
                            let call_id = ids.next().unwrap_or(&call.id);
                            let mut item = serde_json::json!({
                                "type": "function_call",
                                "call_id": openai_call_id(call_id),
                                "name": call.name,
                                "arguments": serde_json::to_string(&call.arguments)
                                    .expect("tool arguments serialize")
                            });
                            if let Some(item_id) = ids.next()
                                && message.model == model
                            {
                                item["id"] = item_id.into();
                            }
                            if message.model == model
                                && let Some(namespace) = &call.namespace
                            {
                                item["namespace"] = namespace.clone().into();
                            }
                            input.push(item);
                        }
                        _ => {}
                    }
                }
            }
            Message::ToolResult(result) => input.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": openai_call_id(&result.tool_call_id),
                "output": tool_result_output(result)
            })),
        }
    }
    input
}

fn response_input_content(content: &InputContent) -> serde_json::Value {
    match content {
        InputContent::Text(text) => serde_json::json!({"type": "input_text", "text": text.text}),
        InputContent::Image(image) => serde_json::json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{};base64,{}", image.mime_type, image.data)
        }),
    }
}

fn parse_text_signature(signature: &str) -> Option<(String, Option<String>)> {
    let value: serde_json::Value = serde_json::from_str(signature).ok()?;
    if value.get("v").and_then(serde_json::Value::as_u64) != Some(1) {
        return None;
    }
    let id = value.get("id")?.as_str()?.to_owned();
    let phase = value
        .get("phase")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    Some((id, phase))
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
                    StreamEvent::OutputTextDelta { output_index, delta }
                    | StreamEvent::RefusalDelta { output_index, delta } => {
                        let Some(Slot::Text(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        if let Content::Text(text) = &mut result.content[*content_index] {
                            text.push_str(&delta);
                        }
                        yield Ok(Event::TextDelta { content_index: *content_index, delta });
                    }
                    StreamEvent::ReasoningSummaryTextDelta { output_index, delta }
                    | StreamEvent::ReasoningTextDelta { output_index, delta } => {
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
                            .filter_map(|content| match content.r#type.as_str() {
                                "output_text" => Some(content.text.as_str()),
                                "refusal" => Some(content.refusal.as_str()),
                                _ => None,
                            })
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
                        let reasoning = if item.summary.is_empty() {
                            item.content
                                .iter()
                                .map(|content| content.text.as_str())
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        } else {
                            item.summary
                                .iter()
                                .map(|content| content.text.as_str())
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        };
                        if !reasoning.is_empty() {
                            result.content[*content_index] = Content::Reasoning(reasoning);
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
                        apply_terminal_response(&mut result, response.terminal);
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
                    StreamEvent::Incomplete { response } => {
                        let reason = response.incomplete_details.reason;
                        apply_terminal_response(&mut result, response.terminal);
                        if reason == "max_output_tokens" {
                            result.stop_reason = StopReason::Length;
                            result.raw_stop_reason = Some("incomplete.max_output_tokens".into());
                            yield Ok(Event::Done(Box::new(result)));
                            return;
                        }
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

fn backfill_reasoning(result: &mut Response, output: &[OutputItem]) {
    for item in output {
        let (Some(id), Some(encrypted)) = (&item.id, &item.encrypted_content) else {
            continue;
        };
        if item.r#type == "reasoning" {
            result.backfill_openai_reasoning(id, encrypted);
        }
    }
}

fn apply_terminal_response(result: &mut Response, response: TerminalResponse) {
    backfill_reasoning(result, &response.output);
    if response.id.is_some() {
        result.id = response.id;
    }
    result.service_tier = response.service_tier;
    result.end_turn = response.end_turn;
    result.usage = usage(response.usage);
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
        cache_write_1h: None,
        reasoning: Some(usage.output_tokens_details.reasoning_tokens),
        total_tokens: usage.input_tokens + usage.output_tokens,
        cost: Default::default(),
    }
}

pub(crate) fn tool_result_output(result: &ToolResultMessage) -> serde_json::Value {
    let text = result
        .content
        .iter()
        .filter_map(|content| match content {
            InputContent::Text(text) => Some(text.text.as_str()),
            InputContent::Image(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let images = result
        .content
        .iter()
        .filter_map(|content| match content {
            InputContent::Image(image) => Some((&image.mime_type, &image.data)),
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
