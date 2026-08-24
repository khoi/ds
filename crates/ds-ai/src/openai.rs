use crate::{
    AssistantContent, CacheRetention, Content, Context, Error, Event, InputContent, Message,
    Response, ResponseMetadata, ResponseStream, StopReason, TimeoutPhase, ToolCall,
    ToolResultMessage, Usage, UserContent,
    constrained_sampling::{self, GrammarInputBuffer},
    deferred_tools::{DeferredToolsMode, ToolPlacement},
    http, json, retry, transport,
    types::{OpenAiReplay, normalize_id},
};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};
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
    auth: crate::ProviderAuth,
}

impl Provider {
    pub fn new(models: impl IntoIterator<Item = crate::Model>) -> Self {
        Self {
            id: crate::ProviderId::new("openai"),
            models: models.into_iter().collect(),
            headers: BTreeMap::new(),
            auth: crate::ProviderAuth::api_key(crate::EnvApiKeyAuth::new(
                "OpenAI API key",
                ["OPENAI_API_KEY"],
            )),
        }
    }

    fn request(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::OpenAiResponsesOptions,
    ) -> crate::AssistantMessageEventStream {
        let requested_model = model.clone();
        let context = context.clone();
        let options = options.clone();
        crate::legacy::adapt(requested_model.clone(), async move {
            let stream_options = options.stream;
            let request_hooks = stream_options.request_hooks(&requested_model);
            let api_key = stream_options
                .api_key
                .ok_or_else(|| Error::InvalidRequest("OpenAI API key is required".into()))?;
            let provider_model =
                Model::new(&requested_model.id).with_base_url(requested_model.base_url.clone());
            let mut provider_options = Options::new(api_key)
                .with_cancellation(stream_options.cancellation)
                .with_max_retries(stream_options.max_retries.unwrap_or_default())
                .with_max_retry_delay(stream_options.max_retry_delay)
                .with_cache_retention(stream_options.cache_retention)
                .with_request_options(stream_options.headers, request_hooks);
            let compat = match &requested_model.compat {
                Some(crate::ModelCompatibility::OpenAi(compat)) => compat.clone(),
                _ => Default::default(),
            };
            let mode = if compat.supports_additional_tools == Some(true) {
                Some(DeferredToolsMode::AdditionalTools)
            } else if compat.supports_tool_search == Some(true) {
                Some(DeferredToolsMode::ToolSearch)
            } else {
                None
            };
            provider_options = provider_options
                .with_deferred_tools_mode(mode)
                .with_compatibility(&compat);
            if let Some(max_tokens) = stream_options.max_tokens {
                provider_options = provider_options.with_max_output_tokens(max_tokens);
            }
            provider_options =
                provider_options.with_sampling_params(stream_options.sampling_params);
            if let Some(temperature) = stream_options.temperature {
                provider_options = provider_options.with_temperature(temperature);
            }
            if let Some(timeout) = stream_options.timeout {
                provider_options = provider_options.with_overall_timeout(timeout);
            }
            if let Some(session_id) = stream_options.session_id {
                provider_options = provider_options.with_session_id(session_id);
            }
            if requested_model.reasoning {
                if options.reasoning_effort.is_some() || options.reasoning_summary.is_some() {
                    let effort = options.reasoning_effort.map_or_else(
                        || "medium".into(),
                        |effort| mapped_reasoning_effort(&requested_model, effort),
                    );
                    provider_options = provider_options.with_reasoning_value(
                        effort,
                        Some(options.reasoning_summary.unwrap_or(ReasoningSummary::Auto)),
                    );
                } else if let Some(effort) = default_reasoning_effort(&requested_model) {
                    provider_options = provider_options.with_reasoning_value(effort, None);
                }
            }
            if let Some(service_tier) = options.service_tier {
                provider_options = provider_options.with_service_tier(service_tier);
            }
            if let Some(tool_choice) = options.tool_choice {
                provider_options = provider_options.with_tool_choice(tool_choice);
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

    fn auth(&self) -> &crate::ProviderAuth {
        &self.auth
    }

    fn models(&self) -> Vec<crate::Model> {
        self.models.clone()
    }

    fn stream(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::ApiStreamOptions,
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
        let crate::ApiStreamOptions::OpenAiResponses(options) = options else {
            let model = model.clone();
            return crate::legacy::adapt(model, async {
                Err(Error::InvalidRequest(
                    "OpenAI Responses options are required".into(),
                ))
            });
        };
        self.request(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::SimpleStreamOptions,
    ) -> crate::AssistantMessageEventStream {
        let stream =
            crate::provider::build_simple_stream_options(model, context, options.stream.clone());
        self.request(
            model,
            context,
            &crate::OpenAiResponsesOptions {
                stream,
                reasoning_effort: options
                    .thinking
                    .map(|level| model.clamp_thinking_level(level))
                    .and_then(reasoning_effort),
                tool_choice: Some(match options.tool_choice {
                    crate::ToolChoice::Auto => ToolChoice::Auto,
                    crate::ToolChoice::None => ToolChoice::None,
                }),
                ..Default::default()
            },
        )
    }
}

pub fn provider() -> Arc<dyn crate::Provider> {
    Arc::new(Provider::new(crate::openai_models().iter().cloned()))
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

fn mapped_reasoning_effort(model: &crate::Model, effort: ReasoningEffort) -> String {
    model
        .thinking_level_map
        .get(&effort.thinking_level())
        .and_then(Clone::clone)
        .unwrap_or_else(|| effort.as_str().into())
}

fn default_reasoning_effort(model: &crate::Model) -> Option<String> {
    model
        .thinking_level_map
        .get(&crate::ThinkingLevel::Off)
        .cloned()
        .unwrap_or_else(|| Some("none".into()))
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
    sampling_params: BTreeMap<String, serde_json::Value>,
    reasoning: Option<Reasoning>,
    tool_choice: Option<ToolChoice>,
    service_tier: Option<ServiceTier>,
    connection_timeout: Option<Duration>,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_timeout: Option<Duration>,
    session_id: Option<String>,
    cache_retention: CacheRetention,
    session_affinity_format: crate::SessionAffinityFormat,
    deferred_tools_mode: Option<DeferredToolsMode>,
    supports_developer_role: bool,
    supports_long_cache_retention: bool,
    supports_strict_mode: bool,
    supports_grammar_tools: bool,
    supports_explicit_prompt_cache_mode: bool,
    headers: BTreeMap<String, Option<String>>,
    request_hooks: Option<crate::provider::RequestHooks>,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
    Custom(String),
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Function(name) | Self::Custom(name) => {
                let mut map = serializer.serialize_map(Some(2))?;
                let kind = if matches!(self, Self::Function(_)) {
                    "function"
                } else {
                    "custom"
                };
                map.serialize_entry("type", kind)?;
                map.serialize_entry("name", name)?;
                map.end()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Priority,
}

impl ServiceTier {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Default => "default",
            Self::Flex => "flex",
            Self::Priority => "priority",
        }
    }
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
        }
    }

    fn thinking_level(self) -> crate::ThinkingLevel {
        match self {
            Self::Minimal => crate::ThinkingLevel::Minimal,
            Self::Low => crate::ThinkingLevel::Low,
            Self::Medium => crate::ThinkingLevel::Medium,
            Self::High => crate::ThinkingLevel::High,
            Self::XHigh => crate::ThinkingLevel::XHigh,
            Self::Max => crate::ThinkingLevel::Max,
        }
    }
}

#[derive(Clone, Debug)]
struct Reasoning {
    effort: String,
    summary: Option<ReasoningSummary>,
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
            sampling_params: BTreeMap::new(),
            reasoning: None,
            tool_choice: None,
            service_tier: None,
            connection_timeout: None,
            first_event_timeout: None,
            idle_timeout: None,
            overall_timeout: None,
            session_id: None,
            cache_retention: CacheRetention::Short,
            session_affinity_format: crate::SessionAffinityFormat::OpenAi,
            deferred_tools_mode: None,
            supports_developer_role: true,
            supports_long_cache_retention: true,
            supports_strict_mode: true,
            supports_grammar_tools: false,
            supports_explicit_prompt_cache_mode: true,
            headers: BTreeMap::new(),
            request_hooks: None,
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

    pub fn with_sampling_params(
        mut self,
        sampling_params: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        self.sampling_params = sampling_params;
        self
    }

    pub fn with_reasoning(mut self, effort: ReasoningEffort, summary: ReasoningSummary) -> Self {
        self.reasoning = Some(Reasoning {
            effort: effort.as_str().into(),
            summary: Some(summary),
        });
        self
    }

    fn with_reasoning_value(mut self, effort: String, summary: Option<ReasoningSummary>) -> Self {
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

    pub(crate) fn with_deferred_tools_mode(mut self, mode: Option<DeferredToolsMode>) -> Self {
        self.deferred_tools_mode = mode;
        self
    }

    fn with_compatibility(mut self, compat: &crate::OpenAiResponsesCompatibility) -> Self {
        self.supports_developer_role = compat.supports_developer_role.unwrap_or(true);
        self.session_affinity_format = compat
            .session_affinity_format
            .unwrap_or(crate::SessionAffinityFormat::OpenAi);
        self.supports_long_cache_retention = compat.supports_long_cache_retention.unwrap_or(true);
        self.supports_strict_mode = compat.supports_strict_mode.unwrap_or(false);
        self.supports_grammar_tools = compat.supports_open_ai_grammar_tools.unwrap_or(false);
        self.supports_explicit_prompt_cache_mode =
            compat.supports_explicit_prompt_cache_mode.unwrap_or(false);
        self
    }

    fn with_request_options(
        mut self,
        headers: BTreeMap<String, Option<String>>,
        request_hooks: crate::provider::RequestHooks,
    ) -> Self {
        self.headers = headers;
        self.request_hooks = Some(request_hooks);
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
    reasoning: Option<RequestReasoningOptions<'a>>,
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
#[serde(untagged)]
enum RequestTool {
    Function {
        r#type: &'static str,
        name: String,
        description: String,
        parameters: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<Option<bool>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
    },
    Custom {
        r#type: &'static str,
        name: String,
        description: String,
        format: RequestGrammarFormat,
        #[serde(skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
    },
}

#[derive(Serialize)]
struct RequestGrammarFormat {
    r#type: &'static str,
    syntax: &'static str,
    definition: String,
}

#[derive(Serialize)]
struct RequestReasoningOptions<'a> {
    effort: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    summary: Option<ReasoningSummary>,
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
    #[serde(rename = "response.custom_tool_call_input.delta")]
    CustomToolCallInputDelta { output_index: usize, delta: String },
    #[serde(rename = "response.custom_tool_call_input.done")]
    CustomToolCallInputDone { output_index: usize, input: String },
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
    input: Option<String>,
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
        arguments: ToolArguments,
    },
}

enum ToolArguments {
    Json(String),
    Grammar {
        property: String,
        buffer: GrammarInputBuffer,
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
    incomplete_details: Option<IncompleteDetails>,
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
    reason: Option<String>,
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
    #[serde(default)]
    total_tokens: u64,
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
    let placement = crate::deferred_tools::split(
        context,
        options.deferred_tools_mode.is_some(),
        str::to_owned,
    );
    let tool_options =
        ResponseToolOptions::openai(options.supports_strict_mode, options.supports_grammar_tools);
    let grammar_input_properties =
        grammar_input_properties(context.tools(), options.supports_grammar_tools)
            .map_err(Error::InvalidRequest)?;
    let input = response_input(
        ResponseInputTarget::openai(&model.id),
        context,
        Some(if options.supports_developer_role {
            "developer"
        } else {
            "system"
        }),
        options.deferred_tools_mode.map(|mode| (&placement, mode)),
        &grammar_input_properties,
        tool_options,
    )
    .map_err(Error::InvalidRequest)?;
    let tools = request_tools(&placement.immediate, tool_options).map_err(Error::InvalidRequest)?;
    let request = Request {
        model: &model.id,
        input,
        tools,
        stream: true,
        store: false,
        max_output_tokens: options.max_output_tokens,
        temperature: options.temperature,
        reasoning: options
            .reasoning
            .as_ref()
            .map(|reasoning| RequestReasoningOptions {
                effort: &reasoning.effort,
                summary: reasoning.summary,
            }),
        include: options
            .reasoning
            .as_ref()
            .filter(|reasoning| reasoning.summary.is_some())
            .map(|_| vec!["reasoning.encrypted_content"])
            .unwrap_or_default(),
        tool_choice: options.tool_choice.clone(),
        service_tier: options.service_tier,
        prompt_cache_key: match options.cache_retention {
            CacheRetention::None => None,
            CacheRetention::Short | CacheRetention::Long => {
                options.session_id.as_deref().map(clamp_cache_key)
            }
        },
        prompt_cache_retention: match options.cache_retention {
            CacheRetention::Long if options.supports_long_cache_retention => Some("24h"),
            CacheRetention::None | CacheRetention::Short => None,
            CacheRetention::Long => None,
        },
        prompt_cache_options: (matches!(options.cache_retention, CacheRetention::None)
            && options.supports_explicit_prompt_cache_mode)
            .then_some(PromptCacheOptions { mode: "explicit" }),
    };
    let mut request =
        serde_json::to_value(request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let fields = request
        .as_object_mut()
        .ok_or_else(|| Error::InvalidRequest("request body must be an object".into()))?;
    fields.extend(options.sampling_params.clone());
    let request = match &options.request_hooks {
        Some(hooks) => hooks.payload(request).await?,
        None => request,
    };
    let body =
        serde_json::to_vec(&request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let mut default_headers = BTreeMap::from([
        (
            "authorization".into(),
            format!("Bearer {}", options.api_key),
        ),
        ("content-type".into(), "application/json".into()),
    ]);
    if options.cache_retention != CacheRetention::None
        && let Some(session_id) = &options.session_id
    {
        if options.session_affinity_format == crate::SessionAffinityFormat::OpenAi {
            default_headers.insert("session_id".into(), session_id.clone());
        }
        default_headers.insert("x-client-request-id".into(), session_id.clone());
    }
    let headers =
        http::request_headers(default_headers, &options.headers).map_err(Error::InvalidRequest)?;
    let client = reqwest::Client::new();
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: options.max_retries,
                max_delay: options.max_retry_delay,
                cancellation: &options.cancellation,
                deadline: overall_deadline,
                profile: retry::Profile::Standard,
                request_timeout: None,
            },
            || {
                client
                    .post(&url)
                    .headers(headers.clone())
                    .body(body.clone())
                    .send()
            },
            |response| async {
                match &options.request_hooks {
                    Some(hooks) => hooks.response(response).await,
                    None => Ok(()),
                }
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
        ResponseEventOptions {
            grammar_input_properties,
            requested_service_tier: options
                .service_tier
                .map(|service_tier| service_tier.as_str().into()),
            use_requested_for_default: false,
        },
    ))
}

fn fallback_message_id(message_index: usize, text_index: usize) -> String {
    format!("msg_ds_{message_index}_{text_index}")
}

fn openai_call_id(id: &str) -> String {
    normalize_id(id.split('|').next().unwrap_or(id))
}

pub(crate) struct ResponseInputTarget<'a> {
    model: &'a str,
    api: crate::Api,
    provider: &'a str,
}

impl<'a> ResponseInputTarget<'a> {
    fn openai(model: &'a str) -> Self {
        Self {
            model,
            api: crate::Api::OpenAiResponses,
            provider: "openai",
        }
    }

    pub(crate) fn codex(model: &'a str) -> Self {
        Self {
            model,
            api: crate::Api::OpenAiCodexResponses,
            provider: "openai-codex",
        }
    }
}

pub(crate) fn response_input(
    target: ResponseInputTarget<'_>,
    context: &Context,
    system_role: Option<&str>,
    deferred_tools: Option<(&ToolPlacement, DeferredToolsMode)>,
    grammar_input_properties: &BTreeMap<String, String>,
    tool_options: ResponseToolOptions,
) -> Result<Vec<serde_json::Value>, String> {
    let mut input = Vec::new();
    let mut loaded_tools = std::collections::BTreeSet::new();
    if let Some(role) = system_role
        && let Some(system) = context.system()
    {
        input.push(serde_json::json!({
            "role": role,
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
                let same_api_provider =
                    message.api == target.api && message.provider.as_str() == target.provider;
                let same_model = same_api_provider && message.model == target.model;
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
                            let mut item = if let Some(input_property) =
                                grammar_input_properties.get(&call.name)
                            {
                                serde_json::json!({
                                    "type": "custom_tool_call",
                                    "call_id": openai_call_id(call_id),
                                    "name": call.name,
                                    "input": constrained_sampling::grammar_input(
                                        &call.name,
                                        &call.arguments,
                                        input_property,
                                    )?
                                })
                            } else {
                                serde_json::json!({
                                    "type": "function_call",
                                    "call_id": openai_call_id(call_id),
                                    "name": call.name,
                                    "arguments": serde_json::to_string(&call.arguments)
                                        .expect("tool arguments serialize")
                                })
                            };
                            if let Some(item_id) = ids.next()
                                && same_model
                            {
                                item["id"] = item_id.into();
                            }
                            let can_replay_namespace = same_model
                                || deferred_tools.is_some_and(|(placement, _)| {
                                    placement.deferred_tool(&call.name).is_some()
                                });
                            if can_replay_namespace && let Some(namespace) = &call.namespace {
                                item["namespace"] = namespace.clone().into();
                            }
                            input.push(item);
                        }
                        _ => {}
                    }
                }
            }
            Message::ToolResult(result) => {
                let output_type = if grammar_input_properties.contains_key(&result.tool_name) {
                    "custom_tool_call_output"
                } else {
                    "function_call_output"
                };
                input.push(serde_json::json!({
                    "type": output_type,
                    "call_id": openai_call_id(&result.tool_call_id),
                    "output": tool_result_output(result)
                }));
                let Some((placement, mode)) = deferred_tools else {
                    continue;
                };
                let tools = result
                    .added_tool_names
                    .iter()
                    .flatten()
                    .filter_map(|name| {
                        let tool = placement.deferred_tool(name)?;
                        loaded_tools.insert(name.clone()).then_some(tool.clone())
                    })
                    .collect::<Vec<_>>();
                if tools.is_empty() {
                    continue;
                }
                match mode {
                    DeferredToolsMode::AdditionalTools => input.push(serde_json::json!({
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": request_tools(&tools, tool_options)?
                    })),
                    DeferredToolsMode::ToolSearch => {
                        let names = tools
                            .iter()
                            .map(|tool| tool.name.as_str())
                            .collect::<Vec<_>>();
                        let call_id = format!(
                            "pi_tool_load_{}",
                            short_hash(&format!("{}:{}", result.tool_call_id, names.join(",")))
                        );
                        input.push(serde_json::json!({
                            "type": "tool_search_call",
                            "call_id": call_id,
                            "execution": "client",
                            "status": "completed",
                            "arguments": {"query": names.join(" "), "limit": names.len()}
                        }));
                        input.push(serde_json::json!({
                            "type": "tool_search_output",
                            "call_id": call_id,
                            "execution": "client",
                            "status": "completed",
                            "tools": request_tools(&tools, tool_options.deferred(true))?
                        }));
                    }
                }
            }
        }
    }
    Ok(input)
}

pub(crate) fn response_tools(
    tools: &[crate::Tool],
    options: ResponseToolOptions,
) -> Result<Vec<serde_json::Value>, String> {
    request_tools(tools, options)?
        .into_iter()
        .map(|tool| serde_json::to_value(tool).map_err(|error| error.to_string()))
        .collect()
}

#[derive(Clone, Copy)]
pub(crate) struct ResponseToolOptions {
    supports_strict_mode: bool,
    supports_grammar_tools: bool,
    default_strict: Option<Option<bool>>,
    defer_loading: bool,
}

impl ResponseToolOptions {
    fn openai(supports_strict_mode: bool, supports_grammar_tools: bool) -> Self {
        Self {
            supports_strict_mode,
            supports_grammar_tools,
            default_strict: supports_strict_mode.then_some(Some(false)),
            defer_loading: false,
        }
    }

    pub(crate) fn codex(supports_strict_mode: bool, supports_grammar_tools: bool) -> Self {
        Self {
            supports_strict_mode,
            supports_grammar_tools,
            default_strict: supports_strict_mode.then_some(None),
            defer_loading: false,
        }
    }

    fn deferred(mut self, defer_loading: bool) -> Self {
        self.defer_loading = defer_loading;
        self
    }
}

pub(crate) fn grammar_input_properties(
    tools: &[crate::Tool],
    supported: bool,
) -> Result<BTreeMap<String, String>, String> {
    tools
        .iter()
        .filter_map(
            |tool| match constrained_sampling::grammar(tool, supported) {
                Ok(Some(grammar)) => Some(Ok((tool.name.clone(), grammar.input_property))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            },
        )
        .collect()
}

fn request_tools(
    tools: &[crate::Tool],
    options: ResponseToolOptions,
) -> Result<Vec<RequestTool>, String> {
    tools
        .iter()
        .map(|tool| {
            if let Some(grammar) =
                constrained_sampling::grammar(tool, options.supports_grammar_tools)?
            {
                return Ok(RequestTool::Custom {
                    r#type: "custom",
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    format: RequestGrammarFormat {
                        r#type: "grammar",
                        syntax: grammar.syntax,
                        definition: grammar.definition,
                    },
                    defer_loading: options.defer_loading.then_some(true),
                });
            }
            let sampling = constrained_sampling::json_schema(tool, options.supports_strict_mode)?;
            Ok(RequestTool::Function {
                r#type: "function",
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: sampling.as_ref().map_or_else(
                    || tool.parameters.clone(),
                    |sampling| sampling.parameters.clone(),
                ),
                strict: sampling
                    .map(|sampling| Some(sampling.strict))
                    .or(options.default_strict),
                defer_loading: options.defer_loading.then_some(true),
            })
        })
        .collect()
}

fn short_hash(value: &str) -> String {
    let mut h1 = 0xdead_beefu32;
    let mut h2 = 0x41c6_ce57u32;
    for code_unit in value.encode_utf16() {
        h1 = (h1 ^ u32::from(code_unit)).wrapping_mul(2_654_435_761);
        h2 = (h2 ^ u32::from(code_unit)).wrapping_mul(1_597_334_677);
    }
    h1 = (h1 ^ (h1 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h2 ^ (h2 >> 13)).wrapping_mul(3_266_489_909);
    h2 = (h2 ^ (h2 >> 16)).wrapping_mul(2_246_822_507)
        ^ (h1 ^ (h1 >> 13)).wrapping_mul(3_266_489_909);
    format!("{}{}", base36(h2), base36(h1))
}

fn base36(mut value: u32) -> String {
    if value == 0 {
        return "0".into();
    }
    let mut digits = Vec::new();
    while value > 0 {
        let digit = value % 36;
        digits.push(if digit < 10 {
            char::from(b'0' + digit as u8)
        } else {
            char::from(b'a' + (digit - 10) as u8)
        });
        value /= 36;
    }
    digits.into_iter().rev().collect()
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

pub(crate) struct ResponseEventOptions {
    pub(crate) grammar_input_properties: BTreeMap<String, String>,
    pub(crate) requested_service_tier: Option<String>,
    pub(crate) use_requested_for_default: bool,
}

pub(crate) fn decode_stream(
    response: reqwest::Response,
    response_model: String,
    stream_cancellation: CancellationToken,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
    options: ResponseEventOptions,
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
    decode_events(Box::pin(events), response_model, metadata, options)
}

pub(crate) type ProviderEvents =
    Pin<Box<dyn Stream<Item = Result<String, transport::ReadError>> + Send>>;

pub(crate) fn decode_events(
    mut events: ProviderEvents,
    response_model: String,
    metadata: ResponseMetadata,
    options: ResponseEventOptions,
) -> ResponseStream {
    let output = stream! {
        let mut result = Response::openai(response_model);
        result.metadata = metadata;
        let mut slots = HashMap::new();
        let ResponseEventOptions {
            grammar_input_properties,
            requested_service_tier,
            use_requested_for_default,
        } = options;

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
                                        arguments: ToolArguments::Json(arguments),
                                    },
                                );
                            }
                            "custom_tool_call" => {
                                let (Some(id), Some(name)) = (item.call_id, item.name) else {
                                    continue;
                                };
                                let property = grammar_input_properties
                                    .get(&name)
                                    .cloned()
                                    .unwrap_or_else(|| "input".into());
                                let input = item.input.unwrap_or_default();
                                result.content.push(Content::ToolCall(ToolCall {
                                    id,
                                    name,
                                    arguments: serde_json::json!({&property: input}),
                                }));
                                slots.insert(
                                    output_index,
                                    Slot::ToolCall {
                                        content_index,
                                        arguments: ToolArguments::Grammar {
                                            property,
                                            buffer: GrammarInputBuffer {
                                                input: String::new(),
                                                started: false,
                                                closed: false,
                                            },
                                        },
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
                        let ToolArguments::Json(arguments) = arguments else {
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
                        let ToolArguments::Json(arguments) = arguments else {
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
                    StreamEvent::CustomToolCallInputDelta { output_index, delta } => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get_mut(&output_index) else {
                            continue;
                        };
                        let ToolArguments::Grammar { property, buffer } = arguments else {
                            continue;
                        };
                        let next_input = format!("{}{delta}", buffer.input);
                        let delta = match constrained_sampling::append_grammar_input_delta(
                            buffer,
                            property,
                            &next_input,
                            false,
                        ) {
                            Ok(delta) => delta,
                            Err(message) => {
                                yield Err(Error::Stream { message, partial: result });
                                return;
                            }
                        };
                        if let Content::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = serde_json::json!({property.as_str(): buffer.input});
                        }
                        if let Some(delta) = delta {
                            yield Ok(Event::ToolCallDelta { content_index: *content_index, delta });
                        }
                    }
                    StreamEvent::CustomToolCallInputDone { output_index, input } => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get_mut(&output_index) else {
                            continue;
                        };
                        let ToolArguments::Grammar { property, buffer } = arguments else {
                            continue;
                        };
                        let delta = match constrained_sampling::append_grammar_input_delta(
                            buffer,
                            property,
                            &input,
                            true,
                        ) {
                            Ok(delta) => delta,
                            Err(message) => {
                                yield Err(Error::Stream { message, partial: result });
                                return;
                            }
                        };
                        if let Content::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = serde_json::json!({property.as_str(): input});
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
                        let ToolArguments::Json(arguments) = arguments else {
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
                    StreamEvent::OutputItemDone { output_index, item } if item.r#type == "custom_tool_call" => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get_mut(&output_index) else {
                            continue;
                        };
                        let ToolArguments::Grammar { property, buffer } = arguments else {
                            continue;
                        };
                        let input = item.input.unwrap_or_else(|| buffer.input.clone());
                        let delta = match constrained_sampling::append_grammar_input_delta(
                            buffer,
                            property,
                            &input,
                            true,
                        ) {
                            Ok(delta) => delta,
                            Err(message) => {
                                yield Err(Error::Stream { message, partial: result });
                                return;
                            }
                        };
                        if let Content::ToolCall(call) = &mut result.content[*content_index] {
                            if let Some(id) = item.call_id {
                                call.id = id;
                            }
                            if let Some(name) = item.name {
                                call.name = name;
                            }
                            call.arguments = serde_json::json!({property.as_str(): buffer.input});
                        }
                        if let Some(delta) = delta {
                            yield Ok(Event::ToolCallDelta { content_index: *content_index, delta });
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
                        apply_terminal_response(
                            &mut result,
                            response.terminal,
                            requested_service_tier.as_deref(),
                            use_requested_for_default,
                        );
                        match response.status.as_deref() {
                            Some("incomplete") => {
                                let reason = response
                                    .incomplete_details
                                    .and_then(|details| details.reason);
                                if reason.as_deref() == Some("max_output_tokens") {
                                    result.stop_reason = StopReason::Length;
                                    result.raw_stop_reason =
                                        Some("incomplete.max_output_tokens".into());
                                    yield Ok(Event::Done(Box::new(result)));
                                } else {
                                    result.stop_reason = StopReason::Error;
                                    result.raw_stop_reason = Some(reason.as_ref().map_or_else(
                                        || "incomplete".into(),
                                        |reason| format!("incomplete.{reason}"),
                                    ));
                                    yield Err(Error::Response {
                                        code: None,
                                        message: reason.map_or_else(
                                            || "Response incomplete without a provider reason".into(),
                                            |reason| format!("Response incomplete: {reason}"),
                                        ),
                                        partial: result,
                                    });
                                }
                                return;
                            }
                            Some("failed" | "cancelled") => {
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
                            Some("completed" | "in_progress" | "queued") | None => {}
                            Some(status) => {
                                result.stop_reason = StopReason::Error;
                                result.raw_stop_reason = Some(status.into());
                                yield Err(Error::Response {
                                    code: None,
                                    message: format!("Unhandled stop reason: {status}"),
                                    partial: result,
                                });
                                return;
                            }
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
                        result.raw_stop_reason = response.status;
                        yield Ok(Event::Done(Box::new(result)));
                        return;
                    }
                    StreamEvent::Incomplete { response } => {
                        let reason = response
                            .incomplete_details
                            .and_then(|details| details.reason);
                        apply_terminal_response(
                            &mut result,
                            response.terminal,
                            requested_service_tier.as_deref(),
                            use_requested_for_default,
                        );
                        if reason.as_deref() == Some("max_output_tokens") {
                                result.stop_reason = StopReason::Length;
                                result.raw_stop_reason =
                                    Some("incomplete.max_output_tokens".into());
                                yield Ok(Event::Done(Box::new(result)));
                            return;
                        }
                        result.stop_reason = StopReason::Error;
                        result.raw_stop_reason = Some(reason.as_ref().map_or_else(
                            || "incomplete".into(),
                            |reason| format!("incomplete.{reason}"),
                        ));
                        yield Err(Error::Response {
                            code: None,
                            message: reason.map_or_else(
                                || "Response incomplete without a provider reason".into(),
                                |reason| format!("Response incomplete: {reason}"),
                            ),
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
                                    .and_then(|details| details.reason)
                                    .map(|reason| format!("Response incomplete: {reason}"))
                            })
                            .unwrap_or_else(|| "Unknown provider error".into());
                        if response.id.is_some() {
                            result.id = response.id;
                        }
                        result.service_tier = resolve_service_tier(
                            response.service_tier,
                            requested_service_tier.as_deref(),
                            use_requested_for_default,
                        );
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

fn apply_terminal_response(
    result: &mut Response,
    response: TerminalResponse,
    requested_service_tier: Option<&str>,
    use_requested_for_default: bool,
) {
    backfill_reasoning(result, &response.output);
    if response.id.is_some() {
        result.id = response.id;
    }
    result.service_tier = resolve_service_tier(
        response.service_tier,
        requested_service_tier,
        use_requested_for_default,
    );
    result.end_turn = response.end_turn;
    result.usage = usage(response.usage);
}

fn resolve_service_tier(
    response: Option<String>,
    requested: Option<&str>,
    use_requested_for_default: bool,
) -> Option<String> {
    match response.as_deref() {
        None => requested.map(str::to_owned),
        Some("default") if use_requested_for_default => requested.map(str::to_owned).or(response),
        Some(_) => response,
    }
}

pub(crate) fn apply_service_tier_pricing(
    model: &crate::Model,
    usage: &mut Usage,
    service_tier: Option<&str>,
) {
    let multiplier = match service_tier {
        Some("flex") => 0.5,
        Some("priority") if model.id == "gpt-5.5" => 2.5,
        Some("priority") => 2.0,
        _ => return,
    };
    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
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
        total_tokens: usage.total_tokens,
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
