use crate::{
    AssistantContent, AssistantToolCall, CacheRetention, Context, Error, InputContent, Message,
    Response, StopReason, TextContent, ThinkingContent, TimeoutPhase, ToolResultMessage, Usage,
    UserContent,
    constrained_sampling::{self, GrammarInputBuffer},
    deferred_tools::{DeferredToolsMode, ToolPlacement},
    http, json, retry, transport,
};
use async_stream::stream;
use futures_core::Stream;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize, Serializer, ser::SerializeMap};
use std::{
    collections::{BTreeMap, HashMap},
    pin::Pin,
    sync::Arc,
    time::Duration,
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
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
}

pub fn stream(
    model: &crate::OpenAiResponsesModel,
    context: &Context,
    options: &crate::OpenAiResponsesOptions,
) -> crate::AssistantMessageEventStream {
    stream_model(model.as_model(), context, options)
}

fn stream_model(
    model: &crate::Model,
    context: &Context,
    options: &crate::OpenAiResponsesOptions,
) -> crate::AssistantMessageEventStream {
    let requested_model = model.clone();
    let context = context.for_model(&requested_model);
    let options = options.clone();
    let cancellation = options.stream.cancellation.clone();
    crate::provider_stream::adapt(requested_model.clone(), cancellation, async move {
        let mut stream_options = options.stream;
        let request_hooks = stream_options.request_hooks(&requested_model);
        let caller_headers = stream_options.headers.clone();
        let api_key = stream_options
            .api_key
            .take()
            .filter(|api_key| !api_key.is_empty());
        if api_key.is_none() && !has_auth_header(&caller_headers) {
            return Err(Error::MissingApiKey(requested_model.provider.clone()));
        }
        let headers = stream_options.request_headers(&requested_model).await?;
        let provider_model =
            Model::new(&requested_model.id).with_base_url(requested_model.base_url.clone());
        let mut provider_options =
            Options::new(api_key, stream_options.http_client.unwrap_or_default())
                .with_cancellation(stream_options.cancellation)
                .with_max_retries(stream_options.max_retries.unwrap_or_default())
                .with_max_retry_delay(stream_options.max_retry_delay)
                .with_cache_retention(stream_options.cache_retention)
                .with_request_options(headers, request_hooks);
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
            .with_compatibility(&compat)
            .with_reasoning_model(requested_model.reasoning);
        if let Some(max_tokens) = stream_options.max_tokens {
            provider_options = provider_options.with_max_output_tokens(max_tokens);
        }
        provider_options = provider_options.with_sampling_params(stream_options.sampling_params);
        if let Some(temperature) = stream_options.temperature {
            provider_options = provider_options.with_temperature(temperature);
        }
        provider_options =
            provider_options.with_timeout(stream_options.timeout.unwrap_or(DEFAULT_TIMEOUT));
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
        response_events(&provider_model, &context, &provider_options).await
    })
}

fn has_auth_header(headers: &BTreeMap<String, Option<String>>) -> bool {
    headers.iter().any(|(name, value)| {
        (name.eq_ignore_ascii_case("authorization")
            || name.eq_ignore_ascii_case("cf-aig-authorization"))
            && value
                .as_deref()
                .is_some_and(|value| !value.trim().is_empty())
    })
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
            return crate::provider_stream::failure(
                model,
                Error::InvalidRequest(format!(
                    "OpenAI provider has no API implementation for {api}"
                )),
            );
        }
        let crate::ApiStreamOptions::OpenAiResponses(options) = options else {
            let model = model.clone();
            return crate::provider_stream::failure(
                model,
                Error::InvalidRequest("OpenAI Responses options are required".into()),
            );
        };
        stream_model(model, context, options)
    }

    fn stream_simple(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::SimpleStreamOptions,
    ) -> crate::AssistantMessageEventStream {
        let stream_options =
            crate::provider::build_simple_stream_options(model, context, options.stream.clone());
        stream_model(
            model,
            context,
            &crate::OpenAiResponsesOptions {
                stream: stream_options,
                reasoning_effort: options
                    .reasoning
                    .map(|level| model.clamp_thinking_level(level))
                    .and_then(reasoning_effort),
                tool_choice: options.tool_choice.map(|tool_choice| match tool_choice {
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
struct Model {
    id: String,
    base_url: String,
}

impl Model {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: DEFAULT_BASE_URL.into(),
        }
    }

    fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

struct Options {
    api_key: Option<String>,
    http_client: reqwest::Client,
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
    max_output_tokens: Option<u64>,
    temperature: Option<f64>,
    sampling_params: BTreeMap<String, serde_json::Value>,
    reasoning: Option<Reasoning>,
    tool_choice: Option<ToolChoice>,
    service_tier: Option<ServiceTier>,
    timeout: Duration,
    session_id: Option<String>,
    cache_retention: CacheRetention,
    session_affinity_format: crate::SessionAffinityFormat,
    deferred_tools_mode: Option<DeferredToolsMode>,
    supports_developer_role: bool,
    supports_long_cache_retention: bool,
    supports_strict_mode: bool,
    supports_grammar_tools: bool,
    supports_explicit_prompt_cache_mode: bool,
    reasoning_model: bool,
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
    AllowedTools {
        mode: AllowedToolsMode,
        tools: Vec<serde_json::Value>,
    },
    Hosted(HostedTool),
    Mcp {
        server_label: String,
        name: Option<Option<String>>,
    },
    ApplyPatch,
    Shell,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AllowedToolsMode {
    Auto,
    Required,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostedTool {
    FileSearch,
    WebSearchPreview,
    Computer,
    ComputerUsePreview,
    ComputerUse,
    WebSearchPreview20250311,
    ImageGeneration,
    CodeInterpreter,
    Mcp,
}

impl HostedTool {
    fn as_str(self) -> &'static str {
        match self {
            Self::FileSearch => "file_search",
            Self::WebSearchPreview => "web_search_preview",
            Self::Computer => "computer",
            Self::ComputerUsePreview => "computer_use_preview",
            Self::ComputerUse => "computer_use",
            Self::WebSearchPreview20250311 => "web_search_preview_2025_03_11",
            Self::ImageGeneration => "image_generation",
            Self::CodeInterpreter => "code_interpreter",
            Self::Mcp => "mcp",
        }
    }
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
            Self::AllowedTools { mode, tools } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "allowed_tools")?;
                map.serialize_entry("mode", mode)?;
                map.serialize_entry("tools", tools)?;
                map.end()
            }
            Self::Hosted(tool) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("type", tool.as_str())?;
                map.end()
            }
            Self::Mcp { server_label, name } => {
                let mut map = serializer.serialize_map(Some(2 + usize::from(name.is_some())))?;
                map.serialize_entry("type", "mcp")?;
                map.serialize_entry("server_label", server_label)?;
                if let Some(name) = name {
                    map.serialize_entry("name", name)?;
                }
                map.end()
            }
            Self::ApplyPatch | Self::Shell => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(
                    "type",
                    if matches!(self, Self::ApplyPatch) {
                        "apply_patch"
                    } else {
                        "shell"
                    },
                )?;
                map.end()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
    Null,
}

impl ServiceTier {
    pub(crate) fn as_str(self) -> Option<&'static str> {
        match self {
            Self::Auto => Some("auto"),
            Self::Default => Some("default"),
            Self::Flex => Some("flex"),
            Self::Scale => Some("scale"),
            Self::Priority => Some("priority"),
            Self::Null => None,
        }
    }
}

impl Serialize for ServiceTier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_str().serialize(serializer)
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
    fn new(api_key: Option<String>, http_client: reqwest::Client) -> Self {
        Self {
            api_key,
            http_client,
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
            max_output_tokens: None,
            temperature: None,
            sampling_params: BTreeMap::new(),
            reasoning: None,
            tool_choice: None,
            service_tier: None,
            timeout: DEFAULT_TIMEOUT,
            session_id: None,
            cache_retention: CacheRetention::Short,
            session_affinity_format: crate::SessionAffinityFormat::OpenAi,
            deferred_tools_mode: None,
            supports_developer_role: true,
            supports_long_cache_retention: true,
            supports_strict_mode: true,
            supports_grammar_tools: false,
            supports_explicit_prompt_cache_mode: true,
            reasoning_model: false,
            headers: BTreeMap::new(),
            request_hooks: None,
        }
    }

    fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    fn with_max_retry_delay(mut self, max_retry_delay: Option<Duration>) -> Self {
        self.max_retry_delay = max_retry_delay;
        self
    }

    fn with_max_output_tokens(mut self, max_output_tokens: u64) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    fn with_sampling_params(
        mut self,
        sampling_params: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        self.sampling_params = sampling_params;
        self
    }

    fn with_reasoning_value(mut self, effort: String, summary: Option<ReasoningSummary>) -> Self {
        self.reasoning = Some(Reasoning { effort, summary });
        self
    }

    fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    fn with_service_tier(mut self, service_tier: ServiceTier) -> Self {
        self.service_tier = Some(service_tier);
        self
    }

    fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    fn with_cache_retention(mut self, retention: CacheRetention) -> Self {
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

    fn with_reasoning_model(mut self, reasoning: bool) -> Self {
        self.reasoning_model = reasoning;
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
        #[serde(default)]
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        #[serde(default)]
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        #[serde(default)]
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        #[serde(default)]
        output_index: usize,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        #[serde(default)]
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        #[serde(default)]
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        #[serde(default)]
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        #[serde(default)]
        output_index: usize,
        arguments: String,
    },
    #[serde(rename = "response.custom_tool_call_input.delta")]
    CustomToolCallInputDelta {
        #[serde(default)]
        output_index: usize,
        item_id: Option<String>,
        delta: String,
    },
    #[serde(rename = "response.custom_tool_call_input.done")]
    CustomToolCallInputDone {
        #[serde(default)]
        output_index: usize,
        item_id: Option<String>,
        input: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        #[serde(default)]
        output_index: usize,
        item: OutputItem,
    },
    #[serde(rename = "response.completed", alias = "response.incomplete")]
    Completed { response: CompletedResponse },
    #[serde(rename = "response.done")]
    Done { response: CompletedResponse },
    #[serde(rename = "response.failed")]
    Failed { response: FailedResponse },
    #[serde(rename = "error")]
    Error {
        code: Option<String>,
        message: Option<String>,
        error: Option<serde_json::Value>,
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
    r#type: String,
    id: Option<String>,
    call_id: Option<String>,
    name: Option<String>,
    arguments: Option<String>,
    input: Option<String>,
    content: Option<Vec<OutputContent>>,
    summary: Option<Vec<SummaryContent>>,
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

fn output_slot(
    item: &OutputItem,
    content_index: usize,
    grammar_input_properties: &BTreeMap<String, String>,
) -> Option<(
    AssistantContent,
    Slot,
    crate::provider_stream::ProviderEvent,
)> {
    match item.r#type.as_str() {
        "message" => Some((
            AssistantContent::Text(TextContent {
                text: String::new(),
                text_signature: None,
            }),
            Slot::Text(content_index),
            crate::provider_stream::ProviderEvent::TextStart {
                content_index,
                content: TextContent {
                    text: String::new(),
                    text_signature: None,
                },
                stop_reason: (item.phase.as_deref() == Some("final_answer"))
                    .then_some(StopReason::Stop),
            },
        )),
        "reasoning" => Some((
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: None,
                redacted: None,
            }),
            Slot::Reasoning(content_index),
            crate::provider_stream::ProviderEvent::ThinkingStart {
                content_index,
                content: ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: None,
                    redacted: None,
                },
            },
        )),
        "function_call" => {
            let call_id = item.call_id.clone()?;
            let name = item.name.clone()?;
            let arguments = item.arguments.clone().unwrap_or_default();
            let call = AssistantToolCall {
                id: call_id,
                name,
                arguments: serde_json::json!({}),
                thought_signature: None,
                namespace: None,
            };
            let tool_call = assistant_tool_call(&call, item.id.as_deref(), item.namespace.clone());
            Some((
                AssistantContent::ToolCall(tool_call.clone()),
                Slot::ToolCall {
                    content_index,
                    arguments: ToolArguments::Json(arguments),
                },
                crate::provider_stream::ProviderEvent::ToolCallStart {
                    content_index,
                    tool_call,
                },
            ))
        }
        "custom_tool_call" => {
            let call_id = item.call_id.clone()?;
            let name = item.name.clone()?;
            let property = grammar_input_properties
                .get(&name)
                .cloned()
                .unwrap_or_else(|| "input".into());
            let call = AssistantToolCall {
                id: call_id,
                name,
                arguments: serde_json::json!({}),
                thought_signature: None,
                namespace: None,
            };
            let tool_call = assistant_tool_call(&call, item.id.as_deref(), item.namespace.clone());
            Some((
                AssistantContent::ToolCall(tool_call.clone()),
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
                crate::provider_stream::ProviderEvent::ToolCallStart {
                    content_index,
                    tool_call,
                },
            ))
        }
        _ => None,
    }
}

fn assistant_tool_call(
    call: &AssistantToolCall,
    item_id: Option<&str>,
    namespace: Option<String>,
) -> AssistantToolCall {
    let id = item_id.filter(|_| !call.id.contains('|')).map_or_else(
        || call.id.clone(),
        |item_id| format!("{}|{item_id}", call.id),
    );
    AssistantToolCall {
        id,
        name: call.name.clone(),
        arguments: call.arguments.clone(),
        thought_signature: call.thought_signature.clone(),
        namespace,
    }
}

fn thinking_content(
    item: &OutputItem,
    raw_item: Option<&serde_json::Value>,
    thinking: String,
) -> ThinkingContent {
    let thinking_signature = item
        .id
        .as_ref()
        .and(raw_item)
        .and_then(|item| serde_json::to_string(item).ok());
    ThinkingContent {
        thinking,
        thinking_signature,
        redacted: None,
    }
}

#[derive(Deserialize)]
struct CompletedResponse {
    #[serde(flatten)]
    terminal: TerminalResponse,
    status: Option<String>,
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(Deserialize)]
struct TerminalResponse {
    id: Option<String>,
    service_tier: Option<String>,
    end_turn: Option<bool>,
    usage: Option<CompletedUsage>,
    #[serde(default)]
    output: Vec<OutputItem>,
}

#[derive(Deserialize)]
struct IncompleteDetails {
    reason: Option<String>,
}

#[derive(Deserialize)]
struct FailedResponse {
    status: Option<String>,
    error: Option<FailedDetail>,
    incomplete_details: Option<IncompleteDetails>,
}

#[derive(Deserialize)]
struct FailedDetail {
    code: Option<String>,
    message: Option<String>,
}

fn failed_error(
    error: Option<&FailedDetail>,
    incomplete_details: Option<&IncompleteDetails>,
    codex: bool,
) -> (Option<String>, String) {
    if codex {
        let code = error
            .and_then(|error| error.code.as_deref())
            .filter(|code| !code.is_empty())
            .map(str::to_owned);
        return (
            code,
            error
                .and_then(|error| error.message.as_deref())
                .filter(|message| !message.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| "Codex response failed".into()),
        );
    }
    if let Some(error) = error {
        let code = error
            .code
            .as_deref()
            .filter(|code| !code.is_empty())
            .map(str::to_owned);
        let message = error
            .message
            .as_deref()
            .filter(|message| !message.is_empty());
        return (
            code.clone(),
            format!(
                "{}: {}",
                code.as_deref().unwrap_or("unknown"),
                message.unwrap_or("no message")
            ),
        );
    }
    if let Some(reason) = incomplete_details
        .and_then(|details| details.reason.as_deref())
        .filter(|reason| !reason.is_empty())
    {
        return (None, format!("incomplete: {reason}"));
    }
    (None, "Unknown error (no error details in response)".into())
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

async fn response_events(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<crate::provider_stream::ProviderEventStream, Error> {
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
        Some(
            if options.reasoning_model && options.supports_developer_role {
                "developer"
            } else {
                "system"
            },
        ),
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
        max_output_tokens: options
            .max_output_tokens
            .filter(|tokens| *tokens > 0)
            .map(|tokens| tokens.max(16)),
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
        ("content-type".into(), "application/json".into()),
        (
            "user-agent".into(),
            concat!("ds-ai/", env!("CARGO_PKG_VERSION")).into(),
        ),
    ]);
    if let Some(api_key) = &options.api_key {
        default_headers.insert("authorization".into(), format!("Bearer {api_key}"));
    }
    if options.cache_retention != CacheRetention::None
        && let Some(session_id) = &options.session_id
        && !session_id.is_empty()
    {
        if options.session_affinity_format == crate::SessionAffinityFormat::OpenAi {
            default_headers.insert("session_id".into(), session_id.clone());
        }
        default_headers.insert("x-client-request-id".into(), session_id.clone());
    }
    let headers =
        http::request_headers(default_headers, &options.headers).map_err(Error::InvalidRequest)?;
    let client = &options.http_client;
    let url = format!("{}/responses", model.base_url.trim_end_matches('/'));
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: options.max_retries,
                max_delay: options.max_retry_delay,
                cancellation: &options.cancellation,
                deadline: None,
                profile: retry::Profile::Standard,
                request_timeout: Some(options.timeout),
            },
            || {
                client
                    .post(&url)
                    .headers(headers.clone())
                    .body(body.clone())
                    .timeout(retry::NO_BODY_TIMEOUT)
                    .send()
            },
            |response| async {
                match &options.request_hooks {
                    Some(hooks) => hooks.response(response).await,
                    None => Ok(()),
                }
            },
        ),
        None,
        None,
    )
    .await?;
    if !response.status().is_success() {
        return Err(http::openai_provider_error(response, &options.cancellation, None).await);
    }
    Ok(decode_stream(
        response,
        model.id.clone(),
        options.cancellation.clone(),
        None,
        None,
        None,
        ResponseEventOptions {
            grammar_input_properties,
            requested_service_tier: options
                .service_tier
                .and_then(|service_tier| service_tier.as_str().map(str::to_owned)),
            use_requested_for_default: false,
            mode: ResponseMode::OpenAi,
        },
    ))
}

fn fallback_message_id(message_index: usize, text_index: usize) -> String {
    if text_index == 0 {
        format!("msg_pi_{message_index}")
    } else {
        format!("msg_pi_{message_index}_{text_index}")
    }
}

fn openai_call_id(id: &str) -> String {
    normalize_response_id(id.split('|').next().unwrap_or(id))
}

fn normalize_response_id(id: &str) -> String {
    id.encode_utf16()
        .map(|code_unit| match u8::try_from(code_unit) {
            Ok(byte) if byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-') => {
                char::from(byte)
            }
            _ => '_',
        })
        .take(64)
        .collect::<String>()
        .trim_end_matches('_')
        .into()
}

fn response_tool_call_id(id: &str, foreign: bool) -> String {
    let Some((call_id, item_id)) = id.split_once('|') else {
        return normalize_response_id(id);
    };
    let call_id = normalize_response_id(call_id);
    let mut item_id = if foreign {
        format!("fc_{}", short_hash(item_id))
    } else {
        normalize_response_id(item_id)
    };
    if !item_id.starts_with("fc_") {
        item_id = normalize_response_id(&format!("fc_{item_id}"));
    }
    format!("{call_id}|{item_id}")
}

fn response_message(
    text: &str,
    message_index: usize,
    text_index: usize,
    signature: Option<(String, Option<String>)>,
) -> serde_json::Value {
    let fallback_id = fallback_message_id(message_index, text_index);
    let (id, phase) = match signature {
        None => (fallback_id, None),
        Some((id, phase)) => {
            let id = if id.is_empty() {
                fallback_id
            } else if id.encode_utf16().count() > 64 {
                format!("msg_{}", short_hash(&id))
            } else {
                id
            };
            (id, phase)
        }
    };
    let mut item = serde_json::json!({
        "type": "message",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": []
        }],
        "status": "completed",
        "id": id
    });
    if let Some(phase) = phase {
        item["phase"] = phase.into();
    }
    item
}

struct PendingResponseToolCall {
    id: String,
    name: String,
}

fn finish_response_tool_calls(
    input: &mut Vec<serde_json::Value>,
    pending: &mut Vec<PendingResponseToolCall>,
    results: &mut std::collections::BTreeSet<String>,
    grammar_input_properties: &BTreeMap<String, String>,
) -> usize {
    let mut inserted = 0;
    for call in pending.drain(..) {
        if results.contains(&call.id) {
            continue;
        }
        let output_type = if grammar_input_properties.contains_key(&call.name) {
            "custom_tool_call_output"
        } else {
            "function_call_output"
        };
        input.push(serde_json::json!({
            "type": output_type,
            "call_id": openai_call_id(&call.id),
            "output": "No result provided"
        }));
        inserted += 1;
    }
    results.clear();
    inserted
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
    let mut tool_call_ids = BTreeMap::new();
    let mut pending_tool_calls = Vec::new();
    let mut tool_results = std::collections::BTreeSet::new();
    let mut message_index = 0;
    if let Some(role) = system_role
        && let Some(system) = context.system().filter(|system| !system.is_empty())
    {
        input.push(serde_json::json!({
            "role": role,
            "content": system
        }));
    }
    for message in context.messages() {
        match message {
            Message::User(message) => {
                message_index += finish_response_tool_calls(
                    &mut input,
                    &mut pending_tool_calls,
                    &mut tool_results,
                    grammar_input_properties,
                );
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
                    message_index += 1;
                }
            }
            Message::Assistant(message) => {
                message_index += finish_response_tool_calls(
                    &mut input,
                    &mut pending_tool_calls,
                    &mut tool_results,
                    grammar_input_properties,
                );
                let same_api_provider =
                    message.api == target.api && message.provider.as_str() == target.provider;
                let same_model = same_api_provider && message.model == target.model;
                let different_model = same_api_provider && !same_model;
                if matches!(message.stop_reason, StopReason::Error | StopReason::Aborted) {
                    for content in &message.content {
                        let AssistantContent::ToolCall(call) = content else {
                            continue;
                        };
                        let normalized_id = if same_model {
                            call.id.clone()
                        } else {
                            response_tool_call_id(&call.id, !same_api_provider)
                        };
                        tool_call_ids.insert(call.id.clone(), normalized_id);
                    }
                    continue;
                }
                let input_start = input.len();
                let mut text_index = 0;
                for content in &message.content {
                    match content {
                        AssistantContent::Thinking(thinking) => {
                            if !same_model {
                                if thinking.redacted != Some(true)
                                    && !thinking.thinking.trim().is_empty()
                                {
                                    input.push(response_message(
                                        &thinking.thinking,
                                        message_index,
                                        text_index,
                                        None,
                                    ));
                                    text_index += 1;
                                }
                            } else if let Some(signature) = &thinking.thinking_signature {
                                let item = serde_json::from_str(signature)
                                    .map_err(|error| error.to_string())?;
                                input.push(item);
                            }
                        }
                        AssistantContent::Text(text) => {
                            let signature = same_model
                                .then_some(text.text_signature.as_deref())
                                .flatten();
                            let signature = signature.and_then(parse_text_signature);
                            input.push(response_message(
                                &text.text,
                                message_index,
                                text_index,
                                signature,
                            ));
                            text_index += 1;
                        }
                        AssistantContent::ToolCall(call) => {
                            let normalized_id = if same_model {
                                call.id.clone()
                            } else {
                                response_tool_call_id(&call.id, !same_api_provider)
                            };
                            tool_call_ids.insert(call.id.clone(), normalized_id.clone());
                            pending_tool_calls.push(PendingResponseToolCall {
                                id: normalized_id.clone(),
                                name: call.name.clone(),
                            });
                            let (call_id, item_id) = normalized_id
                                .split_once('|')
                                .map_or((normalized_id.as_str(), None), |(call_id, item_id)| {
                                    (call_id, Some(item_id))
                                });
                            let input_property = grammar_input_properties.get(&call.name);
                            let mut item = if let Some(input_property) = input_property {
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
                            if let Some(item_id) = item_id
                                && !(different_model && item_id.starts_with("fc_"))
                                && (input_property.is_some() || item_id.starts_with("fc_"))
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
                    }
                }
                if input.len() > input_start {
                    message_index += 1;
                }
            }
            Message::ToolResult(result) => {
                let normalized_id = tool_call_ids
                    .get(&result.tool_call_id)
                    .cloned()
                    .unwrap_or_else(|| normalize_response_id(&result.tool_call_id));
                tool_results.insert(normalized_id.clone());
                let output_type = if grammar_input_properties.contains_key(&result.tool_name) {
                    "custom_tool_call_output"
                } else {
                    "function_call_output"
                };
                input.push(serde_json::json!({
                    "type": output_type,
                    "call_id": openai_call_id(&normalized_id),
                    "output": tool_result_output(result)
                }));
                message_index += 1;
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
    finish_response_tool_calls(
        &mut input,
        &mut pending_tool_calls,
        &mut tool_results,
        grammar_input_properties,
    );
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
        .filter(|phase| matches!(*phase, "commentary" | "final_answer"))
        .map(str::to_owned);
    Some((id, phase))
}

pub(crate) struct ResponseEventOptions {
    pub(crate) grammar_input_properties: BTreeMap<String, String>,
    pub(crate) requested_service_tier: Option<String>,
    pub(crate) use_requested_for_default: bool,
    pub(crate) mode: ResponseMode,
}

#[derive(Clone, Copy)]
pub(crate) enum ResponseMode {
    OpenAi,
    CodexSse,
    CodexWebSocket,
}

pub(crate) fn decode_stream(
    response: reqwest::Response,
    response_model: String,
    stream_cancellation: CancellationToken,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_deadline: Option<Instant>,
    options: ResponseEventOptions,
) -> crate::provider_stream::ProviderEventStream {
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
    decode_events(Box::pin(events), response_model, options)
}

pub(crate) type ProviderEvents =
    Pin<Box<dyn Stream<Item = Result<String, transport::ReadError>> + Send>>;

pub(crate) fn decode_events(
    mut events: ProviderEvents,
    response_model: String,
    options: ResponseEventOptions,
) -> crate::provider_stream::ProviderEventStream {
    let output = stream! {
        let mut result = Response::new(response_model);
        let mut slots = HashMap::new();
        let mut item_slots = HashMap::new();
        let ResponseEventOptions {
            grammar_input_properties,
            requested_service_tier,
            use_requested_for_default,
            mode,
        } = options;
        let codex = !matches!(mode, ResponseMode::OpenAi);
        let websocket = matches!(mode, ResponseMode::CodexWebSocket);

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
            let mut raw_event = match json::parse::<serde_json::Value>(&data) {
                Ok(event) => event,
                Err(error) => {
                    yield Err(if websocket {
                        Error::Protocol {
                            message: error,
                            partial: result,
                        }
                    } else {
                        Error::Stream {
                            message: error,
                            partial: result,
                        }
                    });
                    return;
                }
            };
            let event_type = raw_event.get("type").and_then(serde_json::Value::as_str);
            if !codex && event_type == Some("response.done") {
                continue;
            }
            if codex && event_type.is_none() {
                continue;
            }
            let terminal_codex_event = codex
                && matches!(
                    event_type,
                    Some(
                        "response.completed"
                            | "response.done"
                            | "response.incomplete"
                            | "response.failed",
                    )
                );
            if terminal_codex_event
                && let Some(response) = raw_event
                    .get_mut("response")
                    .and_then(serde_json::Value::as_object_mut)
            {
                if response
                    .get("status")
                    .is_some_and(|status| !status.is_string())
                {
                    response.insert("status".into(), serde_json::Value::Null);
                }
                if response
                    .get("end_turn")
                    .is_some_and(|end_turn| !end_turn.is_boolean())
                {
                    response.insert("end_turn".into(), serde_json::Value::Null);
                }
            }
            let event = match serde_json::from_value::<StreamEvent>(raw_event.clone()) {
                Ok(event) => event,
                Err(error) => {
                    yield Err(if websocket {
                        Error::Protocol {
                            message: error.to_string(),
                            partial: result,
                        }
                    } else {
                        Error::Stream {
                            message: error.to_string(),
                            partial: result,
                        }
                    });
                    return;
                }
            };
            match event {
                    StreamEvent::Created { response } => {
                        result.id = Some(response.id.clone());
                        yield Ok(crate::provider_stream::ProviderEvent::ResponseId(response.id));
                    }
                    StreamEvent::OutputItemAdded { output_index, item } => {
                        if let Some(item_id) = &item.id {
                            item_slots.insert(item_id.clone(), output_index);
                        }
                        let initial_custom_input = (item.r#type == "custom_tool_call")
                            .then(|| item.input.clone())
                            .flatten()
                            .filter(|input| !input.is_empty());
                        let initial_function_arguments = (item.r#type == "function_call")
                            .then(|| item.arguments.clone())
                            .flatten()
                            .filter(|arguments| !arguments.is_empty());
                        let content_index = result.content.len();
                        if let Some((content, slot, start)) =
                            output_slot(&item, content_index, &grammar_input_properties)
                        {
                            if matches!(
                                start,
                                crate::provider_stream::ProviderEvent::TextStart {
                                    stop_reason: Some(_),
                                    ..
                                }
                            ) {
                                result.stop_reason = StopReason::Stop;
                            }
                            result.content.push(content);
                            slots.insert(output_index, slot);
                            yield Ok(start);
                            if let Some(prefix) = initial_function_arguments {
                                yield Ok(crate::provider_stream::ProviderEvent::ToolCallArgumentsPrefix {
                                    content_index,
                                    prefix,
                                });
                            }
                            if let Some(input) = initial_custom_input {
                                let Some(Slot::ToolCall {
                                    content_index,
                                    arguments: ToolArguments::Grammar { property, buffer },
                                }) = slots.get_mut(&output_index) else {
                                    continue;
                                };
                                let delta = match constrained_sampling::append_grammar_input_delta(
                                    buffer,
                                    property,
                                    &input,
                                    false,
                                ) {
                                    Ok(delta) => delta,
                                    Err(message) => {
                                        yield Err(Error::Stream { message, partial: result });
                                        return;
                                    }
                                };
                                if let AssistantContent::ToolCall(call) =
                                    &mut result.content[*content_index]
                                {
                                    call.arguments =
                                        serde_json::json!({property.as_str(): buffer.input});
                                }
                                if let Some(delta) = delta {
                                    yield Ok(crate::provider_stream::ProviderEvent::ToolCallDelta {
                                        content_index: *content_index,
                                        delta,
                                    });
                                }
                            }
                        }
                    }
                    StreamEvent::OutputTextDelta { output_index, delta }
                    | StreamEvent::RefusalDelta { output_index, delta } => {
                        let Some(Slot::Text(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        if let AssistantContent::Text(content) = &mut result.content[*content_index] {
                            content.text.push_str(&delta);
                        }
                        yield Ok(crate::provider_stream::ProviderEvent::TextDelta {
                            content_index: *content_index,
                            delta,
                        });
                    }
                    StreamEvent::ReasoningSummaryTextDelta { output_index, delta }
                    | StreamEvent::ReasoningTextDelta { output_index, delta } => {
                        let Some(Slot::Reasoning(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        if let AssistantContent::Thinking(content) = &mut result.content[*content_index] {
                            content.thinking.push_str(&delta);
                        }
                        yield Ok(crate::provider_stream::ProviderEvent::ReasoningDelta {
                            content_index: *content_index,
                            delta,
                        });
                    }
                    StreamEvent::ReasoningSummaryPartDone { output_index } => {
                        let Some(Slot::Reasoning(content_index)) = slots.get(&output_index) else {
                            continue;
                        };
                        if let AssistantContent::Thinking(content) =
                            &mut result.content[*content_index]
                        {
                            content.thinking.push_str("\n\n");
                        }
                        yield Ok(crate::provider_stream::ProviderEvent::ReasoningDelta {
                            content_index: *content_index,
                            delta: "\n\n".into(),
                        });
                    }
                    StreamEvent::FunctionCallArgumentsDelta { output_index, delta } => {
                        let Some(Slot::ToolCall { content_index, arguments }) = slots.get_mut(&output_index) else {
                            continue;
                        };
                        let ToolArguments::Json(arguments) = arguments else {
                            continue;
                        };
                        arguments.push_str(&delta);
                        if let AssistantContent::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = parse_arguments(arguments);
                        }
                        yield Ok(crate::provider_stream::ProviderEvent::ToolCallDelta {
                            content_index: *content_index,
                            delta,
                        });
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
                        if let AssistantContent::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = parse_arguments(arguments);
                        }
                        if let Some(delta) = delta {
                            yield Ok(crate::provider_stream::ProviderEvent::ToolCallDelta {
                                content_index: *content_index,
                                delta,
                            });
                        }
                    }
                    StreamEvent::CustomToolCallInputDelta {
                        output_index,
                        item_id,
                        delta,
                    } => {
                        let output_index = resolved_output_index(
                            output_index,
                            item_id.as_deref(),
                            &item_slots,
                        );
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
                        if let AssistantContent::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = serde_json::json!({property.as_str(): buffer.input});
                        }
                        if let Some(delta) = delta {
                            yield Ok(crate::provider_stream::ProviderEvent::ToolCallDelta {
                                content_index: *content_index,
                                delta,
                            });
                        }
                    }
                    StreamEvent::CustomToolCallInputDone {
                        output_index,
                        item_id,
                        input,
                    } => {
                        let output_index = resolved_output_index(
                            output_index,
                            item_id.as_deref(),
                            &item_slots,
                        );
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
                        if let AssistantContent::ToolCall(call) = &mut result.content[*content_index] {
                            call.arguments = serde_json::json!({property.as_str(): input});
                        }
                        if let Some(delta) = delta {
                            yield Ok(crate::provider_stream::ProviderEvent::ToolCallDelta {
                                content_index: *content_index,
                                delta,
                            });
                        }
                    }
                    StreamEvent::OutputItemDone { output_index, item } => {
                        let raw_item = raw_event.get("item").cloned();
                        let item_slot_id = item.id.clone();
                        let output_index = resolved_output_index(
                            output_index,
                            item_slot_id.as_deref(),
                            &item_slots,
                        );
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            slots.entry(output_index)
                        {
                            let content_index = result.content.len();
                            if let Some((content, slot, start)) =
                                output_slot(&item, content_index, &grammar_input_properties)
                            {
                                if matches!(
                                    start,
                                    crate::provider_stream::ProviderEvent::TextStart {
                                        stop_reason: Some(_),
                                        ..
                                    }
                                ) {
                                    result.stop_reason = StopReason::Stop;
                                }
                                result.content.push(content);
                                entry.insert(slot);
                                yield Ok(start);
                            }
                        }

                        match item.r#type.as_str() {
                            "message" => {
                                let Some(Slot::Text(content_index)) = slots.get(&output_index) else {
                                    continue;
                                };
                                let content_index = *content_index;
                                let text = item
                                    .content
                                    .as_deref()
                                    .unwrap_or_default()
                                    .iter()
                                    .filter_map(|content| match content.r#type.as_str() {
                                        "output_text" => Some(content.text.as_str()),
                                        "refusal" => Some(content.refusal.as_str()),
                                        _ => None,
                                    })
                                    .collect::<String>();
                                let phase = item.phase;
                                let stop_reason = (phase.as_deref() == Some("final_answer"))
                                    .then_some(StopReason::Stop);
                                let text_signature = item.id.as_ref().and_then(|id| {
                                    let mut signature = serde_json::json!({
                                        "v": 1,
                                        "id": id,
                                    });
                                    if let Some(phase) = phase.as_deref() {
                                        signature["phase"] = phase.into();
                                    }
                                    serde_json::to_string(&signature).ok()
                                });
                                let content = TextContent {
                                    text,
                                    text_signature,
                                };
                                result.content[content_index] = AssistantContent::Text(content.clone());
                                if stop_reason.is_some() {
                                    result.stop_reason = StopReason::Stop;
                                }
                                yield Ok(crate::provider_stream::ProviderEvent::TextEnd {
                                    content_index,
                                    content,
                                    stop_reason,
                                });
                            }
                            "reasoning" => {
                                let Some(Slot::Reasoning(content_index)) = slots.get(&output_index) else {
                                    continue;
                                };
                                let content_index = *content_index;
                                let summary = item.summary.as_deref().unwrap_or_default();
                                let content = item.content.as_deref().unwrap_or_default();
                                let reasoning = if summary.is_empty() {
                                    content
                                        .iter()
                                        .map(|content| content.text.as_str())
                                        .collect::<Vec<_>>()
                                        .join("\n\n")
                                } else {
                                    summary
                                        .iter()
                                        .map(|content| content.text.as_str())
                                        .collect::<Vec<_>>()
                                        .join("\n\n")
                                };
                                let reasoning = if reasoning.is_empty() {
                                    match &result.content[content_index] {
                                        AssistantContent::Thinking(content) => {
                                            content.thinking.clone()
                                        }
                                        _ => String::new(),
                                    }
                                } else {
                                    reasoning
                                };
                                let content =
                                    thinking_content(&item, raw_item.as_ref(), reasoning);
                                result.content[content_index] = AssistantContent::Thinking(content.clone());
                                yield Ok(crate::provider_stream::ProviderEvent::ThinkingEnd {
                                    content_index,
                                    content,
                                });
                            }
                            "function_call" => {
                                let Some(Slot::ToolCall { content_index, arguments: ToolArguments::Json(arguments) }) = slots.get(&output_index) else {
                                    continue;
                                };
                                let content_index = *content_index;
                                let arguments = item
                                    .arguments
                                    .as_deref()
                                    .filter(|arguments| !arguments.is_empty())
                                    .unwrap_or(arguments);
                                let AssistantContent::ToolCall(call) = &mut result.content[content_index] else {
                                    continue;
                                };
                                call.arguments = parse_arguments(arguments);
                                if let Some(namespace) = item.namespace {
                                    call.namespace = Some(namespace);
                                }
                                let tool_call = call.clone();
                                yield Ok(crate::provider_stream::ProviderEvent::ToolCallEnd {
                                    content_index,
                                    tool_call,
                                });
                            }
                            "custom_tool_call" => {
                                let Some(Slot::ToolCall { content_index, arguments: ToolArguments::Grammar { property, buffer } }) = slots.get_mut(&output_index) else {
                                    continue;
                                };
                                let content_index = *content_index;
                                let property = property.clone();
                                let input = item.input.unwrap_or_else(|| buffer.input.clone());
                                let delta = match constrained_sampling::append_grammar_input_delta(
                                    buffer,
                                    &property,
                                    &input,
                                    true,
                                ) {
                                    Ok(delta) => delta,
                                    Err(message) => {
                                        yield Err(Error::Stream { message, partial: result });
                                        return;
                                    }
                                };
                                let AssistantContent::ToolCall(call) = &mut result.content[content_index] else {
                                    continue;
                                };
                                call.arguments = serde_json::json!({property: input});
                                if let Some(delta) = delta {
                                    yield Ok(crate::provider_stream::ProviderEvent::ToolCallDelta {
                                        content_index,
                                        delta,
                                    });
                                }
                                if let Some(namespace) = item.namespace {
                                    call.namespace = Some(namespace);
                                }
                                let tool_call = call.clone();
                                yield Ok(crate::provider_stream::ProviderEvent::ToolCallEnd {
                                    content_index,
                                    tool_call,
                                });
                            }
                            _ => {}
                        }
                        slots.remove(&output_index);
                        if let Some(item_id) = item_slot_id {
                            item_slots.remove(&item_id);
                        }
                    }
                    StreamEvent::Completed { response } | StreamEvent::Done { response } => {
                        apply_terminal_response(
                            &mut result,
                            response.terminal,
                            requested_service_tier.as_deref(),
                            use_requested_for_default,
                            codex,
                        );
                        match response
                            .status
                            .as_deref()
                            .filter(|status| !status.is_empty())
                        {
                            Some("incomplete") => {
                                let reason = response
                                    .incomplete_details
                                    .and_then(|details| details.reason)
                                    .filter(|reason| !reason.is_empty());
                                if reason.as_deref() == Some("max_output_tokens") {
                                    result.stop_reason = StopReason::Length;
                                    result.raw_stop_reason =
                                        Some("incomplete.max_output_tokens".into());
                                    yield Ok(crate::provider_stream::ProviderEvent::Done(Box::new(result)));
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
                                result.stop_reason = StopReason::Error;
                                result.raw_stop_reason = response.status;
                                yield Err(Error::Response {
                                    code: None,
                                    message: "An unknown error occurred".into(),
                                    partial: result,
                                });
                                return;
                            }
                            Some("completed" | "in_progress" | "queued") | None => {}
                            Some(status) => {
                                if codex {
                                    result.stop_reason = if result
                                        .content
                                        .iter()
                                        .any(|content| matches!(content, AssistantContent::ToolCall(_)))
                                    {
                                        StopReason::ToolUse
                                    } else {
                                        StopReason::Stop
                                    };
                                    result.raw_stop_reason = None;
                                    yield Ok(crate::provider_stream::ProviderEvent::Done(Box::new(result)));
                                    return;
                                }
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
                            .any(|content| matches!(content, AssistantContent::ToolCall(_)))
                        {
                            StopReason::ToolUse
                        } else {
                            StopReason::Stop
                        };
                        result.raw_stop_reason = response.status.filter(|status| !status.is_empty());
                        yield Ok(crate::provider_stream::ProviderEvent::Done(Box::new(result)));
                        return;
                    }
                    StreamEvent::Failed { response } => {
                        let (code, message) = failed_error(
                            response.error.as_ref(),
                            response.incomplete_details.as_ref(),
                            codex,
                        );
                        result.stop_reason = StopReason::Error;
                        if !codex {
                            result.raw_stop_reason = response.status.filter(|status| !status.is_empty());
                        }
                        yield Err(Error::Response {
                            code,
                            message,
                            partial: result,
                        });
                        return;
                    }
                    StreamEvent::Error {
                        code,
                        message,
                        error,
                    } => {
                        let code = if codex {
                            code.or_else(|| {
                                error
                                    .as_ref()
                                    .and_then(|error| error.get("code"))
                                    .and_then(serde_json::Value::as_str)
                                    .map(str::to_owned)
                            })
                        } else {
                            code
                        };
                        let message = if codex {
                            message.or_else(|| {
                                error
                                    .as_ref()
                                    .and_then(|error| error.get("message"))
                                    .and_then(serde_json::Value::as_str)
                                    .map(str::to_owned)
                            })
                        } else {
                            message
                        };
                        let message = if codex {
                            let detail = message
                                .as_deref()
                                .filter(|message| !message.is_empty())
                                .or_else(|| code.as_deref().filter(|code| !code.is_empty()))
                                .map(str::to_owned)
                                .unwrap_or_else(|| {
                                    serde_json::to_string(&raw_event)
                                        .unwrap_or_else(|_| "Unknown error".into())
                                });
                            format!("Codex error: {detail}")
                        } else {
                            format!(
                                "Error Code {}: {}",
                                code.as_deref().unwrap_or("undefined"),
                                message.as_deref().unwrap_or("undefined")
                            )
                        };
                        result.stop_reason = StopReason::Error;
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

fn resolved_output_index(
    output_index: usize,
    item_id: Option<&str>,
    item_slots: &HashMap<String, usize>,
) -> usize {
    item_id
        .and_then(|item_id| item_slots.get(item_id).copied())
        .unwrap_or(output_index)
}

fn backfill_reasoning(result: &mut Response, output: &[OutputItem]) {
    for item in output {
        let (Some(id), Some(encrypted)) = (&item.id, &item.encrypted_content) else {
            continue;
        };
        if item.r#type != "reasoning" || encrypted.is_empty() {
            continue;
        }
        for content in &mut result.content {
            let AssistantContent::Thinking(content) = content else {
                continue;
            };
            let Some(signature) = &content.thinking_signature else {
                continue;
            };
            let Ok(mut signature) = serde_json::from_str::<serde_json::Value>(signature) else {
                continue;
            };
            if signature.get("id").and_then(serde_json::Value::as_str) != Some(id) {
                continue;
            }
            if signature
                .get("encrypted_content")
                .is_some_and(http::json_truthy)
            {
                continue;
            }
            signature["encrypted_content"] = encrypted.clone().into();
            content.thinking_signature = serde_json::to_string(&signature).ok();
            break;
        }
    }
}

fn apply_terminal_response(
    result: &mut Response,
    response: TerminalResponse,
    requested_service_tier: Option<&str>,
    use_requested_for_default: bool,
    codex: bool,
) {
    backfill_reasoning(result, &response.output);
    if response.id.as_deref().is_some_and(|id| !id.is_empty()) {
        result.id = response.id;
    }
    result.service_tier = resolve_service_tier(
        response.service_tier,
        requested_service_tier,
        use_requested_for_default,
    );
    if codex {
        result.end_turn = response.end_turn;
    }
    if let Some(terminal_usage) = response.usage {
        result.usage = usage(terminal_usage);
    }
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
