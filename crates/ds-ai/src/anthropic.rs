use crate::{
    AssistantContent, CacheRetention, Content, Context, Error, Event, InputContent, Message,
    RateLimits, Response, ResponseMetadata, ResponseStream, StopReason, ToolResultMessage, Usage,
    UserContent, constrained_sampling, http, json, retry, schema, transport,
    types::{AnthropicReasoning, normalize_id},
};
use async_stream::stream;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

pub mod auth;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_MAX_RETRY_DELAY: Duration = Duration::from_secs(60);
const OAUTH_TOOL_NAMES: [&str; 17] = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
];

pub struct Provider {
    id: crate::ProviderId,
    models: Vec<crate::Model>,
    headers: BTreeMap<String, Option<String>>,
    auth: crate::ProviderAuth,
}

impl Provider {
    pub fn new(models: impl IntoIterator<Item = crate::Model>) -> Self {
        Self {
            id: crate::ProviderId::new("anthropic"),
            models: models.into_iter().collect(),
            headers: BTreeMap::new(),
            auth: crate::ProviderAuth {
                api_key: Some(Arc::new(AnthropicApiKeyAuth)),
                oauth: Some(Arc::new(auth::OAuth::new())),
            },
        }
    }

    fn request(
        &self,
        model: &crate::Model,
        context: &Context,
        options: &crate::AnthropicOptions,
    ) -> crate::AssistantMessageEventStream {
        let requested_model = model.clone();
        let context = context.clone();
        let options = options.clone();
        crate::legacy::adapt(requested_model.clone(), async move {
            let thinking = resolve_thinking(&requested_model, &options);
            let tool_choice = options.tool_choice;
            let stream_options = options.stream;
            let request_hooks = stream_options.request_hooks(&requested_model);
            let api_key = stream_options.api_key;
            if api_key.as_deref().is_none_or(str::is_empty)
                && !has_auth_header(&stream_options.headers)
            {
                return Err(Error::InvalidRequest(
                    "Anthropic request authentication is required".into(),
                ));
            }
            let provider_model = Model::from_public(&requested_model);
            let max_tokens = stream_options
                .max_tokens
                .unwrap_or(requested_model.max_tokens)
                .min(requested_model.max_tokens);
            let mut provider_options = Options::with_auth(api_key)
                .with_max_tokens(max_tokens)
                .with_cancellation(stream_options.cancellation)
                .with_max_retries(stream_options.max_retries.unwrap_or_default())
                .with_max_retry_delay(stream_options.max_retry_delay)
                .with_cache_retention(stream_options.cache_retention)
                .with_interleaved_thinking(options.interleaved_thinking.unwrap_or(true))
                .with_session_id(stream_options.session_id)
                .with_request_options(stream_options.headers, request_hooks);
            if let Some(temperature) = stream_options.temperature {
                provider_options = provider_options.with_temperature(temperature);
            }
            if let Some(timeout) = stream_options.timeout {
                provider_options = provider_options.with_overall_timeout(timeout);
            }
            if let Some(user_id) = stream_options
                .metadata
                .get("user_id")
                .and_then(serde_json::Value::as_str)
            {
                provider_options = provider_options.with_metadata_user_id(user_id);
            }
            if let Some(thinking) = thinking {
                provider_options = provider_options.with_thinking(thinking);
            }
            if let Some(tool_choice) = tool_choice {
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
        "Anthropic"
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
        if model.api != crate::Api::AnthropicMessages {
            let model = model.clone();
            let api = model.api.clone();
            return crate::legacy::adapt(model, async move {
                Err(Error::InvalidRequest(format!(
                    "Anthropic provider has no API implementation for {api}"
                )))
            });
        }
        let crate::ApiStreamOptions::AnthropicMessages(options) = options else {
            let model = model.clone();
            return crate::legacy::adapt(model, async {
                Err(Error::InvalidRequest(
                    "Anthropic Messages options are required".into(),
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
        let thinking = options
            .thinking
            .map(|level| model.clamp_thinking_level(level));
        self.request(
            model,
            context,
            &crate::AnthropicOptions {
                stream,
                thinking_enabled: Some(!matches!(thinking, None | Some(crate::ThinkingLevel::Off))),
                effort: thinking.and_then(anthropic_effort),
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
    Arc::new(Provider::new(crate::anthropic_models().iter().cloned()))
}

fn anthropic_effort(level: crate::ThinkingLevel) -> Option<Effort> {
    match level {
        crate::ThinkingLevel::Off => None,
        crate::ThinkingLevel::Minimal | crate::ThinkingLevel::Low => Some(Effort::Low),
        crate::ThinkingLevel::Medium => Some(Effort::Medium),
        crate::ThinkingLevel::High => Some(Effort::High),
        crate::ThinkingLevel::XHigh => Some(Effort::XHigh),
        crate::ThinkingLevel::Max => Some(Effort::Max),
    }
}

fn resolve_thinking(model: &crate::Model, options: &crate::AnthropicOptions) -> Option<Thinking> {
    if !model.reasoning {
        return None;
    }
    match options.thinking_enabled {
        None => None,
        Some(false) if model.thinking_level_map.get(&crate::ThinkingLevel::Off) != Some(&None) => {
            Some(Thinking::Disabled)
        }
        Some(false) => None,
        Some(true)
            if matches!(
                &model.compat,
                Some(crate::ModelCompatibility::Anthropic(compat))
                    if compat.force_adaptive_thinking == Some(true)
            ) =>
        {
            Some(Thinking::Adaptive {
                effort: options.effort,
                display: options
                    .thinking_display
                    .unwrap_or(ThinkingDisplay::Summarized),
            })
        }
        Some(true) => Some(Thinking::Enabled {
            budget_tokens: options.thinking_budget_tokens.unwrap_or(1024),
            display: options
                .thinking_display
                .unwrap_or(ThinkingDisplay::Summarized),
        }),
    }
}

fn default_supports_tool_references(model: &crate::Model) -> bool {
    if model.provider.as_str() != "anthropic" || model.id.contains("haiku") {
        return false;
    }
    let mut parts = model.id.split('-');
    if parts.next() != Some("claude") || !matches!(parts.next(), Some("opus" | "sonnet" | "fable"))
    {
        return false;
    }
    let Some(major) = parts.next().and_then(|value| value.parse::<u64>().ok()) else {
        return false;
    };
    let minor = parts
        .next()
        .filter(|value| value.len() < 8)
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_default();
    major > 4 || major == 4 && minor >= 5
}

fn has_auth_header(headers: &BTreeMap<String, Option<String>>) -> bool {
    ["authorization", "x-api-key", "cf-aig-authorization"]
        .into_iter()
        .any(|expected| {
            headers.iter().any(|(name, value)| {
                name.eq_ignore_ascii_case(expected)
                    && value
                        .as_deref()
                        .is_some_and(|value| !value.trim().is_empty())
            })
        })
}

fn is_oauth_token(api_key: Option<&str>) -> bool {
    api_key.is_some_and(|api_key| api_key.contains("sk-ant-oat"))
}

fn oauth_tool_name(name: &str) -> String {
    OAUTH_TOOL_NAMES
        .iter()
        .find(|candidate| candidate.eq_ignore_ascii_case(name))
        .map_or_else(|| name.to_owned(), |name| (*name).to_owned())
}

struct AnthropicApiKeyAuth;

#[async_trait]
impl crate::ApiKeyAuth for AnthropicApiKeyAuth {
    fn name(&self) -> &str {
        "Anthropic API key"
    }

    async fn login(
        &self,
        interaction: &dyn crate::AuthInteraction,
    ) -> Result<crate::Credential, crate::AuthError> {
        if interaction.cancellation().is_cancelled() {
            return Err(crate::AuthError::Cancelled);
        }
        let key = interaction
            .prompt(crate::AuthPrompt::Secret {
                message: "Enter Anthropic API key".into(),
                placeholder: None,
            })
            .await?;
        Ok(crate::Credential::ApiKey {
            key: Some(key),
            env: BTreeMap::new(),
        })
    }

    async fn resolve(
        &self,
        context: &dyn crate::AuthContext,
        credential: Option<&crate::Credential>,
        cancellation: &CancellationToken,
    ) -> Result<Option<crate::AuthResult>, crate::AuthError> {
        if cancellation.is_cancelled() {
            return Err(crate::AuthError::Cancelled);
        }
        if let Some(crate::Credential::ApiKey {
            key: Some(key),
            env,
        }) = credential
        {
            return Ok(Some(crate::AuthResult {
                auth: crate::ModelAuth {
                    api_key: Some(key.clone()),
                    ..Default::default()
                },
                env: env.clone(),
                source: Some("stored credential".into()),
            }));
        }
        if let Some(token) = context.env("ANTHROPIC_AUTH_TOKEN").await {
            return Ok(Some(crate::AuthResult {
                auth: crate::ModelAuth {
                    headers: BTreeMap::from([(
                        "Authorization".into(),
                        Some(format!("Bearer {token}")),
                    )]),
                    ..Default::default()
                },
                source: Some("ANTHROPIC_AUTH_TOKEN".into()),
                ..Default::default()
            }));
        }
        for name in ["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"] {
            if let Some(key) = context.env(name).await {
                return Ok(Some(crate::AuthResult {
                    auth: crate::ModelAuth {
                        api_key: Some(key),
                        ..Default::default()
                    },
                    source: Some(name.into()),
                    ..Default::default()
                }));
            }
        }
        Ok(None)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    id: String,
    base_url: String,
    eager_tool_input_streaming: bool,
    long_cache_retention: bool,
    session_affinity_headers: bool,
    cache_control_on_tools: bool,
    temperature: bool,
    strict_tools: bool,
    empty_thinking_signatures: bool,
    tool_references: bool,
    fallback_models: Vec<String>,
}

impl Model {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            base_url: DEFAULT_BASE_URL.into(),
            eager_tool_input_streaming: true,
            long_cache_retention: true,
            session_affinity_headers: false,
            cache_control_on_tools: true,
            temperature: true,
            strict_tools: false,
            empty_thinking_signatures: false,
            tool_references: false,
            fallback_models: Vec::new(),
        }
    }

    fn from_public(model: &crate::Model) -> Self {
        let mut result = Self::new(&model.id).with_base_url(&model.base_url);
        result.tool_references = default_supports_tool_references(model);
        if let Some(crate::ModelCompatibility::Anthropic(compat)) = &model.compat {
            result.eager_tool_input_streaming =
                compat.supports_eager_tool_input_streaming.unwrap_or(true);
            result.long_cache_retention = compat.supports_long_cache_retention.unwrap_or(true);
            result.session_affinity_headers = compat.send_session_affinity_headers.unwrap_or(false);
            result.cache_control_on_tools = compat.supports_cache_control_on_tools.unwrap_or(true);
            result.temperature = compat.supports_temperature.unwrap_or(true);
            result.strict_tools = compat.supports_strict_tools.unwrap_or(false);
            result.empty_thinking_signatures = compat.allow_empty_signature.unwrap_or(false);
            result.tool_references = compat
                .supports_tool_references
                .unwrap_or(result.tool_references);
            result.fallback_models = compat
                .allowed_fallback_models
                .iter()
                .map(|fallback| fallback.model.clone())
                .collect();
        }
        result
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_eager_tool_input_streaming(mut self, enabled: bool) -> Self {
        self.eager_tool_input_streaming = enabled;
        self
    }

    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    pub fn with_empty_thinking_signatures(mut self) -> Self {
        self.empty_thinking_signatures = true;
        self
    }
}

pub struct Options {
    api_key: Option<String>,
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
    interleaved_thinking: bool,
    session_id: Option<String>,
    headers: BTreeMap<String, Option<String>>,
    request_hooks: Option<crate::provider::RequestHooks>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Thinking {
    Disabled,
    Enabled {
        budget_tokens: u64,
        display: ThinkingDisplay,
    },
    Adaptive {
        effort: Option<Effort>,
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
        Self::with_auth(Some(api_key.into()))
    }

    fn with_auth(api_key: Option<String>) -> Self {
        Self {
            api_key,
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
            interleaved_thinking: true,
            session_id: None,
            headers: BTreeMap::new(),
            request_hooks: None,
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

    pub fn with_interleaved_thinking(mut self, enabled: bool) -> Self {
        self.interleaved_thinking = enabled;
        self
    }

    pub fn with_session_id(mut self, session_id: Option<String>) -> Self {
        self.session_id = session_id;
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<serde_json::Value>,
    messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<RequestTool>,
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    fallbacks: Vec<FallbackRequest<'a>>,
}

#[derive(Serialize)]
struct FallbackRequest<'a> {
    model: &'a str,
}

#[derive(Serialize)]
struct RequestMessage {
    role: &'static str,
    content: Vec<serde_json::Value>,
}

#[derive(Serialize)]
struct RequestTool {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    eager_input_streaming: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    defer_loading: Option<bool>,
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
    model: Option<String>,
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
    stop_details: Option<StopDetails>,
}

#[derive(Deserialize)]
struct StopDetails {
    explanation: Option<String>,
}

#[derive(Default, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_creation: Option<CacheCreationUsage>,
    output_tokens_details: Option<OutputTokenDetails>,
}

#[derive(Deserialize)]
struct CacheCreationUsage {
    ephemeral_1h_input_tokens: Option<u64>,
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
    let oauth = is_oauth_token(options.api_key.as_deref());
    let cache_control = cache_control(options.cache_retention, model.long_cache_retention);
    let (thinking, output_config) = thinking(&options.thinking);
    let mut placement = crate::deferred_tools::split(context, model.tool_references, |name| {
        if oauth {
            oauth_tool_name(name)
        } else {
            name.to_owned()
        }
    });
    if placement.immediate.is_empty() && !placement.deferred.is_empty() {
        placement.immediate = placement.deferred.drain(..).map(|(_, tool)| tool).collect();
    }
    let tools = request_tools(model, &placement, cache_control.as_ref(), oauth)
        .map_err(Error::InvalidRequest)?;
    let legacy_tool_streaming = !model.eager_tool_input_streaming && !tools.is_empty();
    let request = Request {
        model: &model.id,
        system: system(context, cache_control.as_ref(), oauth),
        messages: messages(model, context, cache_control.as_ref(), &placement, oauth),
        tools,
        max_tokens: options.max_tokens,
        stream: true,
        stop_sequences: options.stop_sequences.clone(),
        temperature: if model.temperature
            && !matches!(
                options.thinking.as_ref(),
                Some(Thinking::Enabled { .. } | Thinking::Adaptive { .. })
            ) {
            options.temperature
        } else {
            None
        },
        thinking,
        output_config,
        metadata: options
            .metadata_user_id
            .as_ref()
            .map(|user_id| serde_json::json!({"user_id": user_id})),
        tool_choice: options
            .tool_choice
            .as_ref()
            .map(|choice| tool_choice(choice, oauth)),
        fallbacks: model
            .fallback_models
            .iter()
            .map(|model| FallbackRequest { model })
            .collect(),
    };
    let request =
        serde_json::to_value(request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let request = match &options.request_hooks {
        Some(hooks) => hooks.payload(request).await?,
        None => request,
    };
    let body =
        serde_json::to_vec(&request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let client = reqwest::Client::new();
    let url = format!("{}/v1/messages", model.base_url.trim_end_matches('/'));
    let mut beta_features = if oauth {
        vec!["claude-code-20250219", "oauth-2025-04-20"]
    } else {
        Vec::new()
    };
    if legacy_tool_streaming {
        beta_features.push("fine-grained-tool-streaming-2025-05-14");
    }
    if options.interleaved_thinking && matches!(options.thinking, Some(Thinking::Enabled { .. })) {
        beta_features.push("interleaved-thinking-2025-05-14");
    }
    if !model.fallback_models.is_empty() {
        beta_features.push("server-side-fallback-2026-07-01");
    }
    let beta_features = beta_features.join(",");
    let mut default_headers = BTreeMap::from([
        ("anthropic-version".into(), "2023-06-01".into()),
        ("content-type".into(), "application/json".into()),
        ("accept".into(), "application/json".into()),
        (
            "anthropic-dangerous-direct-browser-access".into(),
            "true".into(),
        ),
    ]);
    if oauth {
        default_headers.insert(
            "authorization".into(),
            format!("Bearer {}", options.api_key.as_deref().unwrap_or_default()),
        );
        default_headers.insert("user-agent".into(), "claude-cli/2.1.75".into());
        default_headers.insert("x-app".into(), "cli".into());
    } else if let Some(api_key) = &options.api_key {
        default_headers.insert("x-api-key".into(), api_key.clone());
    }
    if !beta_features.is_empty() {
        default_headers.insert("anthropic-beta".into(), beta_features);
    }
    if model.session_affinity_headers
        && options.cache_retention != CacheRetention::None
        && let Some(session_id) = &options.session_id
    {
        default_headers.insert("x-session-affinity".into(), session_id.clone());
    }
    let headers =
        http::request_headers(default_headers, &options.headers).map_err(Error::InvalidRequest)?;
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
        let metadata = metadata(response.headers());
        return Err(http::provider_error(
            response,
            metadata,
            &options.cancellation,
            overall_deadline,
        )
        .await);
    }

    let metadata = metadata(response.headers());
    let response_model = model.id.clone();
    let response_tool_names = context
        .tools()
        .iter()
        .map(|tool| (tool.name.to_ascii_lowercase(), tool.name.clone()))
        .collect::<HashMap<_, _>>();
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
        let mut terminal_error = None;

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
                    StreamEvent::MessageStart { message } => {
                        result.id = Some(message.id);
                        if let Some(model) = message.model {
                            result.set_anthropic_model(model);
                        }
                        result.usage.cache_write_1h = Some(
                            message
                                .usage
                                .cache_creation
                                .as_ref()
                                .and_then(|cache| cache.ephemeral_1h_input_tokens)
                                .unwrap_or_default(),
                        );
                        apply_usage(&mut result.usage, message.usage);
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "text" =>
                    {
                        let content_index = result.content.len();
                        let text = content_block.text;
                        result.content.push(Content::Text(text.clone()));
                        slots.insert(index, Slot::Text(content_index));
                        if !text.is_empty() {
                            yield Ok(Event::TextDelta { content_index, delta: text });
                        }
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "thinking" =>
                    {
                        let content_index = result.content.len();
                        let thinking = content_block.thinking;
                        result.content.push(Content::Reasoning(thinking.clone()));
                        slots.insert(index, Slot::Thinking {
                            content_index,
                            signature: content_block.signature,
                        });
                        if !thinking.is_empty() {
                            yield Ok(Event::ReasoningDelta {
                                content_index,
                                delta: thinking,
                            });
                        }
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
                        yield Ok(Event::ReasoningDelta {
                            content_index,
                            delta: "[Reasoning redacted]".into(),
                        });
                    }
                    StreamEvent::ContentBlockStart { index, content_block }
                        if content_block.r#type == "tool_use" =>
                    {
                        let (Some(id), Some(mut name)) = (content_block.id, content_block.name) else {
                            continue;
                        };
                        if oauth {
                            name = response_tool_names
                                .get(&name.to_ascii_lowercase())
                                .cloned()
                                .unwrap_or(name);
                        }
                        let content_index = result.content.len();
                        let arguments = content_block.input.unwrap_or_else(|| serde_json::json!({}));
                        let initial = (!arguments
                            .as_object()
                            .is_some_and(serde_json::Map::is_empty))
                        .then(|| {
                            serde_json::to_string(&arguments).expect("tool arguments serialize")
                        });
                        result.content.push(Content::ToolCall(crate::ToolCall {
                            id,
                            name,
                            arguments,
                        }));
                        slots.insert(index, Slot::ToolCall {
                            content_index,
                            partial_json: String::new(),
                        });
                        if let Some(delta) = initial {
                            yield Ok(Event::ToolCallDelta { content_index, delta });
                        }
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
                            terminal_error = None;
                            result.stop_reason = match reason.as_str() {
                                "end_turn" | "stop_sequence" => StopReason::Stop,
                                "max_tokens" => StopReason::Length,
                                "tool_use" => StopReason::ToolUse,
                                "pause_turn" => StopReason::Stop,
                                "refusal" => {
                                    terminal_error = Some((
                                        reason.clone(),
                                        delta.stop_details
                                            .and_then(|details| details.explanation)
                                            .filter(|message| !message.is_empty())
                                            .unwrap_or_else(|| {
                                                "The model refused to complete the request".into()
                                            }),
                                    ));
                                    StopReason::Error
                                }
                                "sensitive" => {
                                    terminal_error = Some((
                                        reason.clone(),
                                        "Provider stopped with: sensitive".into(),
                                    ));
                                    StopReason::Error
                                }
                                _ => {
                                    terminal_error = Some((
                                        reason.clone(),
                                        format!("Unhandled stop reason: {reason}"),
                                    ));
                                    StopReason::Error
                                }
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
                        } else if let Some((code, message)) = terminal_error {
                            yield Err(Error::Response {
                                code: Some(code),
                                message,
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

fn metadata(headers: &reqwest::header::HeaderMap) -> ResponseMetadata {
    let mut metadata = http::metadata(headers);
    metadata.rate_limits = RateLimits {
        limit_requests: http::header_u64(headers, "anthropic-ratelimit-requests-limit"),
        remaining_requests: http::header_u64(headers, "anthropic-ratelimit-requests-remaining"),
        reset_requests: http::header(headers, "anthropic-ratelimit-requests-reset"),
        limit_tokens: http::header_u64(headers, "anthropic-ratelimit-tokens-limit"),
        remaining_tokens: http::header_u64(headers, "anthropic-ratelimit-tokens-remaining"),
        reset_tokens: http::header(headers, "anthropic-ratelimit-tokens-reset"),
    };
    metadata
}

fn system(
    context: &Context,
    cache_control: Option<&CacheControl>,
    oauth: bool,
) -> Vec<serde_json::Value> {
    let mut system = Vec::new();
    if oauth {
        system.push(serde_json::json!({
            "type": "text",
            "text": "You are Claude Code, Anthropic's official CLI for Claude."
        }));
    }
    if let Some(text) = context.system() {
        system.push(serde_json::json!({"type": "text", "text": text}));
    }
    for block in &mut system {
        add_cache_control(block, cache_control);
    }
    system
}

fn messages(
    model: &Model,
    context: &Context,
    cache_control: Option<&CacheControl>,
    placement: &crate::deferred_tools::ToolPlacement,
    oauth: bool,
) -> Vec<RequestMessage> {
    let mut messages = Vec::new();
    let mut pending_tool_calls = Vec::new();
    let mut tool_results = HashSet::new();
    let mut loaded_tools = BTreeSet::new();
    for message in context.messages() {
        match message {
            Message::User(message) => {
                finish_tool_calls(&mut messages, &mut pending_tool_calls, &mut tool_results);
                let content = match &message.content {
                    UserContent::Text(text) => {
                        vec![serde_json::json!({"type": "text", "text": text})]
                    }
                    UserContent::Blocks(content) => content.iter().map(input_content).collect(),
                };
                push_message(&mut messages, "user", content);
            }
            Message::Assistant(response) => {
                finish_tool_calls(&mut messages, &mut pending_tool_calls, &mut tool_results);
                if matches!(
                    response.stop_reason,
                    StopReason::Error | StopReason::Aborted
                ) {
                    continue;
                }
                let content = assistant_content(model, response, oauth);
                pending_tool_calls.extend(response.content.iter().filter_map(|content| {
                    if let AssistantContent::ToolCall(call) = content {
                        Some(normalize_id(&call.id))
                    } else {
                        None
                    }
                }));
                push_message(&mut messages, "assistant", content);
            }
            Message::ToolResult(result) => {
                let id = normalize_id(&result.tool_call_id);
                tool_results.insert(id.clone());
                push_message(
                    &mut messages,
                    "user",
                    tool_result(result, &id, placement, &mut loaded_tools, oauth),
                );
            }
        }
    }
    finish_tool_calls(&mut messages, &mut pending_tool_calls, &mut tool_results);
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

fn push_message(
    messages: &mut Vec<RequestMessage>,
    role: &'static str,
    content: Vec<serde_json::Value>,
) {
    if content.is_empty() {
        return;
    }
    if let Some(message) = messages.last_mut()
        && message.role == role
    {
        message.content.extend(content);
    } else {
        messages.push(RequestMessage { role, content });
    }
}

fn finish_tool_calls(
    messages: &mut Vec<RequestMessage>,
    pending: &mut Vec<String>,
    results: &mut HashSet<String>,
) {
    for id in pending.drain(..) {
        if !results.contains(&id) {
            push_message(
                messages,
                "user",
                vec![serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": id,
                    "content": [{"type": "text", "text": "No result provided"}],
                    "is_error": true
                })],
            );
        }
    }
    results.clear();
}

fn request_tools(
    model: &Model,
    placement: &crate::deferred_tools::ToolPlacement,
    cache_control: Option<&CacheControl>,
    oauth: bool,
) -> Result<Vec<RequestTool>, String> {
    let immediate_count = placement.immediate.len();
    placement
        .immediate
        .iter()
        .map(|tool| (tool, false))
        .chain(placement.deferred.iter().map(|(_, tool)| (tool, true)))
        .enumerate()
        .map(|(index, (tool, deferred))| {
            let sampling = constrained_sampling::json_schema(tool, model.strict_tools)?;
            let strict = sampling.is_some();
            let input_schema = sampling.map_or_else(
                || schema::object(&tool.parameters),
                |sampling| sampling.parameters,
            );
            Ok(RequestTool {
                name: if oauth {
                    oauth_tool_name(&tool.name)
                } else {
                    tool.name.clone()
                },
                description: tool.description.clone(),
                eager_input_streaming: model.eager_tool_input_streaming.then_some(true),
                strict: strict.then_some(true),
                input_schema,
                defer_loading: deferred.then_some(true),
                cache_control: if model.cache_control_on_tools
                    && !deferred
                    && index + 1 == immediate_count
                {
                    cache_control.cloned()
                } else {
                    None
                },
            })
        })
        .collect()
}

fn assistant_content(
    model: &Model,
    message: &crate::AssistantMessage,
    oauth: bool,
) -> Vec<serde_json::Value> {
    let same_model = message.model == model.id;
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                Some(serde_json::json!({"type": "text", "text": text.text}))
            }
            AssistantContent::Text(_) => None,
            AssistantContent::ToolCall(call) => Some(serde_json::json!({
                "type": "tool_use",
                "id": normalize_id(&call.id),
                "name": if oauth { oauth_tool_name(&call.name) } else { call.name.clone() },
                "input": call.arguments
            })),
            AssistantContent::Thinking(thinking) if thinking.redacted == Some(true) => thinking
                .thinking_signature
                .as_ref()
                .filter(|_| same_model)
                .map(|signature| {
                    serde_json::json!({
                        "type": "redacted_thinking",
                        "data": signature
                    })
                }),
            AssistantContent::Thinking(thinking) => {
                let signature = thinking
                    .thinking_signature
                    .as_deref()
                    .filter(|_| same_model);
                if signature.is_some_and(|signature| !signature.trim().is_empty())
                    || signature.is_some() && model.empty_thinking_signatures
                {
                    Some(serde_json::json!({
                        "type": "thinking",
                        "thinking": thinking.thinking,
                        "signature": signature.unwrap_or_default().trim()
                    }))
                } else if thinking.thinking.trim().is_empty() {
                    None
                } else {
                    Some(serde_json::json!({
                        "type": "text",
                        "text": thinking.thinking
                    }))
                }
            }
        })
        .collect()
}

fn parse_arguments(arguments: &str) -> serde_json::Value {
    json::value(arguments)
}

fn input_content(content: &InputContent) -> serde_json::Value {
    match content {
        InputContent::Text(text) => serde_json::json!({"type": "text", "text": text.text}),
        InputContent::Image(image) => serde_json::json!({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image.mime_type,
                "data": image.data
            }
        }),
    }
}

fn cache_control(retention: CacheRetention, supports_long_retention: bool) -> Option<CacheControl> {
    match retention {
        CacheRetention::None => None,
        CacheRetention::Short => Some(CacheControl {
            r#type: "ephemeral",
            ttl: None,
        }),
        CacheRetention::Long => Some(CacheControl {
            r#type: "ephemeral",
            ttl: supports_long_retention.then_some("1h"),
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
            effort.map(|effort| serde_json::json!({"effort": effort})),
        ),
    }
}

fn tool_choice(choice: &ToolChoice, oauth: bool) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Any => serde_json::json!({"type": "any"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Tool(name) => serde_json::json!({
            "type": "tool",
            "name": if oauth { oauth_tool_name(name) } else { name.clone() }
        }),
    }
}

fn tool_result(
    result: &ToolResultMessage,
    id: &str,
    placement: &crate::deferred_tools::ToolPlacement,
    loaded_tools: &mut BTreeSet<String>,
    oauth: bool,
) -> Vec<serde_json::Value> {
    let references = result
        .added_tool_names
        .iter()
        .flatten()
        .filter_map(|name| {
            let name = if oauth {
                oauth_tool_name(name)
            } else {
                name.clone()
            };
            let tool = placement.deferred_tool(&name)?;
            loaded_tools.insert(name.clone()).then(|| {
                serde_json::json!({
                    "type": "tool_reference",
                    "tool_name": if oauth { oauth_tool_name(&tool.name) } else { tool.name.clone() }
                })
            })
        })
        .collect::<Vec<_>>();
    let content = result.content.iter().map(input_content).collect::<Vec<_>>();
    let loaded = !references.is_empty();
    let mut blocks = vec![serde_json::json!({
        "type": "tool_result",
        "tool_use_id": id,
        "content": if loaded { references } else { content.clone() },
        "is_error": result.is_error
    })];
    if loaded {
        blocks.extend(content);
    }
    blocks
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
    if let Some(cache_write_1h) = update
        .cache_creation
        .and_then(|cache| cache.ephemeral_1h_input_tokens)
    {
        usage.cache_write_1h = Some(cache_write_1h);
    }
    if let Some(reasoning) = update
        .output_tokens_details
        .and_then(|details| details.thinking_tokens)
    {
        usage.reasoning = Some(reasoning);
    }
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
}

pub(crate) fn calculate_cost(
    model: &crate::Model,
    response_model: Option<&str>,
    usage: &mut Usage,
) {
    let fallback = match (&model.compat, response_model) {
        (Some(crate::ModelCompatibility::Anthropic(compat)), Some(response_model)) => {
            compat.allowed_fallback_models.iter().find(|fallback| {
                fallback.provider == model.provider && fallback.model == response_model
            })
        }
        _ => None,
    };
    if let Some(fallback) = fallback {
        let mut pricing_model = model.clone();
        pricing_model.cost.clone_from(&fallback.cost);
        pricing_model.calculate_cost(usage);
    } else {
        model.calculate_cost(usage);
    }
}
