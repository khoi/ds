use crate::{
    CacheRetention, Content, Context, Error, InputContent, Message, ResponseStream, http, openai,
    retry, schema, transport, types::OpenAiReplay,
};
use base64::prelude::*;
use serde::Serialize;
use std::time::Duration;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api";
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
    access_token: String,
    max_retries: usize,
    max_retry_delay: Option<Duration>,
    cancellation: CancellationToken,
    connection_timeout: Option<Duration>,
    first_event_timeout: Option<Duration>,
    idle_timeout: Option<Duration>,
    overall_timeout: Option<Duration>,
    session_id: Option<String>,
    cache_retention: CacheRetention,
}

impl Options {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            max_retries: 0,
            max_retry_delay: Some(DEFAULT_MAX_RETRY_DELAY),
            cancellation: CancellationToken::new(),
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
    store: bool,
    stream: bool,
    instructions: &'a str,
    input: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    text: TextOptions,
    include: [&'static str; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    prompt_cache_key: Option<String>,
    tool_choice: &'static str,
    parallel_tool_calls: bool,
}

#[derive(Serialize)]
struct TextOptions {
    verbosity: &'static str,
}

pub async fn stream(
    model: &Model,
    context: &Context,
    options: &Options,
) -> Result<ResponseStream, Error> {
    let overall_deadline = options
        .overall_timeout
        .map(|timeout| Instant::now() + timeout);
    let account_id = account_id(&options.access_token).map_err(Error::InvalidRequest)?;
    let session_id = match options.cache_retention {
        CacheRetention::None => None,
        CacheRetention::Short | CacheRetention::Long => {
            options.session_id.as_deref().map(openai::clamp_cache_key)
        }
    };
    let request = Request {
        model: &model.id,
        store: false,
        stream: true,
        instructions: context.system().unwrap_or("You are a helpful assistant."),
        input: input(model, context),
        tools: tools(context).map_err(Error::InvalidRequest)?,
        text: TextOptions { verbosity: "low" },
        include: ["reasoning.encrypted_content"],
        prompt_cache_key: session_id.clone(),
        tool_choice: "auto",
        parallel_tool_calls: true,
    };
    let json =
        serde_json::to_vec(&request).map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let body = zstd::stream::encode_all(json.as_slice(), 3)
        .map_err(|error| Error::Compression(error.to_string()))?;
    let client = reqwest::Client::new();
    let url = response_url(&model.base_url);
    let response = transport::connect(
        retry::send(
            retry::Policy {
                max_retries: options.max_retries,
                max_delay: options.max_retry_delay,
                cancellation: &options.cancellation,
            },
            || {
                let mut request = client
                    .post(&url)
                    .bearer_auth(&options.access_token)
                    .header("chatgpt-account-id", &account_id)
                    .header("openai-beta", "responses=experimental")
                    .header("originator", "ds")
                    .header("user-agent", concat!("ds-ai/", env!("CARGO_PKG_VERSION")))
                    .header("accept", "text/event-stream")
                    .header("content-type", "application/json")
                    .header("content-encoding", "zstd");
                if let Some(session_id) = &session_id {
                    request = request
                        .header("session-id", session_id)
                        .header("x-client-request-id", session_id);
                }
                request.body(body.clone()).send()
            },
        ),
        options.connection_timeout,
        overall_deadline,
    )
    .await?;
    if !response.status().is_success() {
        return Err(http::provider_error(response).await);
    }
    Ok(openai::decode_stream(
        response,
        model.id.clone(),
        options.cancellation.clone(),
        options.first_event_timeout,
        options.idle_timeout,
        overall_deadline,
    ))
}

fn account_id(token: &str) -> Result<String, String> {
    let payload = token
        .split('.')
        .nth(1)
        .ok_or_else(|| "access token is not a JWT".to_string())?;
    let bytes = BASE64_URL_SAFE_NO_PAD
        .decode(payload)
        .or_else(|_| BASE64_URL_SAFE.decode(payload))
        .or_else(|_| BASE64_STANDARD.decode(payload))
        .map_err(|_| "access token has an invalid JWT payload".to_string())?;
    let payload: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|_| "access token has an invalid JWT payload".to_string())?;
    payload
        .get("https://api.openai.com/auth")
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(serde_json::Value::as_str)
        .filter(|account_id| !account_id.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| "access token has no account ID".to_string())
}

fn response_url(base_url: &str) -> String {
    let base_url = base_url.trim().trim_end_matches('/');
    if base_url.ends_with("/codex/responses") {
        base_url.into()
    } else if base_url.ends_with("/codex") {
        format!("{base_url}/responses")
    } else {
        format!("{base_url}/codex/responses")
    }
}

fn input(model: &Model, context: &Context) -> Vec<serde_json::Value> {
    let mut input = Vec::new();
    for message in context.messages() {
        match message {
            Message::User(content) => input.push(serde_json::json!({
                "role": "user",
                "content": content.iter().map(input_content).collect::<Vec<_>>()
            })),
            Message::Assistant(response) => {
                let Some(items) = response.openai_items(&model.id) else {
                    continue;
                };
                for (content_index, content) in response.content.iter().enumerate() {
                    let item = items.iter().find(|item| match item {
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
                    match (content, item) {
                        (
                            Content::Reasoning(text),
                            Some(OpenAiReplay::Reasoning {
                                id,
                                encrypted_content,
                                ..
                            }),
                        ) => input.push(serde_json::json!({
                            "type": "reasoning",
                            "id": id,
                            "summary": [{"type": "summary_text", "text": text}],
                            "encrypted_content": encrypted_content
                        })),
                        (Content::Text(text), Some(OpenAiReplay::Message { id, phase, .. })) => {
                            input.push(serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text, "annotations": []}],
                                "status": "completed",
                                "id": id,
                                "phase": phase
                            }));
                        }
                        (
                            Content::ToolCall(call),
                            Some(OpenAiReplay::ToolCall {
                                item_id, namespace, ..
                            }),
                        ) => input.push(serde_json::json!({
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": serde_json::to_string(&call.arguments).expect("tool arguments serialize"),
                            "namespace": namespace
                        })),
                        _ => {}
                    }
                }
            }
            Message::ToolResult(result) => input.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": result.id,
                "output": openai::tool_result_output(result)
            })),
        }
    }
    input
}

fn input_content(content: &InputContent) -> serde_json::Value {
    match content {
        InputContent::Text(text) => serde_json::json!({"type": "input_text", "text": text}),
        InputContent::Image { media_type, data } => serde_json::json!({
            "type": "input_image",
            "detail": "auto",
            "image_url": format!("data:{media_type};base64,{data}")
        }),
    }
}

fn tools(context: &Context) -> Result<Vec<serde_json::Value>, String> {
    context
        .tools()
        .iter()
        .map(|tool| {
            let parameters = if tool.strict() {
                schema::strict(&tool.parameters).map_err(|error| {
                    format!("tool {:?} has an invalid strict schema: {error}", tool.name)
                })?
            } else {
                tool.parameters.clone()
            };
            Ok(serde_json::json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": tool.strict()
            }))
        })
        .collect()
}
