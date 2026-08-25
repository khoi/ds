use crate::{
    AssistantContent, AssistantMessage, AssistantMessageDiagnostic, AssistantMessageEvent,
    AssistantMessageEventStream, AssistantToolCall, Error, Model, Response, StopReason,
    TextContent, ThinkingContent, json,
};
use async_stream::stream;
use futures_util::StreamExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
    pin::Pin,
};
use tokio_util::sync::CancellationToken;

pub(crate) enum ProviderEvent {
    ResponseId(String),
    ModelOverride(String),
    TextStart {
        content_index: usize,
        content: TextContent,
        stop_reason: Option<StopReason>,
    },
    TextEnd {
        content_index: usize,
        content: TextContent,
        stop_reason: Option<StopReason>,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingStart {
        content_index: usize,
        content: ThinkingContent,
    },
    ThinkingEnd {
        content_index: usize,
        content: ThinkingContent,
    },
    ReasoningDelta {
        content_index: usize,
        delta: String,
    },
    ToolCallStart {
        content_index: usize,
        tool_call: AssistantToolCall,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: AssistantToolCall,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },
    ToolCallArgumentsPrefix {
        content_index: usize,
        prefix: String,
    },
    Done(Box<Response>),
}

pub(crate) type ProviderEventStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<ProviderEvent, Error>> + Send>>;

pub(crate) struct ProviderStreamSetup {
    source: ProviderEventStream,
    initial_diagnostics: Vec<AssistantMessageDiagnostic>,
}

impl ProviderStreamSetup {
    pub(crate) fn new(source: ProviderEventStream) -> Self {
        Self {
            source,
            initial_diagnostics: Vec::new(),
        }
    }

    pub(crate) fn with_diagnostic(mut self, diagnostic: AssistantMessageDiagnostic) -> Self {
        self.initial_diagnostics.push(diagnostic);
        self
    }
}

impl From<ProviderEventStream> for ProviderStreamSetup {
    fn from(source: ProviderEventStream) -> Self {
        Self::new(source)
    }
}

pub(crate) fn failure(model: Model, error: Error) -> AssistantMessageEventStream {
    adapt(model, CancellationToken::new(), async {
        Err::<ProviderEventStream, _>(error)
    })
}

pub(crate) fn adapt<S>(
    model: Model,
    cancellation: CancellationToken,
    setup: impl Future<Output = Result<S, Error>> + Send + 'static,
) -> AssistantMessageEventStream
where
    S: Into<ProviderStreamSetup> + Send + 'static,
{
    let output = stream! {
        let ProviderStreamSetup {
            mut source,
            initial_diagnostics,
        } = match setup.await {
            Ok(setup) => setup.into(),
            Err(error) => {
                let response = error.partial().cloned();
                yield error_event(&model, error, response, None);
                return;
            }
        };
        let mut partial = empty_message(&model);
        let mut started = BTreeSet::new();
        let mut tool_json = BTreeMap::<usize, String>::new();
        for diagnostic in initial_diagnostics {
            add_diagnostic(&mut partial, diagnostic);
        }
        yield AssistantMessageEvent::Start {
            partial: partial.clone(),
        };
        while let Some(event) = source.next().await {
            match event {
                Ok(ProviderEvent::ResponseId(response_id)) => {
                    partial.response_id = Some(response_id);
                }
                Ok(ProviderEvent::ModelOverride(model)) => {
                    partial.model = model;
                    partial.response_model = None;
                }
                Ok(ProviderEvent::TextStart {
                    content_index,
                    content,
                    stop_reason,
                }) => {
                    if let Some(stop_reason) = stop_reason {
                        partial.stop_reason = stop_reason;
                    }
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::Text(content),
                    ) {
                        yield event;
                    }
                }
                Ok(ProviderEvent::TextEnd {
                    content_index,
                    content,
                    stop_reason,
                }) => {
                    if let Some(stop_reason) = stop_reason {
                        partial.stop_reason = stop_reason;
                    }
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }),
                    ) {
                        yield event;
                    }
                    if let Some(slot) = partial.content.get_mut(content_index) {
                        *slot = AssistantContent::Text(content.clone());
                    }
                    yield AssistantMessageEvent::TextEnd {
                        content_index,
                        content: content.text,
                        partial: partial.clone(),
                    };
                }
                Ok(ProviderEvent::TextDelta { content_index, delta }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }),
                    ) {
                        yield event;
                    }
                    if let Some(AssistantContent::Text(content)) = partial.content.get_mut(content_index) {
                        content.text.push_str(&delta);
                    }
                    yield AssistantMessageEvent::TextDelta {
                        content_index,
                        delta,
                        partial: partial.clone(),
                    };
                }
                Ok(ProviderEvent::ThinkingStart {
                    content_index,
                    content,
                }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::Thinking(content),
                    ) {
                        yield event;
                    }
                }
                Ok(ProviderEvent::ThinkingEnd {
                    content_index,
                    content,
                }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                            redacted: None,
                        }),
                    ) {
                        yield event;
                    }
                    if let Some(slot) = partial.content.get_mut(content_index) {
                        *slot = AssistantContent::Thinking(content.clone());
                    }
                    yield AssistantMessageEvent::ThinkingEnd {
                        content_index,
                        content: content.thinking,
                        partial: partial.clone(),
                    };
                }
                Ok(ProviderEvent::ReasoningDelta { content_index, delta }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                            redacted: None,
                        }),
                    ) {
                        yield event;
                    }
                    if let Some(AssistantContent::Thinking(content)) = partial.content.get_mut(content_index) {
                        content.thinking.push_str(&delta);
                    }
                    yield AssistantMessageEvent::ThinkingDelta {
                        content_index,
                        delta,
                        partial: partial.clone(),
                    };
                }
                Ok(ProviderEvent::ToolCallStart {
                    content_index,
                    tool_call,
                }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::ToolCall(tool_call),
                    ) {
                        yield event;
                    }
                }
                Ok(ProviderEvent::ToolCallEnd {
                    content_index,
                    tool_call,
                }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::ToolCall(tool_call.clone()),
                    ) {
                        yield event;
                    }
                    if let Some(slot) = partial.content.get_mut(content_index) {
                        *slot = AssistantContent::ToolCall(tool_call.clone());
                    }
                    yield AssistantMessageEvent::ToolCallEnd {
                        content_index,
                        tool_call,
                        partial: partial.clone(),
                    };
                }
                Ok(ProviderEvent::ToolCallDelta { content_index, delta }) => {
                    if let Some(event) = start_content(
                        &mut partial,
                        &mut started,
                        content_index,
                        AssistantContent::ToolCall(AssistantToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: serde_json::json!({}),
                            thought_signature: None,
                            namespace: None,
                        }),
                    ) {
                        yield event;
                    }
                    let buffer = tool_json.entry(content_index).or_default();
                    buffer.push_str(&delta);
                    if let Some(AssistantContent::ToolCall(content)) = partial.content.get_mut(content_index) {
                        content.arguments = json::streaming_value(buffer);
                    }
                    yield AssistantMessageEvent::ToolCallDelta {
                        content_index,
                        delta,
                        partial: partial.clone(),
                    };
                }
                Ok(ProviderEvent::ToolCallArgumentsPrefix {
                    content_index,
                    prefix,
                }) => {
                    tool_json.insert(content_index, prefix);
                }
                Ok(ProviderEvent::Done(response)) => {
                    if cancellation.is_cancelled() {
                        let response = *response;
                        yield error_event(
                            &model,
                            Error::Cancelled { partial: None },
                            Some(response),
                            Some(&partial),
                        );
                        return;
                    }
                    let mut message = final_message(&model, *response, partial.timestamp);
                    sync_stream_state(&mut message, &partial);
                    match crate::DoneReason::try_from(message.stop_reason) {
                        Ok(reason) => yield AssistantMessageEvent::Done { reason, message },
                        Err(reason) => {
                            message.stop_reason = StopReason::Error;
                            message.error_message = Some(format!(
                                "provider returned invalid terminal reason: {reason:?}"
                            ));
                            yield AssistantMessageEvent::Error {
                                reason: crate::ErrorReason::Error,
                                error: message,
                            };
                        }
                    }
                    return;
                }
                Err(error) => {
                    let response = error.partial().cloned();
                    yield error_event(&model, error, response, Some(&partial));
                    return;
                }
            }
        }
        let error = if cancellation.is_cancelled() {
            Error::Cancelled { partial: None }
        } else {
            Error::IncompleteStream {
                partial: Response::default(),
            }
        };
        yield error_event(&model, error, None, Some(&partial));
    };
    AssistantMessageEventStream::new(output)
}

#[derive(Clone, Copy)]
enum ContentKind {
    Text,
    Thinking,
    ToolCall,
}

fn start_content(
    partial: &mut AssistantMessage,
    started: &mut BTreeSet<usize>,
    content_index: usize,
    content: AssistantContent,
) -> Option<AssistantMessageEvent> {
    if !started.insert(content_index) {
        return None;
    }
    let kind = match &content {
        AssistantContent::Text(_) => ContentKind::Text,
        AssistantContent::Thinking(_) => ContentKind::Thinking,
        AssistantContent::ToolCall(_) => ContentKind::ToolCall,
    };
    partial.content.push(content);
    Some(start_event(content_index, kind, partial.clone()))
}

fn start_event(
    content_index: usize,
    kind: ContentKind,
    partial: AssistantMessage,
) -> AssistantMessageEvent {
    match kind {
        ContentKind::Text => AssistantMessageEvent::TextStart {
            content_index,
            partial,
        },
        ContentKind::Thinking => AssistantMessageEvent::ThinkingStart {
            content_index,
            partial,
        },
        ContentKind::ToolCall => AssistantMessageEvent::ToolCallStart {
            content_index,
            partial,
        },
    }
}

fn final_message(model: &Model, response: Response, timestamp: u64) -> AssistantMessage {
    let service_tier = response.service_tier.clone();
    let mut message = response.into_assistant_message(model, timestamp);
    match model.api {
        crate::Api::AnthropicMessages => crate::anthropic::calculate_cost(
            model,
            message.response_model.as_deref(),
            &mut message.usage,
        ),
        crate::Api::OpenAiResponses | crate::Api::OpenAiCodexResponses => {
            model.calculate_cost(&mut message.usage);
            crate::openai::apply_service_tier_pricing(
                model,
                &mut message.usage,
                service_tier.as_deref(),
            );
        }
        crate::Api::Other(_) => {
            model.calculate_cost(&mut message.usage);
        }
    }
    message
}

fn error_event(
    model: &Model,
    error: Error,
    response: Option<Response>,
    stream_partial: Option<&AssistantMessage>,
) -> AssistantMessageEvent {
    let reason = if matches!(
        error,
        Error::Cancelled { .. } | Error::Hook { aborted: true, .. }
    ) {
        crate::ErrorReason::Aborted
    } else {
        crate::ErrorReason::Error
    };
    let error_message = match (&error, &model.api) {
        (Error::Cancelled { partial: None }, crate::Api::OpenAiResponses) => {
            "Request aborted".into()
        }
        (Error::Cancelled { partial: Some(_) }, crate::Api::OpenAiResponses) => {
            "OpenAI Responses stream ended before a terminal response event".into()
        }
        (
            Error::Cancelled { .. },
            crate::Api::AnthropicMessages | crate::Api::OpenAiCodexResponses,
        ) => "Request was aborted".into(),
        (
            Error::IncompleteStream { .. },
            crate::Api::OpenAiResponses | crate::Api::OpenAiCodexResponses,
        ) => "OpenAI Responses stream ended before a terminal response event".into(),
        (
            Error::Response { message, .. },
            crate::Api::AnthropicMessages
            | crate::Api::OpenAiResponses
            | crate::Api::OpenAiCodexResponses,
        ) => message.clone(),
        (Error::Provider { status, message }, crate::Api::OpenAiResponses) => {
            format!("OpenAI API error ({status}): {message}")
        }
        (Error::EmptyProviderResponse { status }, crate::Api::OpenAiResponses) => {
            format!("OpenAI API error ({status}): {status} status code (no body)")
        }
        (Error::Stream { message, .. }, crate::Api::AnthropicMessages) => message.clone(),
        (Error::Provider { message, .. }, crate::Api::AnthropicMessages) => message.clone(),
        (Error::Provider { message, .. }, crate::Api::OpenAiCodexResponses) => message.clone(),
        _ => error.to_string(),
    };
    let mut message = response.map_or_else(
        || {
            stream_partial
                .cloned()
                .unwrap_or_else(|| empty_message(model))
        },
        |response| {
            final_message(
                model,
                response,
                stream_partial.map_or_else(timestamp, |partial| partial.timestamp),
            )
        },
    );
    if let Some(stream_partial) = stream_partial {
        message.timestamp = stream_partial.timestamp;
        sync_stream_state(&mut message, stream_partial);
    }
    message.stop_reason = reason.into();
    message.error_message = Some(error_message);
    AssistantMessageEvent::Error {
        reason,
        error: message,
    }
}

fn add_diagnostic(message: &mut AssistantMessage, diagnostic: AssistantMessageDiagnostic) {
    let diagnostics = message.diagnostics.get_or_insert_with(Vec::new);
    if !diagnostics.contains(&diagnostic) {
        diagnostics.push(diagnostic);
    }
}

fn sync_stream_state(message: &mut AssistantMessage, partial: &AssistantMessage) {
    message.model.clone_from(&partial.model);
    if message.response_model.as_deref() == Some(partial.model.as_str()) {
        message.response_model = None;
    }
    let diagnostics = message.diagnostics.get_or_insert_with(Vec::new);
    for diagnostic in partial.diagnostics.iter().flatten() {
        if !diagnostics.contains(diagnostic) {
            diagnostics.push(diagnostic.clone());
        }
    }
    if diagnostics.is_empty() {
        message.diagnostics = None;
    }
}

fn empty_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason: StopReason::Pending,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: timestamp(),
    }
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}
