use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantToolCall, Error, Event, Model, Response, ResponseStream, StopReason, TextContent,
    ThinkingContent, json,
};
use async_stream::stream;
use futures_util::StreamExt;
use std::{
    collections::{BTreeMap, BTreeSet},
    future::Future,
};

pub(crate) fn adapt(
    model: Model,
    setup: impl Future<Output = Result<ResponseStream, Error>> + Send + 'static,
) -> AssistantMessageEventStream {
    let output = stream! {
        let mut source = match setup.await {
            Ok(source) => source,
            Err(error) => {
                yield error_event(&model, error, None);
                return;
            }
        };
        let mut partial = empty_message(&model);
        let mut started = BTreeSet::new();
        let mut tool_json = BTreeMap::<usize, String>::new();
        yield AssistantMessageEvent::Start {
            partial: partial.clone(),
        };
        while let Some(event) = source.next().await {
            match event {
                Ok(Event::TextDelta { content_index, delta }) => {
                    if started.insert(content_index) {
                        partial.content.push(AssistantContent::Text(TextContent {
                            text: String::new(),
                            text_signature: None,
                        }));
                        yield AssistantMessageEvent::TextStart {
                            content_index,
                            partial: partial.clone(),
                        };
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
                Ok(Event::ReasoningDelta { content_index, delta }) => {
                    if started.insert(content_index) {
                        partial.content.push(AssistantContent::Thinking(ThinkingContent {
                            thinking: String::new(),
                            thinking_signature: None,
                            redacted: None,
                        }));
                        yield AssistantMessageEvent::ThinkingStart {
                            content_index,
                            partial: partial.clone(),
                        };
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
                Ok(Event::ToolCallDelta { content_index, delta }) => {
                    if started.insert(content_index) {
                        partial.content.push(AssistantContent::ToolCall(AssistantToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: serde_json::json!({}),
                            thought_signature: None,
                            namespace: None,
                        }));
                        yield AssistantMessageEvent::ToolCallStart {
                            content_index,
                            partial: partial.clone(),
                        };
                    }
                    let buffer = tool_json.entry(content_index).or_default();
                    buffer.push_str(&delta);
                    if let Some(AssistantContent::ToolCall(content)) = partial.content.get_mut(content_index) {
                        content.arguments = json::value(buffer);
                    }
                    yield AssistantMessageEvent::ToolCallDelta {
                        content_index,
                        delta,
                        partial: partial.clone(),
                    };
                }
                Ok(Event::Done(response)) => {
                    let message = final_message(&model, *response);
                    for event in end_events(&mut partial, &message, &mut started) {
                        yield event;
                    }
                    yield AssistantMessageEvent::Done {
                        reason: message.stop_reason,
                        message,
                    };
                    return;
                }
                Err(error) => {
                    let response = partial_response(&error);
                    yield error_event(&model, error, response);
                    return;
                }
            }
        }
        let error = Error::IncompleteStream {
            partial: Response::default(),
        };
        yield error_event(&model, error, None);
    };
    AssistantMessageEventStream::new(output)
}

#[derive(Clone, Copy)]
enum ContentKind {
    Text,
    Thinking,
    ToolCall,
}

fn end_events(
    partial: &mut AssistantMessage,
    message: &AssistantMessage,
    started: &mut BTreeSet<usize>,
) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    for (content_index, content) in message.content.iter().enumerate() {
        if started.insert(content_index) {
            if partial.content.len() == content_index {
                partial.content.push(content.clone());
            }
            let kind = match content {
                AssistantContent::Text(_) => ContentKind::Text,
                AssistantContent::Thinking(_) => ContentKind::Thinking,
                AssistantContent::ToolCall(_) => ContentKind::ToolCall,
            };
            events.push(start_event(content_index, kind, partial.clone()));
        }
        if let Some(slot) = partial.content.get_mut(content_index) {
            *slot = content.clone();
        }
        events.push(end_event(content_index, content, partial.clone()));
    }
    events
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

fn end_event(
    content_index: usize,
    content: &AssistantContent,
    partial: AssistantMessage,
) -> AssistantMessageEvent {
    match content {
        AssistantContent::Text(content) => AssistantMessageEvent::TextEnd {
            content_index,
            content: content.text.clone(),
            partial,
        },
        AssistantContent::Thinking(content) => AssistantMessageEvent::ThinkingEnd {
            content_index,
            content: content.thinking.clone(),
            partial,
        },
        AssistantContent::ToolCall(tool_call) => AssistantMessageEvent::ToolCallEnd {
            content_index,
            tool_call: tool_call.clone(),
            partial,
        },
    }
}

fn final_message(model: &Model, response: Response) -> AssistantMessage {
    let service_tier = response.service_tier.clone();
    let mut message = response.into_assistant_message(timestamp());
    if message.model != model.id {
        message.response_model = Some(message.model.clone());
    }
    message.api = model.api.clone();
    message.provider = model.provider.clone();
    message.model = model.id.clone();
    model.calculate_cost(&mut message.usage);
    if matches!(
        model.api,
        crate::Api::OpenAiResponses | crate::Api::OpenAiCodexResponses
    ) {
        crate::openai::apply_service_tier_pricing(
            model,
            &mut message.usage,
            service_tier.as_deref(),
        );
    }
    message
}

fn error_event(model: &Model, error: Error, response: Option<Response>) -> AssistantMessageEvent {
    let reason = if matches!(error, Error::Cancelled { .. }) {
        StopReason::Aborted
    } else {
        StopReason::Error
    };
    let error_message = error.to_string();
    let mut message = response.map_or_else(
        || empty_message(model),
        |response| final_message(model, response),
    );
    message.stop_reason = reason;
    message.error_message = Some(error_message);
    AssistantMessageEvent::Error {
        reason,
        error: message,
    }
}

fn partial_response(error: &Error) -> Option<Response> {
    match error {
        Error::Stream { partial, .. }
        | Error::IncompleteStream { partial }
        | Error::Response { partial, .. } => Some(partial.clone()),
        Error::Cancelled { partial } | Error::Timeout { partial, .. } => partial.clone(),
        _ => None,
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
