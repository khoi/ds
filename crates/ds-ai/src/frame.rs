use crate::{AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantToolCall};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum AssistantMessageFrame {
    Start {
        partial: Box<AssistantMessage>,
    },
    TextStart {
        content_index: usize,
        #[serde(
            serialize_with = "serialize_text",
            deserialize_with = "deserialize_text"
        )]
        content: crate::TextContent,
    },
    TextDelta {
        content_index: usize,
        delta: String,
    },
    TextEnd {
        content_index: usize,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        text_signature: Option<String>,
    },
    ThinkingStart {
        content_index: usize,
        #[serde(
            serialize_with = "serialize_thinking",
            deserialize_with = "deserialize_thinking"
        )]
        content: crate::ThinkingContent,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        thinking_signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        redacted: Option<bool>,
    },
    ToolCallStart {
        content_index: usize,
        #[serde(
            serialize_with = "serialize_tool_call",
            deserialize_with = "deserialize_tool_call"
        )]
        tool_call: AssistantToolCall,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
    },
    ToolCallEnd {
        content_index: usize,
        id: String,
        name: String,
        arguments: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        namespace: Option<String>,
    },
}

#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{0}")]
pub struct AssistantMessageFrameError(String);

fn serialize_text<S>(content: &crate::TextContent, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    AssistantContent::Text(content.clone()).serialize(serializer)
}

fn deserialize_text<'de, D>(deserializer: D) -> Result<crate::TextContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match AssistantContent::deserialize(deserializer)? {
        AssistantContent::Text(content) => Ok(content),
        _ => Err(serde::de::Error::custom("expected text content")),
    }
}

fn serialize_thinking<S>(content: &crate::ThinkingContent, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    AssistantContent::Thinking(content.clone()).serialize(serializer)
}

fn deserialize_thinking<'de, D>(deserializer: D) -> Result<crate::ThinkingContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match AssistantContent::deserialize(deserializer)? {
        AssistantContent::Thinking(content) => Ok(content),
        _ => Err(serde::de::Error::custom("expected thinking content")),
    }
}

fn serialize_tool_call<S>(content: &AssistantToolCall, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    AssistantContent::ToolCall(content.clone()).serialize(serializer)
}

fn deserialize_tool_call<'de, D>(deserializer: D) -> Result<AssistantToolCall, D::Error>
where
    D: serde::Deserializer<'de>,
{
    match AssistantContent::deserialize(deserializer)? {
        AssistantContent::ToolCall(content) => Ok(content),
        _ => Err(serde::de::Error::custom("expected tool-call content")),
    }
}

impl AssistantMessageFrameError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

pub fn assistant_message_event_to_frame(
    event: &AssistantMessageEvent,
) -> Result<Option<AssistantMessageFrame>, AssistantMessageFrameError> {
    let frame = match event {
        AssistantMessageEvent::Start { partial } => AssistantMessageFrame::Start {
            partial: Box::new(partial.clone()),
        },
        AssistantMessageEvent::TextStart {
            content_index,
            partial,
        } => AssistantMessageFrame::TextStart {
            content_index: *content_index,
            content: text(partial, *content_index, "text_start")?.clone(),
        },
        AssistantMessageEvent::TextDelta {
            content_index,
            delta,
            ..
        } => AssistantMessageFrame::TextDelta {
            content_index: *content_index,
            delta: delta.clone(),
        },
        AssistantMessageEvent::TextEnd {
            content_index,
            content,
            partial,
        } => AssistantMessageFrame::TextEnd {
            content_index: *content_index,
            content: content.clone(),
            text_signature: text(partial, *content_index, "text_end")?
                .text_signature
                .clone(),
        },
        AssistantMessageEvent::ThinkingStart {
            content_index,
            partial,
        } => AssistantMessageFrame::ThinkingStart {
            content_index: *content_index,
            content: thinking(partial, *content_index, "thinking_start")?.clone(),
        },
        AssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
            ..
        } => AssistantMessageFrame::ThinkingDelta {
            content_index: *content_index,
            delta: delta.clone(),
        },
        AssistantMessageEvent::ThinkingEnd {
            content_index,
            content,
            partial,
        } => {
            let final_content = thinking(partial, *content_index, "thinking_end")?;
            AssistantMessageFrame::ThinkingEnd {
                content_index: *content_index,
                content: content.clone(),
                thinking_signature: final_content.thinking_signature.clone(),
                redacted: final_content.redacted,
            }
        }
        AssistantMessageEvent::ToolCallStart {
            content_index,
            partial,
        } => AssistantMessageFrame::ToolCallStart {
            content_index: *content_index,
            tool_call: tool_call(partial, *content_index, "toolcall_start")?.clone(),
        },
        AssistantMessageEvent::ToolCallDelta {
            content_index,
            delta,
            ..
        } => AssistantMessageFrame::ToolCallDelta {
            content_index: *content_index,
            delta: delta.clone(),
        },
        AssistantMessageEvent::ToolCallEnd {
            content_index,
            tool_call,
            partial,
        } => {
            tool_call_at(partial, *content_index, "toolcall_end")?;
            AssistantMessageFrame::ToolCallEnd {
                content_index: *content_index,
                id: tool_call.id.clone(),
                name: tool_call.name.clone(),
                arguments: tool_call.arguments.clone(),
                thought_signature: tool_call.thought_signature.clone(),
                namespace: tool_call.namespace.clone(),
            }
        }
        AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {
            return Ok(None);
        }
    };
    Ok(Some(frame))
}

fn content<'a>(
    message: &'a AssistantMessage,
    content_index: usize,
    event: &str,
) -> Result<&'a AssistantContent, AssistantMessageFrameError> {
    message.content.get(content_index).ok_or_else(|| {
        AssistantMessageFrameError::new(format!(
            "{event} event has no content block at index {content_index}"
        ))
    })
}

fn text<'a>(
    message: &'a AssistantMessage,
    content_index: usize,
    event: &str,
) -> Result<&'a crate::TextContent, AssistantMessageFrameError> {
    match content(message, content_index, event)? {
        AssistantContent::Text(content) => Ok(content),
        block => Err(wrong_event_block(event, block, content_index)),
    }
}

fn thinking<'a>(
    message: &'a AssistantMessage,
    content_index: usize,
    event: &str,
) -> Result<&'a crate::ThinkingContent, AssistantMessageFrameError> {
    match content(message, content_index, event)? {
        AssistantContent::Thinking(content) => Ok(content),
        block => Err(wrong_event_block(event, block, content_index)),
    }
}

fn tool_call<'a>(
    message: &'a AssistantMessage,
    content_index: usize,
    event: &str,
) -> Result<&'a AssistantToolCall, AssistantMessageFrameError> {
    match content(message, content_index, event)? {
        AssistantContent::ToolCall(content) => Ok(content),
        block => Err(wrong_event_block(event, block, content_index)),
    }
}

fn tool_call_at(
    message: &AssistantMessage,
    content_index: usize,
    event: &str,
) -> Result<(), AssistantMessageFrameError> {
    tool_call(message, content_index, event).map(|_| ())
}

fn wrong_event_block(
    event: &str,
    block: &AssistantContent,
    content_index: usize,
) -> AssistantMessageFrameError {
    AssistantMessageFrameError::new(format!(
        "{event} event points to {} block at index {content_index}",
        block_kind(block)
    ))
}

fn block_kind(block: &AssistantContent) -> &'static str {
    match block {
        AssistantContent::Text(_) => "text",
        AssistantContent::Thinking(_) => "thinking",
        AssistantContent::ToolCall(_) => "toolCall",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Text,
    Thinking,
    ToolCall,
}

struct BlockState {
    kind: Kind,
    ended: bool,
    json: String,
}

pub fn reduce_assistant_message_frames(
    frames: impl IntoIterator<Item = AssistantMessageFrame>,
) -> Result<Option<AssistantMessage>, AssistantMessageFrameError> {
    let frames = frames.into_iter().collect::<Vec<_>>();
    if !frames
        .iter()
        .any(|frame| matches!(frame, AssistantMessageFrame::Start { .. }))
    {
        return Ok(None);
    }

    let mut message = None;
    let mut states = BTreeMap::new();
    for frame in frames {
        if let AssistantMessageFrame::Start { partial } = frame {
            if message.is_some() {
                return Err(AssistantMessageFrameError::new(
                    "Assistant message frame sequence contains more than one start frame",
                ));
            }
            message = Some(*partial);
            continue;
        }
        let output = message.as_mut().ok_or_else(|| {
            AssistantMessageFrameError::new(format!(
                "{} frame appears before the start frame",
                frame_kind(&frame)
            ))
        })?;
        apply_frame(output, &mut states, frame)?;
    }

    let Some(mut message) = message else {
        return Ok(None);
    };
    for (content_index, state) in states {
        if state.kind != Kind::ToolCall || state.ended || state.json.is_empty() {
            continue;
        }
        let Some(AssistantContent::ToolCall(tool_call)) = message.content.get_mut(content_index)
        else {
            return Err(AssistantMessageFrameError::new(
                "Unreachable tool-call frame state",
            ));
        };
        tool_call.arguments = crate::json::streaming_value(&state.json);
    }
    Ok(Some(message))
}

fn apply_frame(
    message: &mut AssistantMessage,
    states: &mut BTreeMap<usize, BlockState>,
    frame: AssistantMessageFrame,
) -> Result<(), AssistantMessageFrameError> {
    match frame {
        AssistantMessageFrame::Start { .. } => Ok(()),
        AssistantMessageFrame::TextStart {
            content_index,
            content,
        } => append(
            message,
            states,
            content_index,
            AssistantContent::Text(content),
            Kind::Text,
        ),
        AssistantMessageFrame::TextDelta {
            content_index,
            delta,
        } => {
            let block = active(message, states, content_index, Kind::Text, "text_delta")?;
            let AssistantContent::Text(content) = block else {
                return Err(AssistantMessageFrameError::new(
                    "Unreachable text frame state",
                ));
            };
            content.text.push_str(&delta);
            Ok(())
        }
        AssistantMessageFrame::TextEnd {
            content_index,
            content,
            text_signature,
        } => {
            let block = active(message, states, content_index, Kind::Text, "text_end")?;
            let AssistantContent::Text(text) = block else {
                return Err(AssistantMessageFrameError::new(
                    "Unreachable text frame state",
                ));
            };
            text.text = content;
            text.text_signature = text_signature;
            state(states, content_index)?.ended = true;
            Ok(())
        }
        AssistantMessageFrame::ThinkingStart {
            content_index,
            content,
        } => append(
            message,
            states,
            content_index,
            AssistantContent::Thinking(content),
            Kind::Thinking,
        ),
        AssistantMessageFrame::ThinkingDelta {
            content_index,
            delta,
        } => {
            let block = active(
                message,
                states,
                content_index,
                Kind::Thinking,
                "thinking_delta",
            )?;
            let AssistantContent::Thinking(content) = block else {
                return Err(AssistantMessageFrameError::new(
                    "Unreachable thinking frame state",
                ));
            };
            content.thinking.push_str(&delta);
            Ok(())
        }
        AssistantMessageFrame::ThinkingEnd {
            content_index,
            content,
            thinking_signature,
            redacted,
        } => {
            let block = active(
                message,
                states,
                content_index,
                Kind::Thinking,
                "thinking_end",
            )?;
            let AssistantContent::Thinking(thinking) = block else {
                return Err(AssistantMessageFrameError::new(
                    "Unreachable thinking frame state",
                ));
            };
            thinking.thinking = content;
            thinking.thinking_signature = thinking_signature;
            thinking.redacted = redacted;
            state(states, content_index)?.ended = true;
            Ok(())
        }
        AssistantMessageFrame::ToolCallStart {
            content_index,
            tool_call,
        } => append(
            message,
            states,
            content_index,
            AssistantContent::ToolCall(tool_call),
            Kind::ToolCall,
        ),
        AssistantMessageFrame::ToolCallDelta {
            content_index,
            delta,
        } => {
            active(
                message,
                states,
                content_index,
                Kind::ToolCall,
                "toolcall_delta",
            )?;
            state(states, content_index)?.json.push_str(&delta);
            Ok(())
        }
        AssistantMessageFrame::ToolCallEnd {
            content_index,
            id,
            name,
            arguments,
            thought_signature,
            namespace,
        } => {
            let block = active(
                message,
                states,
                content_index,
                Kind::ToolCall,
                "toolcall_end",
            )?;
            let AssistantContent::ToolCall(tool_call) = block else {
                return Err(AssistantMessageFrameError::new(
                    "Unreachable tool-call frame state",
                ));
            };
            tool_call.id = id;
            tool_call.name = name;
            tool_call.arguments = arguments;
            tool_call.thought_signature = thought_signature;
            tool_call.namespace = namespace;
            state(states, content_index)?.ended = true;
            Ok(())
        }
    }
}

fn append(
    message: &mut AssistantMessage,
    states: &mut BTreeMap<usize, BlockState>,
    content_index: usize,
    content: AssistantContent,
    kind: Kind,
) -> Result<(), AssistantMessageFrameError> {
    if content_index != message.content.len() {
        let reason = if content_index < message.content.len() {
            "already exists"
        } else {
            "would leave a gap"
        };
        return Err(AssistantMessageFrameError::new(format!(
            "Cannot start assistant message block at index {content_index}: {reason}"
        )));
    }
    message.content.push(content);
    states.insert(
        content_index,
        BlockState {
            kind,
            ended: false,
            json: String::new(),
        },
    );
    Ok(())
}

fn state(
    states: &mut BTreeMap<usize, BlockState>,
    content_index: usize,
) -> Result<&mut BlockState, AssistantMessageFrameError> {
    states.get_mut(&content_index).ok_or_else(|| {
        AssistantMessageFrameError::new(format!(
            "Assistant message block state is missing at index {content_index}"
        ))
    })
}

fn active<'a>(
    message: &'a mut AssistantMessage,
    states: &BTreeMap<usize, BlockState>,
    content_index: usize,
    expected: Kind,
    frame: &str,
) -> Result<&'a mut AssistantContent, AssistantMessageFrameError> {
    let state = states.get(&content_index).ok_or_else(|| {
        AssistantMessageFrameError::new(format!(
            "{frame} frame has no started block at index {content_index}"
        ))
    })?;
    let block = message.content.get_mut(content_index).ok_or_else(|| {
        AssistantMessageFrameError::new(format!(
            "{frame} frame has no started block at index {content_index}"
        ))
    })?;
    if state.kind != expected || kind(block) != expected {
        return Err(AssistantMessageFrameError::new(format!(
            "{frame} frame expected {} block at index {content_index}, found {}",
            kind_name(expected),
            block_kind(block)
        )));
    }
    if state.ended {
        return Err(AssistantMessageFrameError::new(format!(
            "{frame} frame follows the end of block at index {content_index}"
        )));
    }
    Ok(block)
}

fn kind(block: &AssistantContent) -> Kind {
    match block {
        AssistantContent::Text(_) => Kind::Text,
        AssistantContent::Thinking(_) => Kind::Thinking,
        AssistantContent::ToolCall(_) => Kind::ToolCall,
    }
}

fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Text => "text",
        Kind::Thinking => "thinking",
        Kind::ToolCall => "toolCall",
    }
}

fn frame_kind(frame: &AssistantMessageFrame) -> &'static str {
    match frame {
        AssistantMessageFrame::Start { .. } => "start",
        AssistantMessageFrame::TextStart { .. } => "text_start",
        AssistantMessageFrame::TextDelta { .. } => "text_delta",
        AssistantMessageFrame::TextEnd { .. } => "text_end",
        AssistantMessageFrame::ThinkingStart { .. } => "thinking_start",
        AssistantMessageFrame::ThinkingDelta { .. } => "thinking_delta",
        AssistantMessageFrame::ThinkingEnd { .. } => "thinking_end",
        AssistantMessageFrame::ToolCallStart { .. } => "toolcall_start",
        AssistantMessageFrame::ToolCallDelta { .. } => "toolcall_delta",
        AssistantMessageFrame::ToolCallEnd { .. } => "toolcall_end",
    }
}
