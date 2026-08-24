use crate::{AssistantContent, Context, InputContent, Message, StopReason, Usage, UserContent};
use std::collections::BTreeSet;

const CHARS_PER_TOKEN: usize = 4;
const ESTIMATED_IMAGE_CHARS: usize = 4_800;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextUsageEstimate {
    pub tokens: u64,
    pub usage_tokens: u64,
    pub trailing_tokens: u64,
    pub last_usage_index: Option<usize>,
}

pub fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

pub fn estimate_text_tokens(text: &str) -> u64 {
    tokens(utf16_len(text))
}

pub fn estimate_message_tokens(message: &Message) -> u64 {
    let chars = match message {
        Message::User(message) => user_content_chars(&message.content),
        Message::ToolResult(message) => input_content_chars(&message.content),
        Message::Assistant(message) => message
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => utf16_len(&text.text),
                AssistantContent::Thinking(thinking) => utf16_len(&thinking.thinking),
                AssistantContent::ToolCall(call) => {
                    utf16_len(&call.name) + json_chars(&call.arguments)
                }
            })
            .sum(),
    };
    tokens(chars)
}

pub fn estimate_messages_tokens(messages: &[Message]) -> ContextUsageEstimate {
    estimate_messages(messages)
}

pub fn estimate_context_tokens(context: &Context) -> ContextUsageEstimate {
    let mut estimate = estimate_messages(&context.messages);
    let tools = if let Some(index) = estimate.last_usage_index {
        let added_names = context.messages[index + 1..]
            .iter()
            .filter_map(|message| match message {
                Message::ToolResult(result) => result.added_tool_names.as_ref(),
                _ => None,
            })
            .flatten()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        context
            .tools
            .iter()
            .filter(|tool| added_names.contains(tool.name.as_str()))
            .collect::<Vec<_>>()
    } else {
        context.tools.iter().collect()
    };
    let tool_tokens = if tools.is_empty() {
        0
    } else {
        tokens(json_chars(&tools))
    };
    let prefix_tokens = if estimate.last_usage_index.is_some() {
        tool_tokens
    } else {
        context
            .system_prompt
            .as_deref()
            .map_or(0, estimate_text_tokens)
            + tool_tokens
    };
    estimate.tokens += prefix_tokens;
    estimate.trailing_tokens += prefix_tokens;
    estimate
}

pub fn clamp_max_tokens_to_context(
    model: &crate::Model,
    context: &Context,
    max_tokens: u64,
) -> u64 {
    if model.context_window == 0 {
        return max_tokens.max(1);
    }
    let available = model
        .context_window
        .saturating_sub(estimate_context_tokens(context).tokens)
        .saturating_sub(4_096)
        .max(1);
    max_tokens.min(available)
}

fn estimate_messages(messages: &[Message]) -> ContextUsageEstimate {
    if let Some((index, usage)) = last_usage(messages) {
        let usage_tokens = calculate_context_tokens(usage);
        let trailing_tokens = messages[index + 1..]
            .iter()
            .map(estimate_message_tokens)
            .sum();
        return ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(index),
        };
    }
    let tokens = messages.iter().map(estimate_message_tokens).sum();
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}

fn last_usage(messages: &[Message]) -> Option<(usize, &Usage)> {
    let mut latest_prefix_timestamp = None;
    let mut usage = None;
    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message
            && latest_prefix_timestamp.is_none_or(|latest| assistant.timestamp >= latest)
            && !matches!(
                assistant.stop_reason,
                StopReason::Aborted | StopReason::Error
            )
            && calculate_context_tokens(&assistant.usage) > 0
        {
            usage = Some((index, &assistant.usage));
        }
        latest_prefix_timestamp = Some(
            latest_prefix_timestamp
                .unwrap_or_default()
                .max(message_timestamp(message)),
        );
    }
    usage
}

fn message_timestamp(message: &Message) -> u64 {
    match message {
        Message::User(message) => message.timestamp,
        Message::Assistant(message) => message.timestamp,
        Message::ToolResult(message) => message.timestamp,
    }
}

fn user_content_chars(content: &UserContent) -> usize {
    match content {
        UserContent::Text(text) => utf16_len(text),
        UserContent::Blocks(content) => input_content_chars(content),
    }
}

fn input_content_chars(content: &[InputContent]) -> usize {
    content
        .iter()
        .map(|block| match block {
            InputContent::Text(text) => utf16_len(&text.text),
            InputContent::Image(_) => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

fn utf16_len(value: &str) -> usize {
    value.encode_utf16().count()
}

fn json_chars(value: &impl serde::Serialize) -> usize {
    serde_json::to_string(value).map_or(16, |value| utf16_len(&value))
}

fn tokens(chars: usize) -> u64 {
    chars
        .div_ceil(CHARS_PER_TOKEN)
        .try_into()
        .unwrap_or(u64::MAX)
}
