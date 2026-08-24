use crate::{AssistantMessage, StopReason};
use regex::RegexSet;
use std::sync::LazyLock;

static OVERFLOW: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)prompt is too long",
        r"(?i)request_too_large",
        r"(?i)input is too long for requested model",
        r"(?i)exceeds the context window",
        r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"(?i)input token count.*exceeds the maximum",
        r"(?i)maximum prompt length is \d+",
        r"(?i)reduce the length of the messages",
        r"(?i)maximum context length is \d+ tokens",
        r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"(?i)exceeds the limit of \d+",
        r"(?i)exceeds the available context size",
        r"(?i)greater than the context length",
        r"(?i)context window exceeds limit",
        r"(?i)exceeded model token limit",
        r"(?i)too large for model with \d+ maximum context length",
        r"(?i)prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"(?i)model_context_window_exceeded",
        r"(?i)prompt too long; exceeded (?:max )?context length",
        r"(?i)range of input length should be",
        r"(?i)context[_ ]length[_ ]exceeded",
        r"(?i)too many tokens",
        r"(?i)token limit exceeded",
        r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
    ])
    .expect("valid overflow patterns")
});

static NOT_OVERFLOW: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new([
        r"(?i)^(Throttling error|Service unavailable):",
        r"(?i)rate limit",
        r"(?i)too many requests",
    ])
    .expect("valid overflow exclusions")
});

pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    if message.stop_reason == StopReason::Error
        && let Some(error) = &message.error_message
        && !NOT_OVERFLOW.is_match(error)
        && OVERFLOW.is_match(error)
    {
        return true;
    }
    let Some(context_window) = context_window.filter(|value| *value > 0) else {
        return false;
    };
    let input = message.usage.input + message.usage.cache_read;
    if message.stop_reason == StopReason::Stop && input > context_window {
        return true;
    }
    message.stop_reason == StopReason::Length
        && message.usage.output == 0
        && input.saturating_mul(100) >= context_window.saturating_mul(99)
}

pub fn is_recoverable_length(message: &AssistantMessage, desired_max_output: u64) -> bool {
    message.stop_reason == StopReason::Length
        && desired_max_output > 0
        && message.usage.output < desired_max_output
}
