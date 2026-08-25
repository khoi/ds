use ds_ai::{
    AnthropicOptions, Api, AssistantContent, AssistantMessage, AssistantMessageEvent,
    AssistantMessageEventStream, AssistantToolCall, CacheRetention, Context, InputContent, Message,
    Model, ModelCompatibility, ModelInput, OpenAiCodexResponsesOptions, OpenAiResponsesOptions,
    PayloadHook, ProviderId, StopReason, StreamOptions, Tool, ToolResultMessage, Transport, Usage,
    builtin_model, is_context_overflow,
};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum LiveProbe {
    Abort,
    AnthropicFeatures,
    ContextOverflow,
    CrossProviderHandoff,
    EmptyMessages,
    ImageToolResults,
    InterleavedThinking,
    CacheAffinity,
    ReasoningReplay,
    ResponseIds,
    StreamMatrix,
    AbortedUsage,
    ToolCallIds,
    MissingToolResults,
    TotalTokens,
    Unicode,
    XHigh,
}

struct ProbeRows {
    file: &'static str,
    lines: &'static [u32],
    probe: LiveProbe,
}

const LIVE_PROBE_ROWS: &[ProbeRows] = &[
    ProbeRows {
        file: "test/abort.test.ts",
        lines: &[133, 137, 159, 163, 325, 330],
        probe: LiveProbe::Abort,
    },
    ProbeRows {
        file: "test/anthropic-eager-tool-input-e2e.test.ts",
        lines: &[136, 146],
        probe: LiveProbe::AnthropicFeatures,
    },
    ProbeRows {
        file: "test/anthropic-long-cache-retention-e2e.test.ts",
        lines: &[122],
        probe: LiveProbe::AnthropicFeatures,
    },
    ProbeRows {
        file: "test/anthropic-opus-4-8-smoke.test.ts",
        lines: &[25],
        probe: LiveProbe::AnthropicFeatures,
    },
    ProbeRows {
        file: "test/anthropic-tool-name-normalization.test.ts",
        lines: &[28, 70, 111, 164],
        probe: LiveProbe::AnthropicFeatures,
    },
    ProbeRows {
        file: "test/context-overflow.test.ts",
        lines: &[97, 109, 177, 230],
        probe: LiveProbe::ContextOverflow,
    },
    ProbeRows {
        file: "test/cross-provider-handoff.test.ts",
        lines: &[390, 394],
        probe: LiveProbe::CrossProviderHandoff,
    },
    ProbeRows {
        file: "test/empty.test.ts",
        lines: &[
            190, 194, 198, 202, 232, 236, 240, 244, 686, 690, 694, 702, 786, 795, 804, 813,
        ],
        probe: LiveProbe::EmptyMessages,
    },
    ProbeRows {
        file: "test/image-tool-result.test.ts",
        lines: &[242, 246, 268, 272, 476, 484, 532, 541],
        probe: LiveProbe::ImageToolResults,
    },
    ProbeRows {
        file: "test/openai-responses-tool-result-images.test.ts",
        lines: &[150, 183],
        probe: LiveProbe::ImageToolResults,
    },
    ProbeRows {
        file: "test/interleaved-thinking.test.ts",
        lines: &[135, 140],
        probe: LiveProbe::InterleavedThinking,
    },
    ProbeRows {
        file: "test/openai-codex-cache-affinity-e2e.test.ts",
        lines: &[9],
        probe: LiveProbe::CacheAffinity,
    },
    ProbeRows {
        file: "test/openai-responses-cache-affinity-e2e.test.ts",
        lines: &[6],
        probe: LiveProbe::CacheAffinity,
    },
    ProbeRows {
        file: "test/openai-responses-reasoning-replay-e2e.test.ts",
        lines: &[19, 83, 183],
        probe: LiveProbe::ReasoningReplay,
    },
    ProbeRows {
        file: "test/responseid.test.ts",
        lines: &[67, 75, 115],
        probe: LiveProbe::ResponseIds,
    },
    ProbeRows {
        file: "test/stream.test.ts",
        lines: &[
            478, 482, 486, 490, 494, 498, 506, 510, 514, 518, 1315, 1319, 1323, 1327, 1331, 1335,
            1343, 1347, 1351, 1355, 1359, 1363, 1371, 1437, 1441, 1445, 1449, 1453, 1457, 1465,
            1469, 1473, 1477, 1481, 1485, 1494, 1498, 1502, 1506, 1510, 1514,
        ],
        probe: LiveProbe::StreamMatrix,
    },
    ProbeRows {
        file: "test/tokens.test.ts",
        lines: &[111, 129, 318, 348],
        probe: LiveProbe::AbortedUsage,
    },
    ProbeRows {
        file: "test/tool-call-id-normalization.test.ts",
        lines: &[115, 265],
        probe: LiveProbe::ToolCallIds,
    },
    ProbeRows {
        file: "test/tool-call-without-result.test.ts",
        lines: &[122, 140, 320, 350],
        probe: LiveProbe::MissingToolResults,
    },
    ProbeRows {
        file: "test/total-tokens.test.ts",
        lines: &[106, 129, 180, 865],
        probe: LiveProbe::TotalTokens,
    },
    ProbeRows {
        file: "test/unicode-surrogate.test.ts",
        lines: &[321, 325, 355, 359, 375, 379, 812, 821],
        probe: LiveProbe::Unicode,
    },
    ProbeRows {
        file: "test/xhigh.test.ts",
        lines: &[20, 39],
        probe: LiveProbe::XHigh,
    },
];

#[test]
fn live_probe_ledger_covers_121_unique_upstream_rows() {
    let mut rows = BTreeSet::new();
    let mut counts = BTreeMap::new();
    for group in LIVE_PROBE_ROWS {
        for line in group.lines {
            assert!(rows.insert((group.file, *line)));
            *counts.entry(group.probe).or_insert(0_usize) += 1;
        }
    }
    assert_eq!(rows.len(), 121);
    let keys = rows
        .iter()
        .map(|(file, line)| format!("{file}:{line}\n"))
        .collect::<String>();
    let hash = Sha256::digest(keys);
    assert_eq!(
        format!("{hash:x}"),
        "d38a1ee32b9bb0b9710e90413d19a5d4cbbfd21a489d6130c24a75b0fd793723"
    );
    let expected = [
        (LiveProbe::Abort, 6),
        (LiveProbe::AnthropicFeatures, 8),
        (LiveProbe::ContextOverflow, 4),
        (LiveProbe::CrossProviderHandoff, 2),
        (LiveProbe::EmptyMessages, 16),
        (LiveProbe::ImageToolResults, 10),
        (LiveProbe::InterleavedThinking, 2),
        (LiveProbe::CacheAffinity, 2),
        (LiveProbe::ReasoningReplay, 3),
        (LiveProbe::ResponseIds, 3),
        (LiveProbe::StreamMatrix, 41),
        (LiveProbe::AbortedUsage, 4),
        (LiveProbe::ToolCallIds, 2),
        (LiveProbe::MissingToolResults, 4),
        (LiveProbe::TotalTokens, 4),
        (LiveProbe::Unicode, 8),
        (LiveProbe::XHigh, 2),
    ];
    assert_eq!(counts, BTreeMap::from(expected));
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LiveProvider {
    OpenAi,
    AnthropicApiKey,
    AnthropicOAuth,
    Codex,
}

#[derive(Clone, Copy, Debug)]
struct LiveTarget {
    provider: LiveProvider,
    model: &'static str,
    transport: Option<Transport>,
}

const OPENAI_4O: LiveTarget = LiveTarget::new(LiveProvider::OpenAi, "gpt-4o");
const OPENAI_54: LiveTarget = LiveTarget::new(LiveProvider::OpenAi, "gpt-5.4");
const OPENAI_54_MINI: LiveTarget = LiveTarget::new(LiveProvider::OpenAi, "gpt-5.4-mini");
const OPENAI_55: LiveTarget = LiveTarget::new(LiveProvider::OpenAi, "gpt-5.5");
const OPENAI_MINI: LiveTarget = LiveTarget::new(LiveProvider::OpenAi, "gpt-5-mini");
const ANTHROPIC_HAIKU: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicApiKey, "claude-haiku-4-5");
const ANTHROPIC_SONNET: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicApiKey, "claude-sonnet-4-5");
const ANTHROPIC_SONNET_46: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicApiKey, "claude-sonnet-4-6");
const ANTHROPIC_OAUTH_SONNET: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicOAuth, "claude-sonnet-4-6");
const ANTHROPIC_OPUS_45: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicApiKey, "claude-opus-4-5");
const ANTHROPIC_OPUS_46: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicApiKey, "claude-opus-4-6");
const ANTHROPIC_OAUTH_OPUS_46: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicOAuth, "claude-opus-4-6");
const ANTHROPIC_OPUS_48: LiveTarget =
    LiveTarget::new(LiveProvider::AnthropicApiKey, "claude-opus-4-8");
const CODEX_54: LiveTarget = LiveTarget::new(LiveProvider::Codex, "gpt-5.4");
const CODEX_55: LiveTarget = LiveTarget::new(LiveProvider::Codex, "gpt-5.5");
const CODEX_55_WS: LiveTarget =
    LiveTarget::new(LiveProvider::Codex, "gpt-5.5").with_transport(Transport::WebSocket);

impl LiveTarget {
    const fn new(provider: LiveProvider, model: &'static str) -> Self {
        Self {
            provider,
            model,
            transport: None,
        }
    }

    const fn with_transport(mut self, transport: Transport) -> Self {
        self.transport = Some(transport);
        self
    }

    fn model(self) -> Model {
        builtin_model(self.provider_id(), self.model)
            .unwrap_or_else(|| panic!("unknown {}/{}", self.provider_id(), self.model))
    }

    fn provider_id(self) -> &'static str {
        match self.provider {
            LiveProvider::OpenAi => "openai",
            LiveProvider::AnthropicApiKey | LiveProvider::AnthropicOAuth => "anthropic",
            LiveProvider::Codex => "openai-codex",
        }
    }

    fn credential(self) -> String {
        required(match self.provider {
            LiveProvider::OpenAi => "OPENAI_API_KEY",
            LiveProvider::AnthropicApiKey => "ANTHROPIC_API_KEY",
            LiveProvider::AnthropicOAuth => "ANTHROPIC_OAUTH_TOKEN",
            LiveProvider::Codex => "DS_AI_CODEX_ACCESS_TOKEN",
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum LiveReasoning {
    #[default]
    None,
    Medium,
    High,
    XHigh,
}

#[derive(Clone, Default)]
struct LiveCall {
    cancellation: CancellationToken,
    max_tokens: Option<u64>,
    reasoning: LiveReasoning,
    force_tool: Option<String>,
    cache_retention: CacheRetention,
    session_id: Option<String>,
    capture: Option<Arc<Mutex<Option<serde_json::Value>>>>,
}

fn live_stream(
    target: LiveTarget,
    context: &Context,
    call: LiveCall,
) -> AssistantMessageEventStream {
    live_stream_model(target, target.model(), context, call)
}

fn live_stream_model(
    target: LiveTarget,
    model: Model,
    context: &Context,
    call: LiveCall,
) -> AssistantMessageEventStream {
    let capture = call.capture.map(|capture| {
        PayloadHook::new(move |payload, _| {
            let capture = capture.clone();
            async move {
                *capture.lock().unwrap() = Some(payload);
                Ok(None)
            }
        })
    });
    let stream = StreamOptions {
        api_key: Some(target.credential()),
        cancellation: call.cancellation,
        max_retries: Some(2),
        max_tokens: call.max_tokens.or(Some(4096)),
        timeout: Some(Duration::from_secs(120)),
        transport: target.transport,
        cache_retention: call.cache_retention,
        session_id: call.session_id,
        on_payload: capture,
        ..Default::default()
    };
    match target.provider {
        LiveProvider::OpenAi => ds_ai::openai::stream(
            &model.typed::<OpenAiResponsesOptions>().unwrap(),
            context,
            &OpenAiResponsesOptions {
                stream,
                reasoning_effort: openai_reasoning(call.reasoning),
                reasoning_summary: (call.reasoning != LiveReasoning::None)
                    .then_some(ds_ai::openai::ReasoningSummary::Auto),
                tool_choice: call.force_tool.map(ds_ai::openai::ToolChoice::Function),
                ..Default::default()
            },
        ),
        LiveProvider::AnthropicApiKey | LiveProvider::AnthropicOAuth => ds_ai::anthropic::stream(
            &model.typed::<AnthropicOptions>().unwrap(),
            context,
            &AnthropicOptions {
                stream,
                thinking_enabled: (call.reasoning != LiveReasoning::None).then_some(true),
                thinking_budget_tokens: (call.reasoning != LiveReasoning::None).then_some(2048),
                effort: anthropic_reasoning(call.reasoning),
                interleaved_thinking: Some(true),
                tool_choice: call.force_tool.map(ds_ai::anthropic::ToolChoice::Tool),
                ..Default::default()
            },
        ),
        LiveProvider::Codex => ds_ai::codex::stream(
            &model.typed::<OpenAiCodexResponsesOptions>().unwrap(),
            context,
            &OpenAiCodexResponsesOptions {
                stream,
                reasoning_effort: codex_reasoning(call.reasoning),
                reasoning_summary: (call.reasoning != LiveReasoning::None)
                    .then_some(ds_ai::codex::ReasoningSummary::Auto),
                tool_choice: call.force_tool.map(|_| ds_ai::codex::ToolChoice::Required),
                ..Default::default()
            },
        ),
    }
}

fn openai_reasoning(reasoning: LiveReasoning) -> Option<ds_ai::openai::ReasoningEffort> {
    match reasoning {
        LiveReasoning::None => None,
        LiveReasoning::Medium => Some(ds_ai::openai::ReasoningEffort::Medium),
        LiveReasoning::High => Some(ds_ai::openai::ReasoningEffort::High),
        LiveReasoning::XHigh => Some(ds_ai::openai::ReasoningEffort::XHigh),
    }
}

fn anthropic_reasoning(reasoning: LiveReasoning) -> Option<ds_ai::anthropic::Effort> {
    match reasoning {
        LiveReasoning::None => None,
        LiveReasoning::Medium => Some(ds_ai::anthropic::Effort::Medium),
        LiveReasoning::High => Some(ds_ai::anthropic::Effort::High),
        LiveReasoning::XHigh => Some(ds_ai::anthropic::Effort::XHigh),
    }
}

fn codex_reasoning(reasoning: LiveReasoning) -> Option<ds_ai::codex::ReasoningEffort> {
    match reasoning {
        LiveReasoning::None => None,
        LiveReasoning::Medium => Some(ds_ai::codex::ReasoningEffort::Medium),
        LiveReasoning::High => Some(ds_ai::codex::ReasoningEffort::High),
        LiveReasoning::XHigh => Some(ds_ai::codex::ReasoningEffort::XHigh),
    }
}

async fn live_complete(target: LiveTarget, context: &Context, call: LiveCall) -> AssistantMessage {
    live_stream(target, context, call).result().await.unwrap()
}

fn assert_success(message: &AssistantMessage) {
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{:?}",
        message.error_message
    );
    assert_ne!(message.stop_reason, StopReason::Aborted);
    assert!(!message.content.is_empty());
}

fn message_text(message: &AssistantMessage) -> String {
    message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn message_tool_call(message: &AssistantMessage) -> AssistantToolCall {
    message
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .expect("missing tool call")
}

fn tool(name: &str) -> Tool {
    Tool::new(
        name,
        "Return the requested value",
        serde_json::json!({
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": false
        }),
    )
}

async fn live_tool_call(
    target: LiveTarget,
    name: &str,
    reasoning: LiveReasoning,
) -> (Context, AssistantMessage, AssistantToolCall) {
    live_tool_call_with_capture(target, name, reasoning, None).await
}

async fn live_tool_call_with_capture(
    target: LiveTarget,
    name: &str,
    reasoning: LiveReasoning,
    capture: Option<Arc<Mutex<Option<serde_json::Value>>>>,
) -> (Context, AssistantMessage, AssistantToolCall) {
    let context = Context::new([Message::user(format!(
        "Call {name} with value set to live-probe"
    ))])
    .with_system(format!("Use {name} when asked."))
    .with_tools([tool(name)]);
    let response = live_complete(
        target,
        &context,
        LiveCall {
            reasoning,
            force_tool: Some(name.into()),
            capture,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(
        response.stop_reason,
        StopReason::ToolUse,
        "{:?}",
        response.error_message
    );
    let call = message_tool_call(&response);
    assert_eq!(call.name, name);
    (context, response, call)
}

fn tool_result(
    call: &AssistantToolCall,
    content: impl IntoIterator<Item = InputContent>,
) -> Message {
    Message::tool_result(ToolResultMessage::new(&call.id, &call.name, content))
}

fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .try_into()
        .unwrap()
}

fn empty_assistant(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: timestamp(),
    }
}
