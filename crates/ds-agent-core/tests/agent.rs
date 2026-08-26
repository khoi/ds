use async_trait::async_trait;
use ds_agent_core::{
    Agent, AgentEvent, AgentModelStream, AgentOutcome, AgentTool, BoundedText,
    ToolExecutionContext, ToolExecutor, ToolOutput, ToolRegistry,
};
use ds_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantToolCall, Context, DoneReason, ErrorReason, Message, Model, SimpleStreamOptions,
    StopReason, TextContent, Tool, Usage, builtin_openai_model,
};
use futures_util::{StreamExt, stream};
use serde_json::json;
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio_util::sync::CancellationToken;

struct ScriptedModel {
    turns: Mutex<VecDeque<Vec<AssistantMessageEvent>>>,
    contexts: Mutex<Vec<Context>>,
}

impl ScriptedModel {
    fn new(turns: impl IntoIterator<Item = Vec<AssistantMessageEvent>>) -> Self {
        Self {
            turns: Mutex::new(turns.into_iter().collect()),
            contexts: Mutex::new(Vec::new()),
        }
    }
}

impl AgentModelStream for ScriptedModel {
    fn stream_simple(
        &self,
        _model: &Model,
        context: &Context,
        _options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.contexts.lock().unwrap().push(context.clone());
        let events = self.turns.lock().unwrap().pop_front().unwrap_or_default();
        AssistantMessageEventStream::new(stream::iter(events))
    }
}

#[derive(Clone)]
struct RecordingTool {
    calls: Arc<AtomicUsize>,
    output: ToolOutput,
}

#[async_trait]
impl ToolExecutor for RecordingTool {
    async fn execute(
        &self,
        _arguments: serde_json::Value,
        _context: ToolExecutionContext<'_>,
    ) -> ToolOutput {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.output.clone()
    }
}

fn model() -> Model {
    builtin_openai_model("gpt-5.6-luna")
        .expect("test model")
        .into_model()
}

fn assistant(stop_reason: StopReason, content: Vec<AssistantContent>) -> AssistantMessage {
    let model = model();
    AssistantMessage {
        content,
        api: model.api,
        provider: model.provider,
        model: model.id,
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

fn text(value: &str) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: value.into(),
        text_signature: None,
    })
}

fn call(name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(AssistantToolCall {
        id: format!("{name}-1"),
        name: name.into(),
        arguments,
        thought_signature: None,
        namespace: None,
    })
}

fn done(message: AssistantMessage) -> AssistantMessageEvent {
    let reason = DoneReason::try_from(message.stop_reason).expect("done reason");
    AssistantMessageEvent::Done { reason, message }
}

fn agent(model_stream: Arc<dyn AgentModelStream>, tools: ToolRegistry) -> Agent {
    Agent::new(model(), model_stream, tools, PathBuf::from("/tmp"))
}

fn read_tool(calls: Arc<AtomicUsize>) -> AgentTool {
    AgentTool::new(
        Tool::new(
            "read",
            "read a file",
            json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
                "additionalProperties": false
            }),
        ),
        RecordingTool {
            calls,
            output: ToolOutput::success(BoundedText::new("file contents", false)),
        },
    )
}

#[tokio::test]
async fn streams_text_and_commits_terminal_message() {
    let message = assistant(StopReason::Stop, vec![text("hello")]);
    let partial = assistant(StopReason::Pending, vec![]);
    let scripted = Arc::new(ScriptedModel::new([vec![
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "hello".into(),
            partial,
        },
        done(message.clone()),
    ]]));
    let mut agent = agent(scripted, ToolRegistry::new([]).unwrap());

    let events = agent
        .run("hi", CancellationToken::new())
        .collect::<Vec<_>>()
        .await;

    assert!(events.contains(&AgentEvent::AssistantTextDelta {
        text: "hello".into()
    }));
    assert!(events.contains(&AgentEvent::AssistantFinished {
        message: Box::new(message)
    }));
    assert_eq!(
        events.last(),
        Some(&AgentEvent::Finished {
            outcome: AgentOutcome::Stopped(StopReason::Stop)
        })
    );
    assert_eq!(agent.context().messages.len(), 2);
}

#[tokio::test]
async fn appends_tool_result_before_follow_up_turn() {
    let calls = Arc::new(AtomicUsize::new(0));
    let first = assistant(
        StopReason::ToolUse,
        vec![call("read", json!({ "path": "Cargo.toml" }))],
    );
    let second = assistant(StopReason::Stop, vec![text("done")]);
    let scripted = Arc::new(ScriptedModel::new([vec![done(first)], vec![done(second)]]));
    let tools = ToolRegistry::new([read_tool(calls.clone())]).unwrap();
    let mut agent = agent(scripted.clone(), tools);

    let events = agent
        .run("inspect", CancellationToken::new())
        .collect::<Vec<_>>()
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AgentEvent::ToolFinished { .. }))
            .count(),
        1
    );
    let contexts = scripted.contexts.lock().unwrap();
    assert_eq!(contexts.len(), 2);
    assert!(matches!(contexts[1].messages[1], Message::Assistant(_)));
    let Message::ToolResult(result) = &contexts[1].messages[2] else {
        panic!("tool result follows assistant")
    };
    assert!(!result.is_error);
}

#[tokio::test]
async fn unknown_and_invalid_calls_become_error_results() {
    let calls = Arc::new(AtomicUsize::new(0));
    let first = assistant(
        StopReason::ToolUse,
        vec![call("missing", json!({})), call("read", json!({}))],
    );
    let second = assistant(StopReason::Stop, vec![text("recovered")]);
    let scripted = Arc::new(ScriptedModel::new([vec![done(first)], vec![done(second)]]));
    let mut agent = agent(
        scripted.clone(),
        ToolRegistry::new([read_tool(calls.clone())]).unwrap(),
    );

    agent
        .run("inspect", CancellationToken::new())
        .for_each(|_| async {})
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let contexts = scripted.contexts.lock().unwrap();
    for message in &contexts[1].messages[2..] {
        let Message::ToolResult(result) = message else {
            panic!("expected tool result")
        };
        assert!(result.is_error);
    }
}

#[tokio::test]
async fn length_response_never_executes_tool_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let message = assistant(
        StopReason::Length,
        vec![call("read", json!({ "path": "Cargo.toml" }))],
    );
    let scripted = Arc::new(ScriptedModel::new([vec![done(message)]]));
    let mut agent = agent(
        scripted,
        ToolRegistry::new([read_tool(calls.clone())]).unwrap(),
    );

    let events = agent
        .run("inspect", CancellationToken::new())
        .collect::<Vec<_>>()
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        events.last(),
        Some(&AgentEvent::Finished {
            outcome: AgentOutcome::Stopped(StopReason::Length)
        })
    );
}

#[tokio::test]
async fn aborted_response_finishes_without_tools() {
    let mut message = assistant(StopReason::Aborted, vec![]);
    message.error_message = Some("cancelled".into());
    let scripted = Arc::new(ScriptedModel::new([vec![AssistantMessageEvent::Error {
        reason: ErrorReason::Aborted,
        error: message,
    }]]));
    let mut agent = agent(scripted, ToolRegistry::new([]).unwrap());

    let events = agent
        .run("stop", CancellationToken::new())
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        events.last(),
        Some(&AgentEvent::Finished {
            outcome: AgentOutcome::Stopped(StopReason::Aborted)
        })
    );
}

#[tokio::test]
async fn stops_after_configured_model_turn_limit() {
    let calls = Arc::new(AtomicUsize::new(0));
    let turn = || {
        vec![done(assistant(
            StopReason::ToolUse,
            vec![call("read", json!({ "path": "Cargo.toml" }))],
        ))]
    };
    let scripted = Arc::new(ScriptedModel::new([turn(), turn(), turn()]));
    let mut agent = agent(
        scripted,
        ToolRegistry::new([read_tool(calls.clone())]).unwrap(),
    )
    .with_max_turns(2);

    let events = agent
        .run("loop", CancellationToken::new())
        .collect::<Vec<_>>()
        .await;

    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        events.last(),
        Some(&AgentEvent::Finished {
            outcome: AgentOutcome::MaxTurns
        })
    );
}
