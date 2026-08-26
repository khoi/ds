use crate::{ToolExecutionContext, ToolOutput, ToolRegistry};
use ds_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    Context, InputContent, Message, Model, Models, SimpleStreamOptions, StopReason,
    ToolResultMessage, validate_tool_call,
};
use futures_core::Stream;
use futures_util::StreamExt;
use std::{path::PathBuf, pin::Pin, sync::Arc};
use tokio_util::sync::CancellationToken;

pub const DEFAULT_MAX_TURNS: usize = 24;

pub trait AgentModelStream: Send + Sync {
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream;
}

impl AgentModelStream for Models {
    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        Models::stream_simple(self, model, context, options)
    }
}

pub struct Agent {
    model: Model,
    model_stream: Arc<dyn AgentModelStream>,
    context: Context,
    tools: ToolRegistry,
    working_directory: PathBuf,
    options: SimpleStreamOptions,
    max_turns: usize,
}

impl Agent {
    pub fn new(
        model: Model,
        model_stream: Arc<dyn AgentModelStream>,
        tools: ToolRegistry,
        working_directory: PathBuf,
    ) -> Self {
        let context = Context::new([]).with_tools(tools.declarations());
        Self {
            model,
            model_stream,
            context,
            tools,
            working_directory,
            options: SimpleStreamOptions::default(),
            max_turns: DEFAULT_MAX_TURNS,
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.context.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_options(mut self, options: SimpleStreamOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    pub fn run<'a>(
        &'a mut self,
        prompt: impl Into<String>,
        cancellation: CancellationToken,
    ) -> AgentEventStream<'a> {
        let prompt = prompt.into();
        Box::pin(async_stream::stream! {
            self.context.messages.push(Message::user(prompt.clone()));
            yield AgentEvent::UserCommitted { text: prompt };

            for _ in 0..self.max_turns {
                let mut options = self.options.clone();
                options.stream.cancellation = cancellation.clone();
                let mut events = self.model_stream.stream_simple(
                    &self.model,
                    &self.context,
                    &options,
                );
                let mut settled = None;

                while let Some(event) = events.next().await {
                    match event {
                        AssistantMessageEvent::TextDelta { delta, .. } => {
                            yield AgentEvent::AssistantTextDelta { text: delta };
                        }
                        AssistantMessageEvent::Done { message, .. } => {
                            settled = Some(SettledAssistantTurn::Done(message));
                            break;
                        }
                        AssistantMessageEvent::Error { error, .. } => {
                            settled = Some(SettledAssistantTurn::Error(error));
                            break;
                        }
                        AssistantMessageEvent::Start { .. }
                        | AssistantMessageEvent::TextStart { .. }
                        | AssistantMessageEvent::TextEnd { .. }
                        | AssistantMessageEvent::ThinkingStart { .. }
                        | AssistantMessageEvent::ThinkingDelta { .. }
                        | AssistantMessageEvent::ThinkingEnd { .. }
                        | AssistantMessageEvent::ToolCallStart { .. }
                        | AssistantMessageEvent::ToolCallDelta { .. }
                        | AssistantMessageEvent::ToolCallEnd { .. } => {}
                    }
                }

                let Some(settled) = settled else {
                    yield AgentEvent::Failed {
                        message: "assistant message stream ended without a terminal event".into(),
                    };
                    return;
                };
                let message = settled.message().clone();
                let stop_reason = message.stop_reason;
                self.context.messages.push(Message::assistant(message.clone()));
                yield AgentEvent::AssistantFinished {
                    message: Box::new(message),
                };

                if !settled.may_execute_tools() {
                    yield AgentEvent::Finished {
                        outcome: AgentOutcome::Stopped(stop_reason),
                    };
                    return;
                }

                for call in settled.tool_calls() {
                    yield AgentEvent::ToolStarted {
                        call_id: call.id.clone(),
                        name: call.name.clone(),
                    };
                    let output = match validate_tool_call(&self.context.tools, &call) {
                        Ok(arguments) => match self.tools.get(&call.name) {
                            Some(tool) => {
                                tool.execute(
                                        arguments,
                                        ToolExecutionContext {
                                            working_directory: &self.working_directory,
                                            cancellation: &cancellation,
                                        },
                                    )
                                    .await
                            }
                            None => ToolOutput::error(format!("Tool \"{}\" not found", call.name)),
                        },
                        Err(error) => ToolOutput::error(error.to_string()),
                    };
                    let result = ToolResultMessage::new(
                        call.id.clone(),
                        call.name.clone(),
                        [InputContent::text(output.content.text.clone())],
                    )
                    .with_error(output.is_error);
                    self.context.messages.push(Message::tool_result(result));
                    yield AgentEvent::ToolFinished {
                        call_id: call.id,
                        name: call.name,
                        output,
                    };
                }
            }

            yield AgentEvent::Finished {
                outcome: AgentOutcome::MaxTurns,
            };
        })
    }
}

pub type AgentEventStream<'a> = Pin<Box<dyn Stream<Item = AgentEvent> + Send + 'a>>;

#[derive(Clone, Debug, PartialEq)]
pub enum AgentEvent {
    UserCommitted {
        text: String,
    },
    AssistantTextDelta {
        text: String,
    },
    AssistantFinished {
        message: Box<AssistantMessage>,
    },
    ToolStarted {
        call_id: String,
        name: String,
    },
    ToolFinished {
        call_id: String,
        name: String,
        output: ToolOutput,
    },
    Finished {
        outcome: AgentOutcome,
    },
    Failed {
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentOutcome {
    Stopped(StopReason),
    MaxTurns,
}

enum SettledAssistantTurn {
    Done(AssistantMessage),
    Error(AssistantMessage),
}

impl SettledAssistantTurn {
    fn message(&self) -> &AssistantMessage {
        match self {
            Self::Done(message) | Self::Error(message) => message,
        }
    }

    fn may_execute_tools(&self) -> bool {
        matches!(self, Self::Done(message) if message.stop_reason == StopReason::ToolUse)
    }

    fn tool_calls(&self) -> Vec<ds_ai::AssistantToolCall> {
        if !self.may_execute_tools() {
            return Vec::new();
        }
        self.message()
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::ToolCall(call) => Some(call.clone()),
                AssistantContent::Text(_) | AssistantContent::Thinking(_) => None,
            })
            .collect()
    }
}
