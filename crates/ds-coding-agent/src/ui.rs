use crate::presentation;
use ds_agent_core::{Agent, AgentEvent, AgentOutcome};
use ds_ai::{StopReason, content_text};
use futures_util::StreamExt;
use std::io::{self, Write};
use tokio_util::sync::CancellationToken;

pub use crate::interactive::{InteractiveBackend, ProviderChoice, run_interactive};

pub async fn run_one_shot(agent: &mut Agent, prompt: String) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let mut assistant_open = false;
    let mut last_error = None;
    let mut events = agent.run(prompt, CancellationToken::new());
    while let Some(event) = events.next().await {
        match event {
            AgentEvent::UserCommitted { .. } | AgentEvent::ToolStarted { .. } => {}
            AgentEvent::AssistantTextDelta { text } => {
                write!(stdout, "{text}")?;
                stdout.flush()?;
                assistant_open = true;
            }
            AgentEvent::AssistantFinished { message } => {
                if !assistant_open {
                    let text = content_text(&message.content);
                    if !text.is_empty() {
                        write!(stdout, "{text}")?;
                    }
                }
                if assistant_open || !message.content.is_empty() {
                    writeln!(stdout)?;
                }
                last_error = message.error_message.clone();
                assistant_open = false;
            }
            AgentEvent::ToolFinished {
                name,
                arguments,
                output,
                duration,
                ..
            } => {
                if assistant_open {
                    writeln!(stdout)?;
                    assistant_open = false;
                }
                let presentation =
                    presentation::finished_tool(&name, &arguments, &output, duration);
                writeln!(stdout, "{}", presentation.headline)?;
                for line in presentation.output {
                    writeln!(stdout, "  {line}")?;
                }
            }
            AgentEvent::Finished { outcome } => {
                stdout.flush()?;
                return outcome_result(outcome, last_error);
            }
            AgentEvent::Failed { message } => return Err(io::Error::other(message)),
        }
    }
    Err(io::Error::other(
        "agent event stream ended without a terminal outcome",
    ))
}

fn outcome_result(outcome: AgentOutcome, error: Option<String>) -> io::Result<()> {
    match outcome {
        AgentOutcome::Stopped(StopReason::Error | StopReason::Aborted) => Err(io::Error::other(
            error.unwrap_or_else(|| "model request failed".into()),
        )),
        AgentOutcome::Stopped(
            StopReason::Stop
            | StopReason::Length
            | StopReason::ToolUse
            | StopReason::Deferred
            | StopReason::Pending,
        )
        | AgentOutcome::MaxTurns => Ok(()),
    }
}
