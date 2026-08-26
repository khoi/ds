use crossterm::{
    cursor,
    event::{
        DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEvent,
        KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};
use ds_agent_core::{Agent, AgentEvent, AgentOutcome};
use ds_ai::{StopReason, content_text};
use futures_util::StreamExt;
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::Line,
    widgets::{Paragraph, Widget, Wrap},
};
use std::io::{self, IsTerminal, Stdout, Write};
use textarea::TextArea;
use tokio_util::sync::CancellationToken;

const VIEWPORT_HEIGHT: u16 = 10;

type InlineTerminal = Terminal<CrosstermBackend<Stdout>>;

pub async fn run_one_shot(agent: &mut Agent, prompt: String) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let mut assistant_open = false;
    let mut last_error = None;
    let mut events = agent.run(prompt, CancellationToken::new());
    while let Some(event) = events.next().await {
        match event {
            AgentEvent::UserCommitted { .. } => {}
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
            AgentEvent::ToolStarted { name, .. } => {
                if assistant_open {
                    writeln!(stdout)?;
                    assistant_open = false;
                }
                writeln!(stdout, "$ {name}")?;
            }
            AgentEvent::ToolFinished { output, .. } => {
                if !output.content.text.is_empty() {
                    writeln!(stdout, "{}", output.content.text)?;
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

pub async fn run_interactive(agent: &mut Agent) -> io::Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(io::Error::other(
            "interactive mode requires a terminal; pass a prompt for one-shot mode",
        ));
    }
    let terminal_mode = TerminalMode::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(VIEWPORT_HEIGHT),
        },
    )?;
    let result = interactive_loop(agent, &mut terminal).await;
    terminal.show_cursor()?;
    drop(terminal);
    drop(terminal_mode);
    result
}

async fn interactive_loop(agent: &mut Agent, terminal: &mut InlineTerminal) -> io::Result<()> {
    let mut input = EventStream::new();
    let mut state = InteractiveState::new();
    draw(terminal, &state)?;

    loop {
        let event = next_terminal_event(&mut input).await?;
        match event {
            Event::Key(key) if is_key_press(key) && is_ctrl_c(key) => {
                if state.prompt().is_empty() {
                    return Ok(());
                }
                state.clear_prompt();
            }
            Event::Key(key)
                if is_key_press(key)
                    && key.code == KeyCode::Char('d')
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && state.prompt().is_empty() =>
            {
                return Ok(());
            }
            Event::Key(key)
                if is_key_press(key)
                    && key.code == KeyCode::Enter
                    && !key.modifiers.contains(KeyModifiers::SHIFT) =>
            {
                let prompt = state.prompt();
                if prompt.is_empty() {
                    continue;
                }
                commit(terminal, &format!("› {prompt}"))?;
                state.clear_prompt();
                run_active(agent, terminal, &mut input, &mut state, prompt).await?;
            }
            Event::Key(key)
                if is_key_press(key)
                    && key.code == KeyCode::Char('j')
                    && key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                state.composer.insert_newline();
            }
            Event::Key(key) if is_key_press(key) => {
                state.composer.input(key);
            }
            Event::Paste(text) => {
                state.composer.insert_str(text);
            }
            Event::Resize(_, _) => {}
            Event::FocusGained | Event::FocusLost | Event::Mouse(_) | Event::Key(_) => continue,
        }
        draw(terminal, &state)?;
    }
}

async fn run_active(
    agent: &mut Agent,
    terminal: &mut InlineTerminal,
    input: &mut EventStream,
    state: &mut InteractiveState,
    prompt: String,
) -> io::Result<()> {
    let cancellation = CancellationToken::new();
    state.status = Some("working · ctrl-c cancel".into());
    draw(terminal, state)?;
    let mut events = agent.run(prompt, cancellation.clone());
    loop {
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else {
                    return Err(io::Error::other("agent event stream ended without a terminal outcome"));
                };
                let finished = apply_agent_event(terminal, state, event)?;
                draw(terminal, state)?;
                if finished {
                    return Ok(());
                }
            }
            event = input.next() => {
                let event = event.transpose()?.ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "terminal event stream ended"))?;
                match event {
                    Event::Key(key) if is_key_press(key) && is_ctrl_c(key) => {
                        cancellation.cancel();
                        state.status = Some("cancelling".into());
                    }
                    Event::Resize(_, _) => {}
                    Event::FocusGained | Event::FocusLost | Event::Key(_) | Event::Mouse(_) | Event::Paste(_) => continue,
                }
                draw(terminal, state)?;
            }
        }
    }
}

fn apply_agent_event(
    terminal: &mut InlineTerminal,
    state: &mut InteractiveState,
    event: AgentEvent,
) -> io::Result<bool> {
    match event {
        AgentEvent::UserCommitted { .. } => {}
        AgentEvent::AssistantTextDelta { text } => {
            state.assistant_streamed = true;
            state.assistant.push_str(&text);
            if let Some(last_newline) = state.assistant.rfind('\n') {
                let mut stable = state.assistant.drain(..=last_newline).collect::<String>();
                stable.pop();
                commit(terminal, &stable)?;
            }
        }
        AgentEvent::AssistantFinished { message } => {
            if !state.assistant_streamed {
                state.assistant = content_text(&message.content);
            }
            if state.assistant.is_empty()
                && let Some(error) = &message.error_message
            {
                state.assistant = format!("error: {error}");
            }
            if !state.assistant.is_empty() {
                commit(terminal, &state.assistant)?;
                state.assistant.clear();
            }
            state.status = None;
            state.assistant_streamed = false;
        }
        AgentEvent::ToolStarted { name, .. } => state.status = Some(format!("$ {name}")),
        AgentEvent::ToolFinished { name, output, .. } => {
            let text = if output.content.text.is_empty() {
                format!("$ {name}")
            } else {
                format!("$ {name}\n{}", output.content.text)
            };
            commit(terminal, &text)?;
            state.status = Some("working · ctrl-c cancel".into());
        }
        AgentEvent::Finished { outcome } => {
            if outcome == AgentOutcome::MaxTurns {
                commit(terminal, "stopped after 24 model turns")?;
            }
            state.status = None;
            return Ok(true);
        }
        AgentEvent::Failed { message } => {
            commit(terminal, &format!("error: {message}"))?;
            state.status = None;
            return Ok(true);
        }
    }
    Ok(false)
}

fn draw(terminal: &mut InlineTerminal, state: &InteractiveState) -> io::Result<()> {
    terminal.draw(|frame| {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(1),
                Constraint::Length(2),
            ])
            .split(frame.area());
        frame.render_widget(
            Paragraph::new(state.assistant.as_str()).wrap(Wrap { trim: false }),
            rows[0],
        );
        frame.render_widget(
            Paragraph::new(state.status.as_deref().unwrap_or(""))
                .style(Style::default().fg(Color::DarkGray)),
            rows[1],
        );
        let composer = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(2), Constraint::Min(1)])
            .split(rows[2]);
        frame.render_widget(Line::from("› "), composer[0]);
        frame.render_widget(&state.composer, composer[1]);
    })?;
    Ok(())
}

fn commit(terminal: &mut InlineTerminal, text: &str) -> io::Result<()> {
    let width = terminal.size()?.width.max(1);
    let paragraph = Paragraph::new(text.to_owned()).wrap(Wrap { trim: false });
    let height = rendered_height(text, width);
    terminal.insert_before(height, move |buffer| {
        paragraph.render(buffer.area, buffer);
    })?;
    Ok(())
}

fn rendered_height(text: &str, width: u16) -> u16 {
    let width = usize::from(width.max(1));
    let text = text.strip_suffix('\n').unwrap_or(text);
    let lines = text
        .split('\n')
        .map(|line| Line::from(line).width().div_ceil(width).max(1))
        .sum::<usize>();
    u16::try_from(lines).unwrap_or(u16::MAX).max(1)
}

fn is_key_press(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
}

fn is_ctrl_c(key: KeyEvent) -> bool {
    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL)
}

async fn next_terminal_event(input: &mut EventStream) -> io::Result<Event> {
    input
        .next()
        .await
        .transpose()?
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "terminal event stream ended"))
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

struct InteractiveState {
    assistant: String,
    assistant_streamed: bool,
    status: Option<String>,
    composer: TextArea<'static>,
}

impl InteractiveState {
    fn new() -> Self {
        let mut composer = TextArea::default();
        composer.set_cursor_line_style(Style::default());
        Self {
            assistant: String::new(),
            assistant_streamed: false,
            status: None,
            composer,
        }
    }

    fn prompt(&self) -> String {
        self.composer.lines().join("\n").trim().to_owned()
    }

    fn clear_prompt(&mut self) {
        self.composer = TextArea::default();
        self.composer.set_cursor_line_style(Style::default());
    }
}

struct TerminalMode;

impl TerminalMode {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = write_terminal_commands(&mut io::stdout(), true) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        Ok(Self)
    }
}

impl Drop for TerminalMode {
    fn drop(&mut self) {
        let _ = write_terminal_commands(&mut io::stdout(), false);
        let _ = disable_raw_mode();
    }
}

fn write_terminal_commands(writer: &mut impl Write, entering: bool) -> io::Result<()> {
    if entering {
        execute!(writer, EnableBracketedPaste, cursor::Show)?;
    } else {
        execute!(writer, DisableBracketedPaste, cursor::Show)?;
    }
    writer.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_commands_never_enter_or_leave_alternate_screen() {
        let mut bytes = Vec::new();
        write_terminal_commands(&mut bytes, true).unwrap();
        write_terminal_commands(&mut bytes, false).unwrap();

        assert!(!bytes.windows(8).any(|bytes| bytes == b"\x1b[?1049h"));
        assert!(!bytes.windows(8).any(|bytes| bytes == b"\x1b[?1049l"));
        assert!(bytes.windows(8).any(|bytes| bytes == b"\x1b[?2004h"));
        assert!(bytes.windows(8).any(|bytes| bytes == b"\x1b[?2004l"));
    }

    #[test]
    fn prompt_is_trimmed_before_submission() {
        let state = InteractiveState {
            assistant: String::new(),
            assistant_streamed: false,
            status: None,
            composer: TextArea::from(["  hello", "world  "]),
        };

        assert_eq!(state.prompt(), "hello\nworld");
    }
}
