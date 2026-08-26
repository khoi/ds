use crate::{
    commands::{self, LoginMethod, SlashCommand},
    presentation::{self, PresentationOutcome, ToolPresentation},
};
use async_trait::async_trait;
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
use ds_ai::{Model, StopReason, content_text};
use futures_util::StreamExt;
use ratatui::{
    Terminal, TerminalOptions, Viewport,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Paragraph, Widget, Wrap},
};
use std::io::{self, IsTerminal, Stdout, Write};
use textarea::TextArea;
use tokio_util::sync::CancellationToken;

const VIEWPORT_HEIGHT: u16 = 14;
const PICKER_VISIBLE_ITEMS: usize = 6;
const SUGGESTION_VISIBLE_ITEMS: usize = 4;

type InlineTerminal = Terminal<CrosstermBackend<Stdout>>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderChoice {
    pub id: String,
    pub name: String,
    pub supports_api_key: bool,
    pub supports_oauth: bool,
}

#[async_trait]
pub trait InteractiveBackend: Send {
    fn models(&self) -> Vec<Model>;
    fn providers(&self) -> Vec<ProviderChoice>;
    fn reasoning_label(&self) -> String;
    fn take_startup_notice(&mut self) -> Option<String> {
        None
    }
    fn persist_model(&mut self, model: &Model) -> Result<(), String>;
    async fn provider_configured(&self, provider: &str) -> Result<bool, String>;
    async fn login(&mut self, provider: &str, method: Option<LoginMethod>) -> Result<(), String>;
    async fn logout(&mut self, provider: &str) -> Result<(), String>;
    async fn auth_status(&self, provider: Option<&str>) -> Result<String, String>;
}

pub async fn run_interactive(
    agent: &mut Agent,
    backend: &mut dyn InteractiveBackend,
) -> io::Result<()> {
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Err(io::Error::other(
            "interactive mode requires a terminal; pass a prompt for one-shot mode",
        ));
    }
    let mut terminal_mode = TerminalMode::enter()?;
    let mut terminal = inline_terminal()?;
    let result = interactive_loop(agent, backend, &mut terminal, &mut terminal_mode).await;
    terminal.show_cursor()?;
    drop(terminal);
    drop(terminal_mode);
    result
}

fn inline_terminal() -> io::Result<InlineTerminal> {
    Terminal::with_options(
        CrosstermBackend::new(io::stdout()),
        TerminalOptions {
            viewport: Viewport::Inline(VIEWPORT_HEIGHT),
        },
    )
}

async fn interactive_loop(
    agent: &mut Agent,
    backend: &mut dyn InteractiveBackend,
    terminal: &mut InlineTerminal,
    terminal_mode: &mut TerminalMode,
) -> io::Result<()> {
    let mut input = EventStream::new();
    let mut state = InteractiveState::new(footer_text(agent, backend));
    draw(terminal, &state)?;
    if let Some(notice) = backend.take_startup_notice() {
        commit_error(terminal, &notice)?;
        draw(terminal, &state)?;
    }

    loop {
        let event = next_terminal_event(&mut input).await?;
        let has_user_input = match &event {
            Event::Key(key) => is_key_press(*key),
            Event::Paste(_) => true,
            _ => false,
        };
        if state.picker.is_none() && has_user_input {
            state.status = None;
        }
        if state.picker.is_some() {
            if let Event::Key(key) = event
                && is_key_press(key)
                && let Some(selection) = handle_picker_key(&mut state, key)
            {
                handle_picker_selection(
                    agent,
                    backend,
                    terminal,
                    terminal_mode,
                    &mut state,
                    selection,
                )
                .await?;
            }
            draw(terminal, &state)?;
            continue;
        }

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
            Event::Key(key) if is_key_press(key) && key.code == KeyCode::Tab => {
                if let Some(suggestion) = commands::suggestions(&state.prompt()).first() {
                    state.set_prompt(&format!("{} ", suggestion.command));
                }
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
                commit_plain(terminal, &format!("› {prompt}"), Style::default())?;
                state.clear_prompt();
                if let Some(command) = commands::parse(&prompt) {
                    drop(input);
                    let keep_running = handle_slash_command(
                        agent,
                        backend,
                        terminal,
                        terminal_mode,
                        &mut state,
                        command,
                    )
                    .await?;
                    input = EventStream::new();
                    if !keep_running {
                        return Ok(());
                    }
                } else {
                    run_active(agent, terminal, &mut input, &mut state, prompt).await?;
                }
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

async fn handle_slash_command(
    agent: &mut Agent,
    backend: &mut dyn InteractiveBackend,
    terminal: &mut InlineTerminal,
    terminal_mode: &mut TerminalMode,
    state: &mut InteractiveState,
    command: SlashCommand,
) -> io::Result<bool> {
    match command {
        SlashCommand::Help => commit_plain(
            terminal,
            &commands::help_text(),
            Style::default().fg(Color::DarkGray),
        )?,
        SlashCommand::Models { query } => {
            let models = backend.models();
            if let Some(model) = exact_model(&models, &query) {
                switch_model(agent, backend, terminal, state, model).await?;
            } else if models.is_empty() {
                commit_error(terminal, "no models are registered")?;
            } else {
                state.picker = Some(Picker::models(models, query, agent.model()));
            }
        }
        SlashCommand::Login { provider, method } => match provider {
            Some(provider) => {
                let choice = backend
                    .providers()
                    .into_iter()
                    .find(|choice| choice.id == provider);
                if choice.is_none() {
                    commit_error(terminal, &format!("unknown provider: {provider}"))?;
                } else if !choice
                    .as_ref()
                    .is_some_and(|choice| supports_login_method(choice, method))
                {
                    commit_error(
                        terminal,
                        &format!(
                            "provider {provider} does not support {} login",
                            login_method_label(method)
                        ),
                    )?;
                } else {
                    run_login(backend, terminal, terminal_mode, state, &provider, method).await?;
                }
            }
            None => state.picker = Some(Picker::login(backend.providers(), method)),
        },
        SlashCommand::Logout { provider } => match provider {
            Some(provider) => run_logout(backend, terminal, state, &provider).await?,
            None => state.picker = Some(Picker::logout(backend.providers())),
        },
        SlashCommand::Auth { provider } => {
            state.status = Some("checking auth".into());
            draw(terminal, state)?;
            let result = backend.auth_status(provider.as_deref()).await;
            state.status = None;
            match result {
                Ok(status) => commit_plain(terminal, &status, Style::default())?,
                Err(error) => commit_error(terminal, &error)?,
            }
        }
        SlashCommand::Status => {
            commit_plain(
                terminal,
                &state.footer,
                Style::default().fg(Color::DarkGray),
            )?;
        }
        SlashCommand::Clear => terminal.clear()?,
        SlashCommand::Quit => return Ok(false),
        SlashCommand::Unknown { command } => {
            commit_error(terminal, &format!("unknown command: {command} · use /help"))?;
        }
    }
    Ok(true)
}

async fn handle_picker_selection(
    agent: &mut Agent,
    backend: &mut dyn InteractiveBackend,
    terminal: &mut InlineTerminal,
    terminal_mode: &mut TerminalMode,
    state: &mut InteractiveState,
    selection: PickerSelection,
) -> io::Result<()> {
    state.picker = None;
    match selection {
        PickerSelection::Model(model) => {
            switch_model(agent, backend, terminal, state, *model).await?
        }
        PickerSelection::Login { provider, method } => {
            run_login(backend, terminal, terminal_mode, state, &provider, method).await?
        }
        PickerSelection::Logout { provider } => {
            run_logout(backend, terminal, state, &provider).await?
        }
    }
    Ok(())
}

async fn switch_model(
    agent: &mut Agent,
    backend: &mut dyn InteractiveBackend,
    terminal: &mut InlineTerminal,
    state: &mut InteractiveState,
    model: Model,
) -> io::Result<()> {
    let model_name = model_identifier(&model);
    if let Err(error) = backend.persist_model(&model) {
        commit_error(terminal, &format!("could not save model: {error}"))?;
        return Ok(());
    }
    if agent.model().is_same_as(&model) {
        commit_plain(
            terminal,
            &format!("using {model_name} · saved as default"),
            Style::default().fg(Color::DarkGray),
        )?;
        return Ok(());
    }
    let provider = model.provider.as_str().to_owned();
    agent.set_model(model);
    state.footer = footer_text(agent, backend);
    let suffix = match backend.provider_configured(&provider).await {
        Ok(true) => String::new(),
        Ok(false) => format!(" · use /login {provider} before prompting"),
        Err(error) => format!(" · auth check failed: {error}"),
    };
    commit_plain(
        terminal,
        &format!("switched to {model_name}{suffix}"),
        Style::default().fg(Color::Cyan),
    )
}

async fn run_login(
    backend: &mut dyn InteractiveBackend,
    terminal: &mut InlineTerminal,
    terminal_mode: &mut TerminalMode,
    state: &mut InteractiveState,
    provider: &str,
    method: Option<LoginMethod>,
) -> io::Result<()> {
    state.status = Some(format!("starting {provider} login"));
    draw(terminal, state)?;
    terminal.show_cursor()?;
    terminal_mode.suspend()?;
    let result = backend.login(provider, method).await;
    terminal_mode.resume()?;
    let (message, success) = match result {
        Ok(()) => (format!("logged in to {provider}"), true),
        Err(error) if error.contains("cancelled") => ("login cancelled".into(), false),
        Err(error) => (format!("login failed: {error}"), false),
    };
    state.status = Some(message.clone());
    if success {
        commit_plain(terminal, &message, Style::default().fg(Color::Green))
    } else {
        commit_error(terminal, &message)
    }
}

async fn run_logout(
    backend: &mut dyn InteractiveBackend,
    terminal: &mut InlineTerminal,
    state: &mut InteractiveState,
    provider: &str,
) -> io::Result<()> {
    state.status = Some(format!("logging out of {provider}"));
    draw(terminal, state)?;
    let result = backend.logout(provider).await;
    state.status = None;
    match result {
        Ok(()) => commit_plain(
            terminal,
            &format!("logged out of {provider}"),
            Style::default().fg(Color::Green),
        ),
        Err(error) => commit_error(terminal, &format!("logout failed: {error}")),
    }
}

fn supports_login_method(provider: &ProviderChoice, method: Option<LoginMethod>) -> bool {
    match method {
        Some(LoginMethod::ApiKey) => provider.supports_api_key,
        Some(LoginMethod::OAuth) => provider.supports_oauth,
        None => provider.supports_api_key || provider.supports_oauth,
    }
}

fn login_method_label(method: Option<LoginMethod>) -> &'static str {
    match method {
        Some(LoginMethod::ApiKey) => "API key",
        Some(LoginMethod::OAuth) => "OAuth",
        None => "any supported",
    }
}

fn exact_model(models: &[Model], query: &str) -> Option<Model> {
    if query.is_empty() {
        return None;
    }
    let query = query.to_ascii_lowercase();
    models
        .iter()
        .find(|model| model_identifier(model).to_ascii_lowercase() == query)
        .cloned()
}

fn model_identifier(model: &Model) -> String {
    format!("{}/{}", model.provider, model.id)
}

fn footer_text(agent: &Agent, backend: &dyn InteractiveBackend) -> String {
    let directory = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "?".into());
    format!(
        "{} · reasoning {} · {}",
        model_identifier(agent.model()),
        backend.reasoning_label(),
        directory
    )
}

async fn run_active(
    agent: &mut Agent,
    terminal: &mut InlineTerminal,
    input: &mut EventStream,
    state: &mut InteractiveState,
    prompt: String,
) -> io::Result<()> {
    let cancellation = CancellationToken::new();
    let max_turns = agent.max_turns();
    state.status = Some("working · ctrl-c cancel".into());
    draw(terminal, state)?;
    let mut events = agent.run(prompt, cancellation.clone());
    loop {
        tokio::select! {
            event = events.next() => {
                let Some(event) = event else {
                    return Err(io::Error::other("agent event stream ended without a terminal outcome"));
                };
                let finished = apply_agent_event(terminal, state, max_turns, event)?;
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
    max_turns: usize,
    event: AgentEvent,
) -> io::Result<bool> {
    match event {
        AgentEvent::UserCommitted { .. } => {}
        AgentEvent::AssistantTextDelta { text } => state.assistant.push_str(&text),
        AgentEvent::AssistantFinished { message } => {
            if state.assistant.is_empty() {
                state.assistant = content_text(&message.content);
            }
            if !state.assistant.is_empty() {
                commit_markdown(terminal, &state.assistant)?;
            }
            if let Some(error) =
                assistant_error(message.stop_reason, message.error_message.as_deref())
            {
                commit_error(terminal, error)?;
            }
            state.assistant.clear();
            state.status = None;
        }
        AgentEvent::ToolStarted {
            name, arguments, ..
        } => state.status = Some(presentation::active_tool(&name, &arguments)),
        AgentEvent::ToolFinished {
            name,
            arguments,
            output,
            duration,
            ..
        } => {
            let presentation = presentation::finished_tool(&name, &arguments, &output, duration);
            commit_tool(terminal, presentation)?;
            state.status = Some("working · ctrl-c cancel".into());
        }
        AgentEvent::Finished { outcome } => {
            if outcome == AgentOutcome::MaxTurns {
                commit_error(terminal, &format!("stopped after {max_turns} model turns"))?;
            }
            state.status = None;
            return Ok(true);
        }
        AgentEvent::Failed { message } => {
            commit_error(terminal, &message)?;
            state.status = None;
            return Ok(true);
        }
    }
    Ok(false)
}

fn assistant_error(stop_reason: StopReason, error_message: Option<&str>) -> Option<&str> {
    (stop_reason == StopReason::Error)
        .then_some(error_message)
        .flatten()
}

fn draw(terminal: &mut InlineTerminal, state: &InteractiveState) -> io::Result<()> {
    terminal.draw(|frame| {
        let suggestions = commands::suggestions(&state.prompt());
        let overlay_height = if let Some(picker) = &state.picker {
            picker.rendered_height()
        } else {
            suggestions.len().min(SUGGESTION_VISIBLE_ITEMS) as u16
        };
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(overlay_height),
                Constraint::Length(1),
                Constraint::Length(2),
            ])
            .split(frame.area());

        let assistant_safe = terminal_safe_text(&state.assistant);
        let assistant_text = tui_markdown::from_str(&assistant_safe);
        let line_count = rendered_text_height(&assistant_text, rows[0].width);
        let assistant = Paragraph::new(assistant_text).wrap(Wrap { trim: false });
        let scroll = line_count.saturating_sub(usize::from(rows[0].height));
        frame.render_widget(
            assistant.scroll((u16::try_from(scroll).unwrap_or(u16::MAX), 0)),
            rows[0],
        );

        if let Some(picker) = &state.picker {
            frame.render_widget(Paragraph::new(picker.render_lines()), rows[1]);
        } else if !suggestions.is_empty() {
            let lines = suggestions
                .iter()
                .take(SUGGESTION_VISIBLE_ITEMS)
                .map(|spec| {
                    Line::from(vec![
                        Span::styled(spec.command, Style::default().fg(Color::Cyan)),
                        Span::styled(
                            format!("  {}", spec.description),
                            Style::default().fg(Color::DarkGray),
                        ),
                    ])
                })
                .collect::<Vec<_>>();
            frame.render_widget(Paragraph::new(lines), rows[1]);
        }

        frame.render_widget(
            Paragraph::new(state.status.as_deref().unwrap_or(&state.footer))
                .style(Style::default().fg(Color::DarkGray)),
            rows[2],
        );
        if let Some(picker) = &state.picker {
            frame.render_widget(
                Paragraph::new(format!("filter › {}", picker.query)),
                rows[3],
            );
        } else {
            let composer = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(2), Constraint::Min(1)])
                .split(rows[3]);
            frame.render_widget(Line::from("› "), composer[0]);
            frame.render_widget(&state.composer, composer[1]);
        }
    })?;
    Ok(())
}

fn commit_plain(terminal: &mut InlineTerminal, text: &str, style: Style) -> io::Result<()> {
    let text = Text::from(expand_tabs(text));
    let height = rendered_text_height(&text, terminal.size()?.width);
    let paragraph = Paragraph::new(text).style(style).wrap(Wrap { trim: false });
    commit_paragraph(terminal, paragraph, height)
}

fn expand_tabs(text: &str) -> String {
    terminal_safe_text(text).replace('\t', "  ")
}

fn commit_error(terminal: &mut InlineTerminal, text: &str) -> io::Result<()> {
    commit_plain(terminal, text, Style::default().fg(Color::Red))
}

fn commit_markdown(terminal: &mut InlineTerminal, markdown: &str) -> io::Result<()> {
    let safe_markdown = terminal_safe_text(markdown);
    let rendered = tui_markdown::from_str(&safe_markdown);
    let height = rendered_text_height(&rendered, terminal.size()?.width);
    let paragraph = Paragraph::new(rendered).wrap(Wrap { trim: false });
    commit_paragraph(terminal, paragraph, height)
}

fn terminal_safe_text(text: &str) -> String {
    text.chars()
        .map(|character| match character {
            '\n' | '\t' => character,
            character if character.is_control() => '�',
            character => character,
        })
        .collect()
}

fn commit_tool(terminal: &mut InlineTerminal, presentation: ToolPresentation) -> io::Result<()> {
    let headline_style = match presentation.outcome {
        PresentationOutcome::Success => Style::default().fg(Color::Green),
        PresentationOutcome::Error => Style::default().fg(Color::Red),
    };
    let mut lines = vec![Line::styled(presentation.headline, headline_style)];
    lines.extend(presentation.output.into_iter().map(|line| {
        let style = if line.starts_with('+') && !line.starts_with("+++") {
            Style::default().fg(Color::Green)
        } else if line.starts_with('-') && !line.starts_with("---") {
            Style::default().fg(Color::Red)
        } else if line.starts_with("@@") {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        Line::styled(format!("  {line}"), style)
    }));
    let text = Text::from(lines);
    let height = rendered_text_height(&text, terminal.size()?.width);
    commit_paragraph(
        terminal,
        Paragraph::new(text).wrap(Wrap { trim: false }),
        height,
    )
}

fn commit_paragraph(
    terminal: &mut InlineTerminal,
    paragraph: Paragraph<'_>,
    height: usize,
) -> io::Result<()> {
    let height = u16::try_from(height).unwrap_or(u16::MAX).max(1);
    terminal.insert_before(height, move |buffer| {
        paragraph.render(buffer.area, buffer);
    })?;
    Ok(())
}

fn rendered_text_height(text: &Text<'_>, width: u16) -> usize {
    let width = usize::from(width.max(1));
    text.lines
        .iter()
        .map(|line| line.width().div_ceil(width).max(1))
        .sum()
}

struct InteractiveState {
    assistant: String,
    status: Option<String>,
    footer: String,
    composer: TextArea<'static>,
    picker: Option<Picker>,
}

impl InteractiveState {
    fn new(footer: String) -> Self {
        let mut composer = TextArea::default();
        composer.set_cursor_line_style(Style::default());
        Self {
            assistant: String::new(),
            status: None,
            footer,
            composer,
            picker: None,
        }
    }

    fn prompt(&self) -> String {
        self.composer.lines().join("\n").trim().to_owned()
    }

    fn clear_prompt(&mut self) {
        self.set_prompt("");
    }

    fn set_prompt(&mut self, value: &str) {
        self.composer = TextArea::from([value]);
        self.composer.set_cursor_line_style(Style::default());
        self.composer.move_cursor(textarea::CursorMove::End);
    }
}

#[derive(Clone)]
enum PickerKind {
    Models {
        models: Vec<Model>,
        current: String,
    },
    Login {
        providers: Vec<ProviderChoice>,
        method: Option<LoginMethod>,
    },
    Logout(Vec<ProviderChoice>),
}

struct Picker {
    title: &'static str,
    query: String,
    selected: usize,
    kind: PickerKind,
}

impl Picker {
    fn models(models: Vec<Model>, query: String, current: &Model) -> Self {
        let current = model_identifier(current);
        let mut picker = Self {
            title: "models",
            query,
            selected: 0,
            kind: PickerKind::Models {
                models,
                current: current.clone(),
            },
        };
        if let Some(selected) = picker
            .entries()
            .iter()
            .position(|(_, identifier, _)| identifier == &current)
        {
            picker.selected = selected;
        }
        picker
    }

    fn login(providers: Vec<ProviderChoice>, method: Option<LoginMethod>) -> Self {
        Self {
            title: "login",
            query: String::new(),
            selected: 0,
            kind: PickerKind::Login { providers, method },
        }
    }

    fn logout(providers: Vec<ProviderChoice>) -> Self {
        Self {
            title: "logout",
            query: String::new(),
            selected: 0,
            kind: PickerKind::Logout(providers),
        }
    }

    fn entries(&self) -> Vec<(usize, String, String)> {
        let query = self.query.to_ascii_lowercase();
        match &self.kind {
            PickerKind::Models { models, current } => models
                .iter()
                .enumerate()
                .filter_map(|(index, model)| {
                    let identifier = model_identifier(model);
                    let search = format!("{identifier} {}", model.name).to_ascii_lowercase();
                    search.contains(&query).then(|| {
                        let reasoning = if model.reasoning {
                            "reasoning"
                        } else {
                            "standard"
                        };
                        let current = if &identifier == current {
                            " · current"
                        } else {
                            ""
                        };
                        (
                            index,
                            identifier,
                            format!("{} · {reasoning}{current}", model.name),
                        )
                    })
                })
                .collect(),
            PickerKind::Login { providers, method } => providers
                .iter()
                .enumerate()
                .filter_map(|(index, provider)| {
                    let search = format!("{} {}", provider.id, provider.name).to_ascii_lowercase();
                    (supports_login_method(provider, *method) && search.contains(&query)).then(
                        || {
                            let methods = match (provider.supports_api_key, provider.supports_oauth)
                            {
                                (true, true) => "API key or OAuth",
                                (true, false) => "API key",
                                (false, true) => "OAuth",
                                (false, false) => "no login",
                            };
                            (
                                index,
                                provider.id.clone(),
                                format!("{} · {methods}", provider.name),
                            )
                        },
                    )
                })
                .collect(),
            PickerKind::Logout(providers) => providers
                .iter()
                .enumerate()
                .filter_map(|(index, provider)| {
                    let search = format!("{} {}", provider.id, provider.name).to_ascii_lowercase();
                    search
                        .contains(&query)
                        .then(|| (index, provider.id.clone(), provider.name.clone()))
                })
                .collect(),
        }
    }

    fn rendered_height(&self) -> u16 {
        let entries = self.entries();
        let rows = entries.len().min(PICKER_VISIBLE_ITEMS) + if entries.is_empty() { 2 } else { 1 };
        u16::try_from(rows).unwrap_or(u16::MAX)
    }

    fn render_lines(&self) -> Vec<Line<'static>> {
        let entries = self.entries();
        let selected = self.selected.min(entries.len().saturating_sub(1));
        let start = selected
            .saturating_add(1)
            .saturating_sub(PICKER_VISIBLE_ITEMS);
        let mut lines = vec![Line::from(vec![
            Span::styled(
                self.title,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "  type to filter · ↑↓ enter esc",
                Style::default().fg(Color::DarkGray),
            ),
        ])];
        if entries.is_empty() {
            lines.push(Line::styled(
                "  no matches",
                Style::default().fg(Color::DarkGray),
            ));
            return lines;
        }
        for (visible_index, (_, label, detail)) in entries
            .iter()
            .enumerate()
            .skip(start)
            .take(PICKER_VISIBLE_ITEMS)
        {
            let selected = visible_index == selected;
            let marker = if selected { "› " } else { "  " };
            let label_style = if selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(vec![
                Span::styled(marker, label_style),
                Span::styled(label.clone(), label_style),
                Span::styled(format!("  {detail}"), Style::default().fg(Color::DarkGray)),
            ]));
        }
        lines
    }

    fn move_selection(&mut self, delta: i32) {
        let count = self.entries().len();
        if count == 0 {
            self.selected = 0;
            return;
        }
        self.selected = if delta < 0 {
            self.selected.checked_sub(1).unwrap_or(count - 1)
        } else {
            (self.selected + 1) % count
        };
    }

    fn selection(&self) -> Option<PickerSelection> {
        let entries = self.entries();
        let source_index = entries.get(self.selected)?.0;
        match &self.kind {
            PickerKind::Models { models, .. } => models
                .get(source_index)
                .cloned()
                .map(Box::new)
                .map(PickerSelection::Model),
            PickerKind::Login { providers, method } => {
                providers
                    .get(source_index)
                    .map(|provider| PickerSelection::Login {
                        provider: provider.id.clone(),
                        method: *method,
                    })
            }
            PickerKind::Logout(providers) => {
                providers
                    .get(source_index)
                    .map(|provider| PickerSelection::Logout {
                        provider: provider.id.clone(),
                    })
            }
        }
    }
}

enum PickerSelection {
    Model(Box<Model>),
    Login {
        provider: String,
        method: Option<LoginMethod>,
    },
    Logout {
        provider: String,
    },
}

fn handle_picker_key(state: &mut InteractiveState, key: KeyEvent) -> Option<PickerSelection> {
    if is_ctrl_c(key) {
        state.picker = None;
        return None;
    }
    let picker = state.picker.as_mut()?;
    match key.code {
        KeyCode::Esc => state.picker = None,
        KeyCode::Up => picker.move_selection(-1),
        KeyCode::Down => picker.move_selection(1),
        KeyCode::Enter => return picker.selection(),
        KeyCode::Backspace => {
            picker.query.pop();
            picker.selected = 0;
        }
        KeyCode::Char(character) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            picker.query.push(character);
            picker.selected = 0;
        }
        _ => {}
    }
    None
}

struct TerminalMode {
    active: bool,
}

impl TerminalMode {
    fn enter() -> io::Result<Self> {
        enable_raw_mode()?;
        if let Err(error) = write_terminal_commands(&mut io::stdout(), true) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        Ok(Self { active: true })
    }

    fn suspend(&mut self) -> io::Result<()> {
        if !self.active {
            return Ok(());
        }
        write_terminal_commands(&mut io::stdout(), false)?;
        disable_raw_mode()?;
        self.active = false;
        Ok(())
    }

    fn resume(&mut self) -> io::Result<()> {
        if self.active {
            return Ok(());
        }
        enable_raw_mode()?;
        if let Err(error) = write_terminal_commands(&mut io::stdout(), true) {
            let _ = disable_raw_mode();
            return Err(error);
        }
        self.active = true;
        Ok(())
    }
}

impl Drop for TerminalMode {
    fn drop(&mut self) {
        if self.active {
            let _ = write_terminal_commands(&mut io::stdout(), false);
            let _ = disable_raw_mode();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ds_ai::builtin_openai_model;

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
            status: None,
            footer: String::new(),
            composer: TextArea::from(["  hello", "world  "]),
            picker: None,
        };

        assert_eq!(state.prompt(), "hello\nworld");
    }

    #[test]
    fn model_picker_filters_provider_id_and_display_name() {
        let model = builtin_openai_model("gpt-5.6-luna").unwrap().into_model();
        let current = model.clone();
        let mut picker = Picker::models(vec![model], "luna".into(), &current);
        assert_eq!(picker.entries().len(), 1);

        picker.query = "anthropic".into();
        assert!(picker.entries().is_empty());
    }

    #[test]
    fn login_picker_only_lists_providers_supporting_the_requested_method() {
        let providers = vec![
            ProviderChoice {
                id: "api".into(),
                name: "API only".into(),
                supports_api_key: true,
                supports_oauth: false,
            },
            ProviderChoice {
                id: "oauth".into(),
                name: "OAuth only".into(),
                supports_api_key: false,
                supports_oauth: true,
            },
        ];

        let picker = Picker::login(providers, Some(LoginMethod::OAuth));
        let entries = picker.entries();

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, "oauth");
    }

    #[test]
    fn control_c_cancels_an_open_picker() {
        let model = builtin_openai_model("gpt-5.6-luna").unwrap().into_model();
        let mut state = InteractiveState::new(String::new());
        state.picker = Some(Picker::models(vec![model.clone()], String::new(), &model));

        let selection = handle_picker_key(
            &mut state,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        );

        assert!(selection.is_none());
        assert!(state.picker.is_none());
    }

    #[test]
    fn partial_error_responses_keep_the_error_visible() {
        assert_eq!(
            assistant_error(StopReason::Error, Some("request failed")),
            Some("request failed")
        );
        assert_eq!(
            assistant_error(StopReason::Aborted, Some("cancelled")),
            None
        );
    }

    #[test]
    fn plain_transcript_text_expands_tabs() {
        assert_eq!(
            expand_tabs("openai\tnot\u{1b}[31m configured"),
            "openai  not�[31m configured"
        );
    }
}
