#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LoginMethod {
    ApiKey,
    OAuth,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SlashCommand {
    Help,
    Models {
        query: String,
    },
    Login {
        provider: Option<String>,
        method: Option<LoginMethod>,
    },
    Logout {
        provider: Option<String>,
    },
    Auth {
        provider: Option<String>,
    },
    Status,
    Clear,
    Quit,
    Unknown {
        command: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandSpec {
    pub command: &'static str,
    pub usage: &'static str,
    pub description: &'static str,
}

pub const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        command: "/help",
        usage: "/help",
        description: "show local commands",
    },
    CommandSpec {
        command: "/models",
        usage: "/models [query]",
        description: "browse and switch models",
    },
    CommandSpec {
        command: "/model",
        usage: "/model [provider/model or query]",
        description: "browse and switch models",
    },
    CommandSpec {
        command: "/login",
        usage: "/login [provider] [api-key|oauth]",
        description: "sign in to a provider",
    },
    CommandSpec {
        command: "/logout",
        usage: "/logout [provider]",
        description: "remove a saved provider login",
    },
    CommandSpec {
        command: "/auth",
        usage: "/auth [provider]",
        description: "show provider login status",
    },
    CommandSpec {
        command: "/status",
        usage: "/status",
        description: "show model and working directory",
    },
    CommandSpec {
        command: "/clear",
        usage: "/clear",
        description: "clear the visible terminal",
    },
    CommandSpec {
        command: "/quit",
        usage: "/quit",
        description: "exit ds",
    },
];

pub fn parse(input: &str) -> Option<SlashCommand> {
    let input = input.trim();
    if !input.starts_with('/') {
        return None;
    }
    let mut tokens = input.split_whitespace();
    let command = tokens.next().unwrap_or_default();
    let arguments = tokens.collect::<Vec<_>>();
    Some(match command {
        "/help" => SlashCommand::Help,
        "/model" | "/models" => SlashCommand::Models {
            query: arguments.join(" "),
        },
        "/login" => {
            let (provider, method) = parse_login_arguments(&arguments);
            SlashCommand::Login { provider, method }
        }
        "/logout" => SlashCommand::Logout {
            provider: arguments.first().map(|value| (*value).to_owned()),
        },
        "/auth" => SlashCommand::Auth {
            provider: arguments.first().map(|value| (*value).to_owned()),
        },
        "/status" => SlashCommand::Status,
        "/clear" => SlashCommand::Clear,
        "/quit" | "/exit" => SlashCommand::Quit,
        command => SlashCommand::Unknown {
            command: command.to_owned(),
        },
    })
}

fn parse_login_arguments(arguments: &[&str]) -> (Option<String>, Option<LoginMethod>) {
    let mut provider = None;
    let mut method = None;
    for argument in arguments {
        match *argument {
            "api-key" => method = Some(LoginMethod::ApiKey),
            "oauth" => method = Some(LoginMethod::OAuth),
            value if provider.is_none() => provider = Some(value.to_owned()),
            _ => {}
        }
    }
    (provider, method)
}

pub fn suggestions(input: &str) -> Vec<CommandSpec> {
    let query = input.trim();
    if !query.starts_with('/') || query.contains(char::is_whitespace) {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .copied()
        .filter(|spec| spec.command.starts_with(query))
        .collect()
}

pub fn help_text() -> String {
    let width = COMMANDS
        .iter()
        .map(|spec| spec.usage.len())
        .max()
        .unwrap_or_default();
    COMMANDS
        .iter()
        .map(|spec| format!("{:<width$}  {}", spec.usage, spec.description))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_prompts_are_not_commands() {
        assert_eq!(parse("explain /models"), None);
    }

    #[test]
    fn model_aliases_share_the_picker() {
        assert_eq!(
            parse("/model codex luna"),
            Some(SlashCommand::Models {
                query: "codex luna".into()
            })
        );
        assert_eq!(
            parse("/models"),
            Some(SlashCommand::Models {
                query: String::new()
            })
        );
    }

    #[test]
    fn login_accepts_provider_and_method_in_either_order() {
        let expected = Some(SlashCommand::Login {
            provider: Some("openai-codex".into()),
            method: Some(LoginMethod::OAuth),
        });
        assert_eq!(parse("/login openai-codex oauth"), expected);
        assert_eq!(parse("/login oauth openai-codex"), expected);
    }

    #[test]
    fn suggestions_only_match_the_command_token() {
        assert_eq!(
            suggestions("/mod")
                .iter()
                .map(|spec| spec.command)
                .collect::<Vec<_>>(),
            ["/models", "/model"]
        );
        assert!(suggestions("/model codex").is_empty());
        assert!(suggestions("hello").is_empty());
    }
}
