use clap::Parser;
use ds_agent_core::{Agent, AgentModelStream};
use ds_ai::builtin_models;
use ds_coding_agent::{coding_tools, ui};
use std::{error::Error, sync::Arc};

const SYSTEM_PROMPT: &str = "You are a concise coding agent. Use read, bash, edit, and write to inspect and change the current project. Explain completed work and verification plainly.";

#[derive(Parser)]
#[command(name = "ds", version, about = "A minimal Pi-style coding agent")]
struct Cli {
    #[arg(long, value_name = "PROVIDER/MODEL")]
    model: String,
    #[arg(value_name = "PROMPT", num_args = 1.., trailing_var_arg = true)]
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("ds: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let (provider, model_id) = parse_model(&cli.model)?;
    let models = builtin_models();
    let model = models
        .model(provider, model_id)
        .ok_or_else(|| format!("unknown model: {}", cli.model))?;
    let model_stream: Arc<dyn AgentModelStream> = Arc::new(models);
    let working_directory = std::env::current_dir()?;
    let mut agent = Agent::new(model, model_stream, coding_tools()?, working_directory)
        .with_system_prompt(SYSTEM_PROMPT);

    if cli.prompt.is_empty() {
        ui::run_interactive(&mut agent).await?;
    } else {
        ui::run_one_shot(&mut agent, cli.prompt.join(" ")).await?;
    }
    Ok(())
}

fn parse_model(value: &str) -> Result<(&str, &str), Box<dyn Error>> {
    let Some((provider, model)) = value.split_once('/') else {
        return Err(format!("model must use provider/model form: {value}").into());
    };
    if provider.is_empty() || model.is_empty() {
        return Err(format!("model must use provider/model form: {value}").into());
    }
    Ok((provider, model))
}
