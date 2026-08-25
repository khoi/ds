use ds_ai::{
    Context, Message, OpenAiResponsesOptions, StopReason, builtin_models, builtin_openai_model,
    content_text,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let models = builtin_models();
    let model = builtin_openai_model("gpt-5.6-luna").expect("model in built-in catalog");
    let context = Context::new([Message::user("Explain this repository in one sentence")]);
    let response = models
        .complete(&model, &context, &OpenAiResponsesOptions::default())
        .await?;

    if matches!(
        response.stop_reason,
        StopReason::Error | StopReason::Aborted
    ) {
        return Err(std::io::Error::other(
            response
                .error_message
                .unwrap_or_else(|| "request failed".into()),
        )
        .into());
    }

    println!("{}", content_text(&response.content));
    Ok(())
}
