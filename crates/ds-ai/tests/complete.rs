use crate::support::{Reply, serve};
use ds_ai::{
    AssistantContent, Context, Message, OpenAiResponsesOptions, StopReason, StreamOptions,
    TextContent, builtin_model,
};

#[tokio::test]
async fn completes_a_provider_stream_from_its_result() {
    let server = serve([Reply::sse(
        [
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_complete\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Done\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_complete\",\"usage\":{}}}\n\n",
        ]
        .concat(),
    )])
    .await;
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.base_url = server.base_url.clone();
    let mut stream = ds_ai::openai::stream(
        &model.typed::<ds_ai::OpenAiResponsesOptions>().unwrap(),
        &Context::new([Message::user("Complete")]),
        &OpenAiResponsesOptions {
            stream: StreamOptions {
                api_key: Some("test-key".into()),
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let response = stream.result().await.unwrap();

    assert_eq!(response.response_id.as_deref(), Some("resp_complete"));
    assert_eq!(
        response.content,
        [AssistantContent::Text(TextContent {
            text: "Done".into(),
            text_signature: None,
        })]
    );
    assert_eq!(response.stop_reason, StopReason::Stop);
    server.requests().await;
}
