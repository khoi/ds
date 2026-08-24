use crate::support::{Reply, serve};
use ds_ai::{Content, Context, Message, StopReason, complete, openai};

#[tokio::test]
async fn completes_a_provider_stream() {
    let server = serve([Reply::sse(
        [
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_complete\",\"type\":\"message\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Done\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_complete\",\"usage\":{}}}\n\n",
        ]
        .concat(),
    )])
    .await;
    let stream = openai::raw_stream(
        &openai::Model::new("gpt-test").with_base_url(&server.base_url),
        &Context::new([Message::user("Complete")]),
        &openai::Options::new("test-key"),
    )
    .await
    .unwrap();

    let response = complete(stream).await.unwrap();

    assert_eq!(response.id.as_deref(), Some("resp_complete"));
    assert_eq!(response.content, [Content::Text("Done".into())]);
    assert_eq!(response.stop_reason, StopReason::Stop);
    server.requests().await;
}
