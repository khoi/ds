use crate::support::{Reply, serve};
use ds_ai::{CacheRetention, Context, Event, InputContent, Message, ToolResult, anthropic, openai};
use futures_util::StreamExt;
use serde_json::{Value, json};

#[tokio::test]
async fn normalizes_a_cross_provider_tool_transcript() {
    let source_sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Need tools\"}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Running\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Running\",\"annotations\":[]}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1|fc_1\",\"name\":\"read\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":2,\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":2,\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_1|fc_1\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":3,\"item\":{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2|fc_2\",\"name\":\"shell\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":3,\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":3,\"item\":{\"id\":\"fc_2\",\"type\":\"function_call\",\"call_id\":\"call_2|fc_2\",\"name\":\"shell\",\"arguments\":\"{\\\"command\\\":\\\"pwd\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_source\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let source_server = serve([Reply::sse(source_sse)]).await;
    let source_model = openai::Model::new("gpt-5.6").with_base_url(&source_server.base_url);
    let source_events = openai::stream(
        &source_model,
        &Context::new([Message::user("Run")]),
        &openai::Options::new("test-key"),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;
    let source_response = done(&source_events).clone();
    source_server.requests().await;

    let target_sse = [
        "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    ]
    .concat();
    let target_server = serve([Reply::sse(target_sse)]).await;
    let target_model =
        anthropic::Model::new("claude-sonnet-4-5").with_base_url(&target_server.base_url);
    let target_context = Context::new([
        Message::user("Run"),
        Message::assistant(source_response),
        Message::tool_result(ToolResult::new(
            "call_1|fc_1",
            "read",
            [InputContent::text("done")],
        )),
    ]);

    anthropic::stream(
        &target_model,
        &target_context,
        &anthropic::Options::new("test-key").with_cache_retention(CacheRetention::None),
    )
    .await
    .unwrap()
    .collect::<Vec<_>>()
    .await;

    let request = target_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["messages"],
        json!([
            {
                "role": "user",
                "content": [{"type": "text", "text": "Run"}]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Need tools"},
                    {"type": "text", "text": "Running"},
                    {
                        "type": "tool_use",
                        "id": "call_1_fc_1",
                        "name": "read",
                        "input": {"path": "README.md"}
                    },
                    {
                        "type": "tool_use",
                        "id": "call_2_fc_2",
                        "name": "shell",
                        "input": {"command": "pwd"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1_fc_1",
                        "content": [{"type": "text", "text": "done"}],
                        "is_error": false
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_2_fc_2",
                        "content": [{"type": "text", "text": "No result provided"}],
                        "is_error": true
                    }
                ]
            }
        ])
    );
}

fn done(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Ok(Event::Done(response))) => response,
        _ => panic!("stream did not complete"),
    }
}
