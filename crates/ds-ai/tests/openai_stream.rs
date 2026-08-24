use crate::support::{Reply, serve};
use ds_ai::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantToolCall, CacheRetention,
    Context, InputContent, Message, OpenAiResponsesOptions, ResponseHook, StopReason,
    StreamOptions, TextContent, ThinkingContent, Tool, ToolResultMessage, builtin_model, openai,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[tokio::test]
async fn streams_openai_text_until_the_provider_completes() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"service_tier\":\"flex\",\"output\":[],\"usage\":{\"input_tokens\":4,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":5}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Hello"
    )));
    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_1"));
    assert_eq!(
        response.content,
        [text("Hello", Some(("msg_1", Some("final_answer"))))]
    );
    assert_eq!(response.usage.input, 4);
    assert_eq!(response.usage.output, 1);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.reasoning, Some(0));
    assert_eq!(response.usage.total_tokens, 5);

    let request = server.requests().await.pop().unwrap();
    assert!(request.starts_with("POST /responses HTTP/1.1\r\n"));
    assert!(request.contains("authorization: Bearer test-key\r\n"));
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "gpt-5.6",
            "input": [{
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}]
            }],
            "stream": true,
            "store": false,
            "reasoning": {"effort": "none"}
        })
    );
}

#[tokio::test]
async fn rejects_an_openai_stream_that_ends_without_a_terminal_event() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_partial\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_partial\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Part\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Part"
    )));
    let partial = failed(&events);
    assert_eq!(partial.response_id.as_deref(), Some("resp_partial"));
    assert_eq!(partial.content, [text("Part", None)]);
    assert_eq!(partial.usage, ds_ai::Usage::default());
    assert_eq!(
        partial.error_message.as_deref(),
        Some("provider stream ended before a terminal event")
    );
    server.requests().await;
}

#[tokio::test]
async fn decodes_openai_sse_across_arbitrary_chunks() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_chunks\"}}\r\n\r\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_chunks\",\"type\":\"message\",\"content\":[]}}\r\n\r\n",
        "event: response.output_text.delta\r\n",
        "data: {\"type\":\"response.output_text.delta\",\r\n",
        "data: \"output_index\":0,\"content_index\":0,\"delta\":\"Hé\"}\r\n\r\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_chunks\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hé\"}]}}\r\n\r\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_chunks\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\r\n\r\n",
    );
    let accent = sse.find('é').unwrap();
    let split_points = [1, 7, 79, accent + 1, accent + 2, sse.len() - 1];
    let mut start = 0;
    let mut chunks = Vec::new();
    for end in split_points {
        chunks.push(sse.as_bytes()[start..end].to_vec());
        start = end;
    }
    chunks.push(sse.as_bytes()[start..].to_vec());
    let server = serve([Reply::sse_chunks(chunks)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Hé"
    )));
    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_chunks"));
    assert_eq!(response.content, [text("Hé", Some(("msg_chunks", None)))]);
    assert_eq!(response.usage.input, 3);
    assert_eq!(response.usage.output, 1);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.reasoning, Some(0));
    assert_eq!(response.usage.total_tokens, 0);
    server.requests().await;
}

#[tokio::test]
async fn retries_openai_before_streaming_starts() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_retry\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after-ms", "0"),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));

    let events = events(&model, &context, &options).await;

    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_retry"));
    assert!(response.content.is_empty());
    assert_eq!(
        response.usage,
        ds_ai::Usage {
            reasoning: Some(0),
            ..Default::default()
        }
    );
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn waits_for_openai_retry_headers() {
    for (header, value, delay_ms) in [("retry-after-ms", "1500", 1500), ("retry-after", "2", 2000)]
    {
        let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_delayed\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
        let server = serve([
            Reply::json(
                429,
                json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
            )
            .with_header(header, value),
            Reply::sse(completed),
        ])
        .await;
        let model = model(&server.base_url);
        let context = Context::new([Message::user("Hello")]);
        let options = options(|options| options.stream.max_retries = Some(1));
        let task = tokio::spawn(async move { events(&model, &context, &options).await });

        server.wait_for_requests(1).await;
        tokio::time::advance(std::time::Duration::from_millis(delay_ms - 1)).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(server.request_count(), 1);

        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        let events = task.await.unwrap();
        done(&events);
        assert_eq!(server.requests().await.len(), 2);
    }
}

#[tokio::test(start_paused = true)]
async fn waits_for_an_openai_retry_after_http_date() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_date\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let retry_at =
        httpdate::fmt_http_date(std::time::SystemTime::now() + std::time::Duration::from_secs(60));
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after", retry_at),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    tokio::time::advance(std::time::Duration::from_secs(58)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_secs(3)).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn cancels_an_openai_retry_wait() {
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after-ms", "60000"),
        Reply::sse(Vec::new()),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| {
        options.stream.max_retries = Some(1);
        options.stream.cancellation = cancellation.clone();
    });
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    cancellation.cancel();
    let events = task.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("request cancelled"));
    assert_eq!(server.request_count(), 1);
}

#[tokio::test]
async fn follows_openai_retry_status_and_override_headers() {
    let cases = [
        (408, None, true),
        (409, None, true),
        (429, None, true),
        (500, None, true),
        (599, None, true),
        (400, None, false),
        (400, Some(("x-should-retry", "true")), true),
        (500, Some(("x-should-retry", "false")), false),
    ];

    for (status, header, should_retry) in cases {
        let mut failure = Reply::json(status, json!({"error": {"message": "failed"}}));
        if let Some((name, value)) = header {
            failure = failure.with_header(name, value);
        }
        if should_retry {
            failure = failure.with_header("retry-after-ms", "0");
        }
        let mut replies = vec![failure];
        if should_retry {
            replies.push(Reply::sse(format!(
                "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"resp_{status}\",\"usage\":{{\"input_tokens\":0,\"input_tokens_details\":{{}},\"output_tokens\":0,\"output_tokens_details\":{{}}}}}}}}\n\n"
            )));
        }
        let server = serve(replies).await;
        let model = model(&server.base_url);
        let context = Context::new([Message::user("Hello")]);
        let options = options(|options| options.stream.max_retries = Some(1));

        let events = events(&model, &context, &options).await;

        if should_retry {
            done(&events);
            assert_eq!(server.requests().await.len(), 2);
        } else {
            assert_eq!(failed(&events).stop_reason, StopReason::Error);
            assert!(
                failed(&events).error_message.as_deref().is_some_and(
                    |message| message.starts_with(&format!("provider returned HTTP {status}:"))
                )
            );
            assert_eq!(server.request_count(), 1);
        }
    }
}

#[tokio::test(start_paused = true)]
async fn retries_openai_network_failures_before_streaming_starts() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_network\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::disconnect(), Reply::sse(completed)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));

    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    tokio::time::advance(std::time::Duration::from_millis(374)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(126)).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn rejects_an_openai_retry_delay_above_a_custom_limit() {
    let server = serve([Reply::json(
        429,
        json!({"error": {"type": "rate_limit_error", "message": "retry later"}}),
    )
    .with_header("retry-after", "2")])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.max_retries = Some(1);
        options.stream.max_retry_delay = Some(std::time::Duration::from_secs(1));
    });

    let events = events(&model, &context, &options).await;
    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some("provider retry delay 2s exceeds 1s")
    );
    assert_eq!(server.requests().await.len(), 1);
}

#[tokio::test(start_paused = true)]
async fn backs_off_before_an_openai_retry_without_a_header() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_backoff\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        ),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    tokio::time::advance(std::time::Duration::from_millis(374)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(126)).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test(start_paused = true)]
async fn accepts_fractional_openai_retry_headers() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fractional\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([
        Reply::json(
            429,
            json!({"error": {"type": "rate_limit_error", "message": "retry"}}),
        )
        .with_header("retry-after", "1.5"),
        Reply::sse(completed),
    ])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| options.stream.max_retries = Some(1));
    let task = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    tokio::time::advance(std::time::Duration::from_millis(1499)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(1)).await;
    let events = task.await.unwrap();
    done(&events);
    assert_eq!(server.requests().await.len(), 2);
}

#[tokio::test]
async fn streams_openai_reasoning_and_text_in_content_order() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_reasoning\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"summary_index\":0,\"delta\":\"Need answer.\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}],\"encrypted_content\":\"encrypted\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_reasoning\",\"usage\":{\"input_tokens\":5,\"input_tokens_details\":{},\"output_tokens\":4,\"output_tokens_details\":{\"reasoning_tokens\":3}}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});

    let events = events(&model, &context, &options).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Need answer."
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 1,
            delta,
            ..
        } if delta == "Hello"
    )));
    let response = done(&events);
    assert_eq!(response.response_id.as_deref(), Some("resp_reasoning"));
    assert_eq!(
        response.content,
        [
            thinking("Need answer.", Some("rs_1"), Some("encrypted")),
            text("Hello", Some(("msg_1", Some("final_answer")))),
        ]
    );
    assert_eq!(response.usage.input, 5);
    assert_eq!(response.usage.output, 4);
    assert_eq!(response.usage.cache_read, 0);
    assert_eq!(response.usage.reasoning, Some(3));
    assert_eq!(response.usage.total_tokens, 0);
}

#[tokio::test]
async fn streams_openai_reasoning_text_and_refusal_content() {
    let sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_text\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_text.delta\",\"output_index\":0,\"delta\":\"Private\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_text\",\"type\":\"reasoning\",\"summary\":[],\"content\":[{\"type\":\"reasoning_text\",\"text\":\"Private\"}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_refusal\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.refusal.delta\",\"output_index\":1,\"delta\":\"Denied\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_refusal\",\"type\":\"message\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"Denied\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_variants\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);

    let context = Context::new([Message::user("Answer")]);
    let events = events(&model, &context, &options(|_| {})).await;

    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "Private"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta {
            content_index: 1,
            delta,
            ..
        } if delta == "Denied"
    )));
    assert_eq!(
        done(&events).content,
        [
            thinking("Private", Some("rs_text"), None),
            text("Denied", Some(("msg_refusal", None))),
        ]
    );
    server.requests().await;
}

#[tokio::test]
async fn replays_serialized_openai_reasoning_and_message_items() {
    let first_sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_first\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Need answer.\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_replay\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_replay\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_first\",\"output\":[{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}],\"encrypted_content\":\"encrypted\"}],\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":2,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = model(&first_server.base_url);
    let first_context = Context::new([Message::user("Hello")]);
    let options = options(|_| {});
    let first_events = events(&first_model, &first_context, &options).await;
    let response: AssistantMessage =
        serde_json::from_value(serde_json::to_value(done(&first_events)).unwrap()).unwrap();
    first_server.requests().await;

    let second_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = model(&second_server.base_url);
    let second_context = Context::new([Message::assistant(response), Message::user("Continue")]);

    let second_events = events(&second_model, &second_context, &options).await;
    done(&second_events);

    let request = second_server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "type": "reasoning",
                "id": "rs_replay",
                "summary": [{"type": "summary_text", "text": "Need answer."}],
                "encrypted_content": "encrypted"
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                "status": "completed",
                "id": "msg_replay",
                "phase": "final_answer"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn generates_distinct_replay_ids_for_text_without_provider_ids() {
    let first_sse = [
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"First\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"First\"}]}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Second\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Second\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_without_ids\",\"usage\":{}}}\n\n",
    ]
    .concat();
    let done_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_next\",\"usage\":{}}}\n\n";
    let server = serve([Reply::sse(first_sse), Reply::sse(done_sse)]).await;
    let model = model(&server.base_url);
    let options = options(|_| {});
    let first = events(&model, &Context::new([Message::user("Split")]), &options).await;
    let replay = done(&first).clone();

    let second = events(
        &model,
        &Context::new([
            Message::user("Split"),
            Message::assistant(replay),
            Message::user("Continue"),
        ]),
        &options,
    )
    .await;
    done(&second);

    let requests = server.requests().await;
    let body: Value = serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    let messages = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "message" && item["role"] == "assistant")
        .collect::<Vec<_>>();
    assert_eq!(messages.len(), 2);
    assert_ne!(messages[0]["id"], messages[1]["id"]);
    assert!(
        messages
            .iter()
            .all(|message| message["id"].as_str().unwrap().starts_with("msg_ds_"))
    );
}

#[tokio::test]
async fn streams_and_replays_openai_tool_calls() {
    let first_sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_edit\",\"type\":\"function_call\",\"call_id\":\"call_edit\",\"name\":\"edit\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"path\\\":\\\"README.md\\\"\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\",\\\"content\\\":\\\"updated\\\"}\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"path\\\":\\\"README.md\\\",\\\"content\\\":\\\"updated\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_edit\",\"type\":\"function_call\",\"call_id\":\"call_edit\",\"name\":\"edit\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\",\\\"content\\\":\\\"updated\\\"}\",\"namespace\":\"dynamic_tools\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":2,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = model(&first_server.base_url);
    let first_context = Context::new([Message::user("Edit the file")]).with_tools([Tool::new(
        "edit",
        "Edit a file",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
    )
    .with_strict()]);
    let options = options(|_| {});

    let first_events = events(&first_model, &first_context, &options).await;

    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta,
            ..
        } if delta == "{\"path\":\"README.md\""
    )));
    assert!(first_events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ToolCallDelta {
            content_index: 0,
            delta,
            ..
        } if delta == ",\"content\":\"updated\"}"
    )));
    let response = done(&first_events);
    assert_eq!(
        response.content,
        [AssistantContent::ToolCall(AssistantToolCall {
            id: "call_edit|fc_edit".into(),
            name: "edit".into(),
            arguments: json!({"path": "README.md", "content": "updated"}),
            thought_signature: None,
            namespace: Some("dynamic_tools".into()),
        })]
    );

    let first_request = first_server.requests().await.pop().unwrap();
    let first_body: Value =
        serde_json::from_str(first_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        first_body["tools"],
        json!([{
            "type": "function",
            "name": "edit",
            "description": "Edit a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"],
                "additionalProperties": false
            },
            "strict": true
        }])
    );

    let restored: AssistantMessage =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    let second_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = model(&second_server.base_url);
    let second_context = Context::new([Message::assistant(restored), Message::user("Continue")]);

    let second_events = events(&second_model, &second_context, &options).await;
    done(&second_events);

    let second_request = second_server.requests().await.pop().unwrap();
    let second_body: Value =
        serde_json::from_str(second_request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        second_body["input"],
        json!([
            {
                "type": "function_call",
                "id": "fc_edit",
                "call_id": "call_edit",
                "name": "edit",
                "arguments": "{\"content\":\"updated\",\"path\":\"README.md\"}",
                "namespace": "dynamic_tools"
            },
            {
                "type": "function_call_output",
                "call_id": "call_edit",
                "output": "No result provided"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn uses_the_provider_terminal_token_total() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_total\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":99}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;

    assert_eq!(done(&events).usage.total_tokens, 99);
    server.requests().await;
}

#[tokio::test]
async fn rejects_an_incomplete_response_without_a_reason() {
    let sse = "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete\",\"status\":\"incomplete\",\"incomplete_details\":null}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let events = events(
        &model,
        &Context::new([Message::user("Hello")]),
        &options(|_| {}),
    )
    .await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider response failed: Response incomplete without a provider reason")
    );
    assert_eq!(error.raw_stop_reason.as_deref(), Some("incomplete"));
    server.requests().await;
}

#[tokio::test]
async fn rejects_invalid_strict_tool_schemas_before_connecting() {
    let model = model("http://127.0.0.1:9");
    let context = Context::new([Message::user("Look up")]).with_tools([Tool::new(
        "lookup",
        "Look up a value",
        json!({
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"type": "string"}}
        }),
    )
    .with_strict()]);

    let events = events(&model, &context, &options(|_| {})).await;

    assert_eq!(
        failed(&events).error_message.as_deref(),
        Some(
            "invalid request: tool \"lookup\" requires JSON-schema constrained sampling, but $defs schemas are unsupported"
        )
    );
}

#[tokio::test]
async fn sends_openai_tool_result_text_images_and_empty_output() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_result\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([
        Message::user("Inspect the image"),
        Message::tool_result(ToolResultMessage::new(
            "call_image",
            "inspect",
            [
                InputContent::text("A red circle"),
                InputContent::image("image/png", "iVBORw0KGgo="),
            ],
        )),
        Message::tool_result(ToolResultMessage::new(
            "call_empty",
            "noop",
            [InputContent::text("")],
        )),
    ]);
    let events = events(&model, &context, &options(|_| {})).await;
    done(&events);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Inspect the image"}]
            },
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": "A red circle"},
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/png;base64,iVBORw0KGgo="
                    }
                ]
            },
            {
                "type": "function_call_output",
                "call_id": "call_empty",
                "output": "(no tool output)"
            }
        ])
    );
}

#[tokio::test]
async fn finalizes_an_incomplete_openai_response_as_a_length_stop() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_incomplete\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_incomplete\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Partial\"}\n\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":30,\"input_tokens_details\":{\"cached_tokens\":5},\"output_tokens\":12,\"output_tokens_details\":{},\"total_tokens\":42}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let response = done(&events);
    assert_eq!(response.stop_reason, StopReason::Length);
    assert_eq!(
        response.raw_stop_reason.as_deref(),
        Some("incomplete.max_output_tokens")
    );
    assert_eq!(response.content, [text("Partial", None)]);
    assert_eq!(response.usage.input, 25);
    assert_eq!(response.usage.cache_read, 5);
    assert_eq!(response.usage.output, 12);
}

#[tokio::test]
async fn rejects_a_content_filtered_openai_response_with_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_filtered\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_filtered\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_filtered\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"content_filter\"},\"usage\":{\"input_tokens\":2,\"input_tokens_details\":{},\"output_tokens\":1,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider response failed: Response incomplete: content_filter")
    );
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.raw_stop_reason.as_deref(),
        Some("incomplete.content_filter")
    );
    assert_eq!(error.content, [text("Visible", None)]);
}

#[tokio::test]
async fn rejects_a_failed_openai_response_with_code_and_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_failed\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_failed\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Partial thought\"}\n\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"boom\"}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Think")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider response failed: boom")
    );
    assert_eq!(error.response_id.as_deref(), Some("resp_failed"));
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("failed"));
    assert_eq!(error.content, [thinking("Partial thought", None, None)]);
}

#[tokio::test]
async fn rejects_an_openai_error_event_with_code_and_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_error\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_error\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
        "data: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\",\"param\":null}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let events = events(&model, &context, &options(|_| {})).await;

    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider response failed: slow down")
    );
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(error.raw_stop_reason.as_deref(), Some("error"));
    assert_eq!(error.content, [text("Visible", None)]);
}

#[tokio::test]
async fn encodes_openai_multimodal_context_and_generation_options() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_options\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user_content([
        InputContent::text("Describe this"),
        InputContent::image("image/png", "iVBORw0KGgo="),
    ])])
    .with_system("Be brief")
    .with_tools([Tool::new(
        "inspect",
        "Inspect an image",
        json!({"type": "object", "properties": {}, "additionalProperties": false}),
    )]);
    let options = options(|options| {
        options.stream.max_tokens = Some(128);
        options.stream.temperature = Some(0.2);
        options.reasoning_effort = Some(openai::ReasoningEffort::High);
        options.reasoning_summary = Some(openai::ReasoningSummary::Concise);
        options.tool_choice = Some(openai::ToolChoice::Required);
        options.service_tier = Some(openai::ServiceTier::Priority);
    });

    let events = events(&model, &context, &options).await;
    done(&events);

    let request = server.requests().await.pop().unwrap();
    let body: Value = serde_json::from_str(request.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(
        body,
        json!({
            "model": "gpt-5.6",
            "input": [
                {
                    "role": "developer",
                    "content": "Be brief"
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this"},
                        {
                            "type": "input_image",
                            "detail": "auto",
                            "image_url": "data:image/png;base64,iVBORw0KGgo="
                        }
                    ]
                }
            ],
            "tools": [{
                "type": "function",
                "name": "inspect",
                "description": "Inspect an image",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                },
                "strict": false
            }],
            "stream": true,
            "store": false,
            "max_output_tokens": 128,
            "temperature": 0.2,
            "reasoning": {"effort": "high", "summary": "concise"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "required",
            "service_tier": "priority"
        })
    );
}

#[tokio::test]
async fn cancels_an_openai_request_before_response_headers() {
    let server = serve([Reply::pending()]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| options.stream.cancellation = cancellation.clone());
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    cancellation.cancel();

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("request cancelled"));
    assert!(error.content.is_empty());
}

#[tokio::test]
async fn cancels_while_reading_an_openai_error_body() {
    let server = serve([Reply::open_json(
        500,
        json!({"error": {"message": "unfinished"}}),
    )])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| options.stream.cancellation = cancellation.clone());
    let request = tokio::spawn(async move { events(&model, &context, &options).await });
    server.wait_for_requests(1).await;
    tokio::task::yield_now().await;

    cancellation.cancel();

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("request cancelled"));
    assert!(error.content.is_empty());
    server.requests().await;
}

#[tokio::test]
async fn cancels_an_active_openai_stream_with_partial_content() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_cancel\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_cancel\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::open_sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = options(|options| options.stream.cancellation = cancellation.clone());
    let mut response = openai::stream(&model, &context, &options);

    while !matches!(
        response.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ) {}
    cancellation.cancel();

    match response.next().await {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, StopReason::Aborted);
            assert_eq!(error.stop_reason, StopReason::Aborted);
            assert_eq!(error.raw_stop_reason.as_deref(), Some("cancelled"));
            assert_eq!(error.error_message.as_deref(), Some("request cancelled"));
            assert_eq!(error.content, [text("Visible", None)]);
        }
        event => panic!("unexpected cancellation event: {event:?}"),
    }
}

#[tokio::test]
async fn times_out_an_openai_request_before_response_headers() {
    let server = serve([Reply::pending()]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_millis(50));
    });
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(error.stop_reason, StopReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider timed out during Overall")
    );
    assert!(error.content.is_empty());
}

#[tokio::test]
async fn times_out_while_reading_an_openai_error_body() {
    let server = serve([Reply::open_json(
        500,
        json!({"error": {"message": "unfinished"}}),
    )])
    .await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_secs(5));
    });
    let request = tokio::spawn(async move { events(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    tokio::task::yield_now().await;
    tokio::time::pause();
    tokio::time::advance(std::time::Duration::from_secs(5)).await;

    let events = request.await.unwrap();
    let error = failed(&events);
    assert_eq!(
        error.error_message.as_deref(),
        Some("provider timed out during Overall")
    );
    assert!(error.content.is_empty());
    server.requests().await;
}

#[tokio::test(start_paused = true)]
async fn times_out_an_openai_stream_before_its_first_event() {
    let server = serve([Reply::open_sse(": keepalive\n\n")]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_secs(5));
    });
    let mut response = openai::stream(&model, &context, &options);
    let next = tokio::spawn(async move { response.next().await });

    tokio::task::yield_now().await;
    tokio::time::advance(std::time::Duration::from_secs(5)).await;

    match next.await.unwrap() {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, StopReason::Error);
            assert_eq!(error.stop_reason, StopReason::Error);
            assert_eq!(error.raw_stop_reason, None);
            assert_eq!(
                error.error_message.as_deref(),
                Some("provider timed out during Overall")
            );
            assert!(error.content.is_empty());
        }
        event => panic!("unexpected timeout event: {event:?}"),
    }
}

#[tokio::test]
async fn enforces_an_overall_openai_stream_deadline() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_overall\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_overall\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Visible\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::open_sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = options(|options| {
        options.stream.timeout = Some(std::time::Duration::from_secs(5));
    });
    let mut response = openai::stream(&model, &context, &options);

    while !matches!(
        response.next().await,
        Some(AssistantMessageEvent::TextDelta { .. })
    ) {}
    tokio::time::pause();
    let next = tokio::spawn(async move { response.next().await });
    tokio::time::advance(std::time::Duration::from_secs(5)).await;

    match next.await.unwrap() {
        Some(AssistantMessageEvent::Error { reason, error }) => {
            assert_eq!(reason, StopReason::Error);
            assert_eq!(error.stop_reason, StopReason::Error);
            assert_eq!(error.raw_stop_reason.as_deref(), Some("timeout.overall"));
            assert_eq!(error.content, [text("Visible", None)]);
        }
        event => panic!("unexpected timeout event: {event:?}"),
    }
}

#[tokio::test]
async fn encodes_openai_prompt_cache_retention_and_session_keys() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_cache\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse), Reply::sse(sse)]).await;
    let model = model(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let session_id = "🦀".repeat(70);
    let long = options(|options| {
        options.stream.session_id = Some(session_id.clone());
        options.stream.cache_retention = CacheRetention::Long;
    });
    let first = events(&model, &context, &long).await;
    done(&first);

    let disabled = options(|options| {
        options.stream.session_id = Some(session_id.clone());
        options.stream.cache_retention = CacheRetention::None;
    });
    let second = events(&model, &context, &disabled).await;
    done(&second);

    let requests = server.requests().await;
    let long_body: Value =
        serde_json::from_str(requests[0].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(long_body["prompt_cache_key"], "🦀".repeat(64));
    assert_eq!(long_body["prompt_cache_retention"], "24h");
    let disabled_body: Value =
        serde_json::from_str(requests[1].split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert!(disabled_body.get("prompt_cache_key").is_none());
    assert!(disabled_body.get("prompt_cache_retention").is_none());
    assert_eq!(
        disabled_body["prompt_cache_options"],
        json!({"mode": "explicit"})
    );
}

#[tokio::test]
async fn preserves_openai_http_error_and_response_metadata() {
    let failure = Reply::json(
        429,
        json!({"error": {"code": "rate_limit_exceeded", "message": "Too many requests"}}),
    )
    .with_header("x-request-id", "req_failure")
    .with_header("retry-after-ms", "250")
    .with_header("x-ratelimit-limit-requests", "100")
    .with_header("x-ratelimit-remaining-requests", "0")
    .with_header("x-ratelimit-reset-requests", "1s");
    let failure_server = serve([failure]).await;
    let failure_model = model(&failure_server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let responses = Arc::new(Mutex::new(Vec::new()));
    let captured = responses.clone();
    let mut options = options(|_| {});
    options.stream.on_response = Some(ResponseHook::new(move |response, _| {
        let captured = captured.clone();
        async move {
            captured.lock().unwrap().push(response);
            Ok(())
        }
    }));

    let failure_events = events(&failure_model, &context, &options).await;
    assert_eq!(
        failed(&failure_events).error_message.as_deref(),
        Some("provider returned HTTP 429: Too many requests")
    );
    let failure_response = responses.lock().unwrap()[0].clone();
    assert_eq!(failure_response.status, 429);
    assert_eq!(
        failure_response
            .headers
            .get("x-request-id")
            .map(String::as_str),
        Some("req_failure")
    );
    assert_eq!(
        failure_response
            .headers
            .get("retry-after-ms")
            .map(String::as_str),
        Some("250")
    );
    assert_eq!(
        failure_response
            .headers
            .get("x-ratelimit-remaining-requests")
            .map(String::as_str),
        Some("0")
    );
    failure_server.requests().await;

    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_metadata\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let success = Reply::sse(sse)
        .with_header("x-request-id", "req_success")
        .with_header("x-ratelimit-limit-tokens", "1000")
        .with_header("x-ratelimit-remaining-tokens", "900")
        .with_header("x-ratelimit-reset-tokens", "2s");
    let success_server = serve([success]).await;
    let success_model = model(&success_server.base_url);
    let success_events = events(&success_model, &context, &options).await;
    done(&success_events);
    let success_response = responses.lock().unwrap()[1].clone();
    assert_eq!(success_response.status, 200);
    assert_eq!(
        success_response
            .headers
            .get("x-request-id")
            .map(String::as_str),
        Some("req_success")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-ratelimit-limit-tokens")
            .map(String::as_str),
        Some("1000")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-ratelimit-remaining-tokens")
            .map(String::as_str),
        Some("900")
    );
    assert_eq!(
        success_response
            .headers
            .get("x-ratelimit-reset-tokens")
            .map(String::as_str),
        Some("2s")
    );
    success_server.requests().await;
}

fn model(base_url: &str) -> ds_ai::Model {
    let mut model = builtin_model("openai", "gpt-5.6-sol").unwrap();
    model.id = "gpt-5.6".into();
    model.name = "gpt-5.6".into();
    model.base_url = base_url.into();
    model
}

fn options(configure: impl FnOnce(&mut OpenAiResponsesOptions)) -> OpenAiResponsesOptions {
    let mut options = OpenAiResponsesOptions {
        stream: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    configure(&mut options);
    options
}

async fn events(
    model: &ds_ai::Model,
    context: &Context,
    options: &OpenAiResponsesOptions,
) -> Vec<AssistantMessageEvent> {
    openai::stream(model, context, options).collect().await
}

fn text(value: &str, signature: Option<(&str, Option<&str>)>) -> AssistantContent {
    AssistantContent::Text(TextContent {
        text: value.into(),
        text_signature: signature.map(|(id, phase)| {
            json!({
                "v": 1,
                "id": id,
                "phase": phase,
            })
            .to_string()
        }),
    })
}

fn thinking(value: &str, id: Option<&str>, encrypted: Option<&str>) -> AssistantContent {
    AssistantContent::Thinking(ThinkingContent {
        thinking: value.into(),
        thinking_signature: id.map(|id| {
            let mut signature = json!({
                "type": "reasoning",
                "id": id,
                "summary": [{"type": "summary_text", "text": value}],
            });
            if let Some(encrypted) = encrypted {
                signature["encrypted_content"] = encrypted.into();
            }
            signature.to_string()
        }),
        redacted: None,
    })
}

fn done(events: &[AssistantMessageEvent]) -> &AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Done { message, .. }) => message,
        _ => panic!("stream did not complete"),
    }
}

fn failed(events: &[AssistantMessageEvent]) -> &AssistantMessage {
    match events.last() {
        Some(AssistantMessageEvent::Error { error, .. }) => error,
        event => panic!("stream did not fail: {event:?}"),
    }
}
