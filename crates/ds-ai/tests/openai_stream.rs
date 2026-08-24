mod support;

use ds_ai::{
    Context, Event, InputContent, Message, StopReason, Tool, ToolCall, ToolResult, openai,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use support::{Reply, serve};

#[tokio::test]
async fn streams_openai_text_until_the_provider_completes() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\",\"annotations\":[]}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":4,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":1,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":5}}}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 2);
    assert_eq!(
        events.first(),
        Some(&Ok(Event::TextDelta {
            content_index: 0,
            delta: "Hello".into(),
        }))
    );
    let response = done(&events);
    assert_eq!(response.id.as_deref(), Some("resp_1"));
    assert_eq!(response.content, [ds_ai::Content::Text("Hello".into())]);
    assert_eq!(
        response.usage,
        ds_ai::Usage {
            input: 4,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            reasoning: 0,
        }
    );

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
            "store": false
        })
    );
}

#[tokio::test]
async fn rejects_an_openai_stream_that_ends_without_a_terminal_event() {
    let sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_partial\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_partial\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"in_progress\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hel\"}\n\n",
    ]
    .concat();
    let server = serve([Reply::sse(sse)]).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 2);
    assert_eq!(
        events.first(),
        Some(&Ok(Event::TextDelta {
            content_index: 0,
            delta: "Hel".into(),
        }))
    );
    let partial = incomplete(&events);
    assert_eq!(partial.id.as_deref(), Some("resp_partial"));
    assert_eq!(partial.content, [ds_ai::Content::Text("Hel".into())]);
    assert_eq!(partial.usage, ds_ai::Usage::default());
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 2);
    assert_eq!(
        events.first(),
        Some(&Ok(Event::TextDelta {
            content_index: 0,
            delta: "Hé".into(),
        }))
    );
    let response = done(&events);
    assert_eq!(response.id.as_deref(), Some("resp_chunks"));
    assert_eq!(response.content, [ds_ai::Content::Text("Hé".into())]);
    assert_eq!(
        response.usage,
        ds_ai::Usage {
            input: 3,
            output: 1,
            cache_read: 0,
            cache_write: 0,
            reasoning: 0,
        }
    );
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key").with_max_retries(1);

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 1);
    let response = done(&events);
    assert_eq!(response.id.as_deref(), Some("resp_retry"));
    assert!(response.content.is_empty());
    assert_eq!(response.usage, ds_ai::Usage::default());
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
        let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
        let context = Context::new([Message::user("Hello")]);
        let options = openai::Options::new("test-key").with_max_retries(1);
        let task = tokio::spawn(async move {
            openai::stream(&model, &context, &options)
                .await
                .unwrap()
                .collect::<Vec<_>>()
                .await
        });

        server.wait_for_requests(1).await;
        tokio::time::advance(std::time::Duration::from_millis(delay_ms - 1)).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(server.request_count(), 1);

        tokio::time::advance(std::time::Duration::from_millis(1)).await;
        let events = task.await.unwrap();
        assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key").with_max_retries(1);
    let task = tokio::spawn(async move {
        openai::stream(&model, &context, &options)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });

    server.wait_for_requests(1).await;
    tokio::time::advance(std::time::Duration::from_secs(58)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_secs(3)).await;
    let events = task.await.unwrap();
    assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let cancellation = tokio_util::sync::CancellationToken::new();
    let options = openai::Options::new("test-key")
        .with_max_retries(1)
        .with_cancellation(cancellation.clone());
    let task = tokio::spawn(async move { openai::stream(&model, &context, &options).await });

    server.wait_for_requests(1).await;
    cancellation.cancel();
    let error = match task.await.unwrap() {
        Ok(_) => panic!("retry wait did not cancel"),
        Err(error) => error,
    };

    assert_eq!(error, ds_ai::Error::Cancelled);
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
        let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
        let context = Context::new([Message::user("Hello")]);
        let options = openai::Options::new("test-key").with_max_retries(1);

        let result = openai::stream(&model, &context, &options).await;

        if should_retry {
            let events = result.unwrap().collect::<Vec<_>>().await;
            assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
            assert_eq!(server.requests().await.len(), 2);
        } else {
            assert!(matches!(
                result,
                Err(ds_ai::Error::Provider {
                    status: actual,
                    ..
                }) if actual == status
            ));
            assert_eq!(server.request_count(), 1);
        }
    }
}

#[tokio::test(start_paused = true)]
async fn retries_openai_network_failures_before_streaming_starts() {
    let completed = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_network\",\"usage\":{\"input_tokens\":0,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::disconnect(), Reply::sse(completed)]).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key").with_max_retries(1);

    let task = tokio::spawn(async move {
        openai::stream(&model, &context, &options)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });

    server.wait_for_requests(1).await;
    tokio::time::advance(std::time::Duration::from_millis(374)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(126)).await;
    let events = task.await.unwrap();
    assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key")
        .with_max_retries(1)
        .with_max_retry_delay(Some(std::time::Duration::from_secs(1)));

    let error = match openai::stream(&model, &context, &options).await {
        Ok(_) => panic!("retry delay was accepted"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        ds_ai::Error::RetryDelayExceeded {
            requested: std::time::Duration::from_secs(2),
            maximum: std::time::Duration::from_secs(1),
        }
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key").with_max_retries(1);
    let task = tokio::spawn(async move {
        openai::stream(&model, &context, &options)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });

    server.wait_for_requests(1).await;
    tokio::time::advance(std::time::Duration::from_millis(374)).await;
    for _ in 0..10 {
        tokio::task::yield_now().await;
    }
    assert_eq!(server.request_count(), 1);

    tokio::time::advance(std::time::Duration::from_millis(126)).await;
    let events = task.await.unwrap();
    assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key").with_max_retries(1);
    let task = tokio::spawn(async move {
        openai::stream(&model, &context, &options)
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
    });

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
    assert!(matches!(events.as_slice(), [Ok(Event::Done(_))]));
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(events.len(), 3);
    assert_eq!(
        events.first(),
        Some(&Ok(Event::ReasoningDelta {
            content_index: 0,
            delta: "Need answer.".into(),
        }))
    );
    assert_eq!(
        events.get(1),
        Some(&Ok(Event::TextDelta {
            content_index: 1,
            delta: "Hello".into(),
        }))
    );
    let response = done(&events);
    assert_eq!(response.id.as_deref(), Some("resp_reasoning"));
    assert_eq!(
        response.content,
        [
            ds_ai::Content::Reasoning("Need answer.".into()),
            ds_ai::Content::Text("Hello".into()),
        ]
    );
    assert_eq!(
        response.usage,
        ds_ai::Usage {
            input: 5,
            output: 4,
            cache_read: 0,
            cache_write: 0,
            reasoning: 3,
        }
    );
}

#[tokio::test]
async fn replays_serialized_openai_reasoning_and_message_items() {
    let first_sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_first\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[]}}\n\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"Need answer.\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"rs_replay\",\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Need answer.\"}],\"encrypted_content\":\"encrypted\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"id\":\"msg_replay\",\"type\":\"message\",\"content\":[]}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"Hello\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"id\":\"msg_replay\",\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}],\"phase\":\"final_answer\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_first\",\"usage\":{\"input_tokens\":1,\"input_tokens_details\":{},\"output_tokens\":2,\"output_tokens_details\":{\"reasoning_tokens\":1}}}}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = openai::Model::new("gpt-5.6").with_base_url(&first_server.base_url);
    let first_context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key");
    let first_events = openai::stream(&first_model, &first_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let response =
        serde_json::from_value(serde_json::to_value(done(&first_events)).unwrap()).unwrap();
    first_server.requests().await;

    let second_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = openai::Model::new("gpt-5.6").with_base_url(&second_server.base_url);
    let second_context = Context::new([Message::assistant(response), Message::user("Continue")]);

    openai::stream(&second_model, &second_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

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
async fn streams_and_replays_openai_tool_calls() {
    let first_sse = [
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"fc_edit\",\"type\":\"function_call\",\"call_id\":\"call_edit\",\"name\":\"edit\",\"arguments\":\"\"}}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"path\\\":\\\"README.md\\\"\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\",\\\"content\\\":\\\"updated\\\"}\"}\n\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"path\\\":\\\"README.md\\\",\\\"content\\\":\\\"updated\\\"}\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"fc_edit\",\"type\":\"function_call\",\"call_id\":\"call_edit\",\"name\":\"edit\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\",\\\"content\\\":\\\"updated\\\"}\"}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":2,\"output_tokens_details\":{}}}}\n\n",
    ]
    .concat();
    let first_server = serve([Reply::sse(first_sse)]).await;
    let first_model = openai::Model::new("gpt-5.6").with_base_url(&first_server.base_url);
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
    )]);
    let options = openai::Options::new("test-key");

    let first_events = openai::stream(&first_model, &first_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        first_events.first(),
        Some(&Ok(Event::ToolCallDelta {
            content_index: 0,
            delta: "{\"path\":\"README.md\"".into(),
        }))
    );
    assert_eq!(
        first_events.get(1),
        Some(&Ok(Event::ToolCallDelta {
            content_index: 0,
            delta: ",\"content\":\"updated\"}".into(),
        }))
    );
    let response = done(&first_events);
    assert_eq!(
        response.content,
        [ds_ai::Content::ToolCall(ToolCall {
            id: "call_edit".into(),
            name: "edit".into(),
            arguments: json!({"path": "README.md", "content": "updated"}),
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
            "strict": false
        }])
    );

    let restored: ds_ai::Response =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    let second_sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_second\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let second_server = serve([Reply::sse(second_sse)]).await;
    let second_model = openai::Model::new("gpt-5.6").with_base_url(&second_server.base_url);
    let second_context = Context::new([Message::assistant(restored), Message::user("Continue")]);

    openai::stream(&second_model, &second_context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

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
                "arguments": "{\"content\":\"updated\",\"path\":\"README.md\"}"
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue"}]
            }
        ])
    );
}

#[tokio::test]
async fn sends_openai_tool_result_text_images_and_empty_output() {
    let sse = "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_result\",\"usage\":{\"input_tokens\":3,\"input_tokens_details\":{},\"output_tokens\":0,\"output_tokens_details\":{}}}}\n\n";
    let server = serve([Reply::sse(sse)]).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([
        Message::user("Inspect the image"),
        Message::tool_result(ToolResult::new(
            "call_image",
            "inspect",
            [
                InputContent::text("A red circle"),
                InputContent::image("image/png", "iVBORw0KGgo="),
            ],
        )),
        Message::tool_result(ToolResult::new(
            "call_empty",
            "noop",
            [InputContent::text("")],
        )),
    ]);
    let options = openai::Options::new("test-key");

    openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    let response = done(&events);
    assert_eq!(response.stop_reason, StopReason::Length);
    assert_eq!(
        response.raw_stop_reason.as_deref(),
        Some("incomplete.max_output_tokens")
    );
    assert_eq!(response.content, [ds_ai::Content::Text("Partial".into())]);
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    match events.last() {
        Some(Err(ds_ai::Error::Response {
            code,
            message,
            partial,
        })) => {
            assert_eq!(code, &None);
            assert_eq!(message, "Response incomplete: content_filter");
            assert_eq!(partial.stop_reason, StopReason::Error);
            assert_eq!(
                partial.raw_stop_reason.as_deref(),
                Some("incomplete.content_filter")
            );
            assert_eq!(partial.content, [ds_ai::Content::Text("Visible".into())]);
        }
        event => panic!("unexpected terminal event: {event:?}"),
    }
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Think")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    match events.last() {
        Some(Err(ds_ai::Error::Response {
            code,
            message,
            partial,
        })) => {
            assert_eq!(code.as_deref(), Some("server_error"));
            assert_eq!(message, "boom");
            assert_eq!(partial.id.as_deref(), Some("resp_failed"));
            assert_eq!(partial.stop_reason, StopReason::Error);
            assert_eq!(partial.raw_stop_reason.as_deref(), Some("failed"));
            assert_eq!(
                partial.content,
                [ds_ai::Content::Reasoning("Partial thought".into())]
            );
        }
        event => panic!("unexpected terminal event: {event:?}"),
    }
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Write")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    match events.last() {
        Some(Err(ds_ai::Error::Response {
            code,
            message,
            partial,
        })) => {
            assert_eq!(code.as_deref(), Some("rate_limit_exceeded"));
            assert_eq!(message, "slow down");
            assert_eq!(partial.stop_reason, StopReason::Error);
            assert_eq!(partial.raw_stop_reason.as_deref(), Some("error"));
            assert_eq!(partial.content, [ds_ai::Content::Text("Visible".into())]);
        }
        event => panic!("unexpected terminal event: {event:?}"),
    }
}

fn done(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Ok(Event::Done(response))) => response,
        _ => panic!("stream did not complete"),
    }
}

fn incomplete(events: &[Result<Event, ds_ai::Error>]) -> &ds_ai::Response {
    match events.last() {
        Some(Err(ds_ai::Error::IncompleteStream { partial })) => partial,
        _ => panic!("stream was not incomplete"),
    }
}
