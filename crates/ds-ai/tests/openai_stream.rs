mod support;

use ds_ai::{Context, Event, Message, openai};
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

    assert_eq!(
        events,
        vec![
            Ok(Event::TextDelta {
                content_index: 0,
                delta: "Hello".into(),
            }),
            Ok(Event::Done(ds_ai::Response {
                id: Some("resp_1".into()),
                content: vec![ds_ai::Content::Text("Hello".into())],
                usage: ds_ai::Usage {
                    input: 4,
                    output: 1,
                    cache_read: 0,
                    cache_write: 0,
                    reasoning: 0,
                },
            })),
        ]
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

    assert_eq!(
        events,
        vec![
            Ok(Event::TextDelta {
                content_index: 0,
                delta: "Hel".into(),
            }),
            Err(ds_ai::Error::IncompleteStream {
                partial: ds_ai::Response {
                    id: Some("resp_partial".into()),
                    content: vec![ds_ai::Content::Text("Hel".into())],
                    usage: ds_ai::Usage::default(),
                },
            }),
        ]
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
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key");

    let events = openai::stream(&model, &context, &options)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(
        events,
        vec![
            Ok(Event::TextDelta {
                content_index: 0,
                delta: "Hé".into(),
            }),
            Ok(Event::Done(ds_ai::Response {
                id: Some("resp_chunks".into()),
                content: vec![ds_ai::Content::Text("Hé".into())],
                usage: ds_ai::Usage {
                    input: 3,
                    output: 1,
                    cache_read: 0,
                    cache_write: 0,
                    reasoning: 0,
                },
            })),
        ]
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

    assert_eq!(
        events,
        vec![Ok(Event::Done(ds_ai::Response {
            id: Some("resp_retry".into()),
            content: Vec::new(),
            usage: ds_ai::Usage::default(),
        }))]
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
async fn rejects_an_openai_retry_delay_above_the_limit() {
    let server = serve([Reply::json(
        429,
        json!({"error": {"type": "rate_limit_error", "message": "retry later"}}),
    )
    .with_header("retry-after", "61")])
    .await;
    let model = openai::Model::new("gpt-5.6").with_base_url(&server.base_url);
    let context = Context::new([Message::user("Hello")]);
    let options = openai::Options::new("test-key").with_max_retries(1);

    let error = match openai::stream(&model, &context, &options).await {
        Ok(_) => panic!("retry delay was accepted"),
        Err(error) => error,
    };

    assert_eq!(
        error,
        ds_ai::Error::RetryDelayExceeded {
            requested: std::time::Duration::from_secs(61),
            maximum: std::time::Duration::from_secs(60),
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
