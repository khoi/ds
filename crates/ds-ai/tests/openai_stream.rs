use ds_ai::{Context, Event, Message, openai};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
    sync::oneshot,
};

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
    let (base_url, request) = serve_once(sse).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(base_url);
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

    let request = request.await.unwrap();
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
    let (base_url, request) = serve_once(sse).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(base_url);
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
    request.await.unwrap();
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
    let (base_url, request) = serve_chunks(chunks).await;
    let model = openai::Model::new("gpt-5.6").with_base_url(base_url);
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
    request.await.unwrap();
}

async fn serve_once(sse: String) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (request_sender, request_receiver) = oneshot::channel();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let header_end = loop {
            let mut bytes = [0; 1024];
            let count = socket.read(&mut bytes).await.unwrap();
            request.extend_from_slice(&bytes[..count]);
            if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap();
        while request.len() < header_end + content_length {
            let mut bytes = [0; 1024];
            let count = socket.read(&mut bytes).await.unwrap();
            request.extend_from_slice(&bytes[..count]);
        }
        request_sender
            .send(String::from_utf8(request).unwrap())
            .unwrap();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            sse.len(),
            sse
        );
        socket.write_all(response.as_bytes()).await.unwrap();
    });

    (format!("http://{address}"), request_receiver)
}

async fn serve_chunks(chunks: Vec<Vec<u8>>) -> (String, oneshot::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (request_sender, request_receiver) = oneshot::channel();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let header_end = loop {
            let mut bytes = [0; 1024];
            let count = socket.read(&mut bytes).await.unwrap();
            request.extend_from_slice(&bytes[..count]);
            if let Some(index) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break index + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length: ")
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap();
        while request.len() < header_end + content_length {
            let mut bytes = [0; 1024];
            let count = socket.read(&mut bytes).await.unwrap();
            request.extend_from_slice(&bytes[..count]);
        }
        request_sender
            .send(String::from_utf8(request).unwrap())
            .unwrap();
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
        for chunk in chunks {
            socket
                .write_all(format!("{:X}\r\n", chunk.len()).as_bytes())
                .await
                .unwrap();
            socket.write_all(&chunk).await.unwrap();
            socket.write_all(b"\r\n").await.unwrap();
        }
        socket.write_all(b"0\r\n\r\n").await.unwrap();
    });

    (format!("http://{address}"), request_receiver)
}
