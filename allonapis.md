# API & Communication Protocols - Brief Notes

## REST
- Architectural style using HTTP methods (GET, POST, PUT, DELETE)
- JSON/XML data format, stateless, cacheable
- Simple, widely adopted for web APIs
- Supports HATEOAS (hypermedia), uses standard HTTP status codes
- Common pattern: RESTful APIs with resource-based URLs

## SOAP
- Protocol for structured data exchange
- XML-based with built-in security and error handling
- Heavyweight, used in enterprise/banking

## GraphQL
- Query language letting clients request specific data
- Prevents over-fetching/under-fetching
- Single endpoint for flexible queries
- Single endpoint (vs multiple in REST), strongly typed schema
- Subscriptions for real-time updates

## Webhooks
- Event-triggered HTTP callbacks
- Server-to-client notifications (one-way)
- Real-time updates for external systems

## WebSocket
- Persistent, bidirectional connection
- Full-duplex real-time communication
- Ideal for chat, gaming, live apps
- Starts as HTTP, upgrades to WS protocol
- Lower overhead than HTTP polling

## WebRTC
- Peer-to-peer real-time communication for video/audio/data
- Browser-based, low latency
- Direct connection between clients without server

## JSON-RPC / XML-RPC
- Lightweight remote procedure call protocols
- Simpler alternatives to SOAP
- Request-response pattern with method calls

## SSE (Server-Sent Events)
- Server pushes updates to client over HTTP
- One-way real-time communication
- Built-in reconnection support

## gRPC
- High-performance RPC framework using HTTP/2
- Protocol Buffers for serialization
- Supports streaming, language-agnostic
- 4 streaming modes (unary, server, client, bidirectional)
- Much faster than REST for microservices

## MQTT
- Lightweight publish-subscribe protocol
- Designed for IoT/low-bandwidth
- Small packet size, efficient
- QoS levels (0,1,2), topic-based routing
- Supports last will & testament

## AMQP
- Advanced message queuing protocol
- Reliable messaging with queues/exchanges
- Enterprise message brokering

## EDA (Event-Driven Architecture)
- Components react to emitted events
- Loose coupling, asynchronous
- Scalable distributed systems

## EDI (Electronic Data Interchange)
- Standard business document exchange
- Structured format for invoices/orders
- B2B communication

## OData
- REST-based protocol for querying/updating data
- Standardized URL conventions
- Supports filtering, sorting, and paging

## STOMP
- Simple text-based messaging protocol
- Often used with message brokers
- Easy to implement and debug

## Long Polling
- HTTP technique where server holds request until data available
- Between REST and WebSocket in functionality
- Reduces polling overhead

## Comet
- Umbrella term for server push techniques
- Includes long polling, streaming, and other methods
- Enables real-time web applications

## AsyncAPI
- Specification for event-driven/async APIs
- Like OpenAPI for REST but for asynchronous communication
- Describes message-driven architectures

## Summary
- **Request-Response:** REST, SOAP, JSON-RPC, XML-RPC
- **Flexible Queries:** GraphQL, OData
- **Real-time:** Webhooks, SSE, WebSocket, WebRTC
- **Efficient Comm:** gRPC, MQTT, AMQP, STOMP
- **Patterns:** EDA (architecture), EDI (business data), AsyncAPI (specification)
- **Server Push:** Comet, Long Polling, SSE

## Comparison Categories
- **Sync vs Async:** REST/SOAP/GraphQL (sync), Webhooks/MQTT/AMQP (async)
- **Human vs Machine:** REST (human-readable), gRPC/MQTT (binary, efficient)
- **Communication Style:** Request-Response vs Event-Driven vs Streaming

