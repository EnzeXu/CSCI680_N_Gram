public class StreamedSelfJoinSourcesPipelinePartitioner extends ExpressionRulePartitioner { public StreamedSelfJoinSourcesPipelinePartitioner() { super( PartitionPipelines, new StreamedSelfJoinSourcesPipelinePartitionExpression(), new ElementAnnotation( ElementCapture.Primary, StreamMode.Streamed ) ); } }