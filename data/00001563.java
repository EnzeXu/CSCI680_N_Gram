public class StreamedAccumulatedTapsPipelinePartitioner extends ExpressionRulePartitioner { public StreamedAccumulatedTapsPipelinePartitioner ( ) { super ( PartitionPipelines , new StreamedAccumulatedTapsPipelinePartitionExpression ( ) , new ElementAnnotation ( ElementCapture . Primary , StreamMode . Streamed ) , new ElementAnnotation ( ElementCapture . Include , StreamMode . Accumulated ) ) ; } }