public class StreamedOnlySourcesPipelinePartitionExpression extends RuleExpression { public StreamedOnlySourcesPipelinePartitionExpression() { super( new NoGroupJoinTapExpressionGraph(), new StreamedOnlySourcesExpressionGraph() ); } }